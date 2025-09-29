import argparse
import json
import math
import random
import sys
from typing import List, Sequence

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import TransformerEncoder, generate_causal_mask
try:
    # When installed as a package with examples available as a module
    from examples.chatbot import Chatbot  # type: ignore
except ImportError:
    # When running this script directly (python examples/train_then_chatbot.py)
    from chatbot import Chatbot


def _softmax_vec(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s == 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [e / s for e in exps]


def _init_matrix(rows: int, cols: int, rng: random.Random, scheme: str = "xavier_uniform") -> List[List[float]]:
    s = scheme.lower()
    if s == "zeros":
        return [[0.0 for _ in range(cols)] for _ in range(rows)]
    elif s in {"uniform", "xavier_uniform"}:
        if s == "uniform":
            limit = 1.0 / math.sqrt(cols if cols > 0 else 1)
        else:
            limit = math.sqrt(6.0 / float((rows + cols) if (rows + cols) > 0 else 1))
        return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]
    else:
        raise ValueError(f"Unknown init scheme: {scheme}")


def _matvec(A: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    m, n = len(A), (len(A[0]) if A else 0)
    if n != len(x):
        raise ValueError(f"matvec mismatch: {n} vs {len(x)}")
    out = [0.0] * m
    for i in range(m):
        s = 0.0
        Ai = A[i]
        for k in range(n):
            s += Ai[k] * x[k]
        out[i] = s
    return out


def _zero_like_matrix(M: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[0.0] * (len(M[0]) if M else 0) for _ in range(len(M))]


def _zero_like_vector(v: Sequence[float]) -> List[float]:
    return [0.0 for _ in range(len(v))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a next-token head and launch the console chatbot.")
    # Shared model/tokenizer args
    parser.add_argument("--corpus", type=str, default="allen.txt", help="Path to corpus text file.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to BPE merges.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to BPE vocab.")
    parser.add_argument("--dim", type=int, default=32, help="Model/embedding dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ff", type=int, default=64, help="FFN hidden size.")
    parser.add_argument("--add_pe", action="store_true", help="Add sinusoidal positional encodings.")
    # Training args
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length for training windows.")
    parser.add_argument("--stride", type=int, default=32, help="Stride between training windows.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of passes over the data.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--adam", action="store_true", help="Use Adam optimizer instead of SGD.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--save_head", type=str, default="head.json", help="Path to save trained head JSON.")
    # Chatbot args
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Generation length per turn.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Softmax temperature.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument("--stop_token", type=str, default="", help="Optional token that stops generation, e.g., '<eos>'.")
    parser.add_argument("--system", type=str, default="", help="Optional system prompt that prefixes the conversation.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Tokenizer, embeddings, encoder
    tok = BPETokenizer(); tok.load(args.merges, args.vocab)
    emb = EmbeddingLayer.from_vocab_file(args.vocab, dim=args.dim, seed=42)
    vocab_size = emb.vocab_size
    encoder = TransformerEncoder(num_layers=args.layers, dim=args.dim, num_heads=args.heads, ff_hidden=args.ff, seed=args.seed)
    pe = SinusoidalPositionalEncoding(dim=args.dim) if args.add_pe else None

    # Prepare data
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus_text = f.read()
    tokens = tok.encode(corpus_text)
    ids_all = emb.tokens_to_ids(tokens)
    if len(ids_all) < args.seq_len + 1:
        raise ValueError("Corpus too small for seq_len; provide more text or reduce --seq_len.")

    # Training windows
    def prepare_windows(ids, L, stride):
        out = []
        i = 0
        N = len(ids)
        while i + L + 1 <= N:
            out.append((i, i + L))
            i += max(1, stride)
        return out

    windows = prepare_windows(ids_all, args.seq_len, args.stride)
    if not windows:
        raise ValueError("No training windows prepared. Adjust --seq_len/--stride.")

    # Head params
    W_out = _init_matrix(vocab_size, args.dim, rng, "xavier_uniform")
    b_out = [0.0] * vocab_size
    mW = _zero_like_matrix(W_out); vW = _zero_like_matrix(W_out)
    mb = _zero_like_vector(b_out); vb = _zero_like_vector(b_out)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step_t = 0

    # Train loop
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(windows)
        running_loss = 0.0
        running_cnt = 0
        for s, e in windows:
            ids_win = ids_all[s : e + 1]
            targets = ids_all[s + 1 : e + 2]
            if len(targets) > len(ids_win):
                targets = targets[: len(ids_win)]
            elif len(targets) < len(ids_win):
                ids_win = ids_win[: len(targets)]
            L = len(ids_win)
            if L < 2:
                continue

            X = emb.embed_ids(ids_win)
            if pe is not None:
                X = pe.add_to(X, offset=0)
            H = encoder(X, mask=generate_causal_mask(L))

            dW = [[0.0] * args.dim for _ in range(vocab_size)]
            db = [0.0] * vocab_size
            loss = 0.0
            cnt = 0
            for t in range(L - 1):
                x_t = H[t]
                y_id = targets[t]
                logits = _matvec(W_out, x_t)
                for i in range(vocab_size):
                    logits[i] += b_out[i]
                probs = _softmax_vec(logits)
                p_y = probs[y_id] if 0 <= y_id < vocab_size else 0.0
                loss += -math.log(p_y) if p_y > 0 else 100.0
                cnt += 1
                grad = [p for p in probs]
                if 0 <= y_id < vocab_size:
                    grad[y_id] -= 1.0
                for v in range(vocab_size):
                    gv = grad[v]
                    if gv == 0.0:
                        continue
                    db[v] += gv
                    row = dW[v]
                    for d in range(args.dim):
                        row[d] += x_t[d] * gv

            if cnt == 0:
                continue
            running_loss += loss
            running_cnt += cnt
            scale = 1.0 / cnt
            for v in range(vocab_size):
                db[v] *= scale
                row = dW[v]
                for d in range(args.dim):
                    row[d] *= scale

            if args.adam:
                step_t += 1
                for v in range(vocab_size):
                    for d in range(args.dim):
                        mW[v][d] = beta1 * mW[v][d] + (1.0 - beta1) * dW[v][d]
                        vW[v][d] = beta2 * vW[v][d] + (1.0 - beta2) * (dW[v][d] * dW[v][d])
                        mhat = mW[v][d] / (1.0 - beta1 ** step_t)
                        vhat = vW[v][d] / (1.0 - beta2 ** step_t)
                        W_out[v][d] -= args.lr * mhat / (math.sqrt(vhat) + eps)
                    mb[v] = beta1 * mb[v] + (1.0 - beta1) * db[v]
                    vb[v] = beta2 * vb[v] + (1.0 - beta2) * (db[v] * db[v])
                    mhat_b = mb[v] / (1.0 - beta1 ** step_t)
                    vhat_b = vb[v] / (1.0 - beta2 ** step_t)
                    b_out[v] -= args.lr * mhat_b / (math.sqrt(vhat_b) + eps)
            else:
                for v in range(vocab_size):
                    for d in range(args.dim):
                        W_out[v][d] -= args.lr * dW[v][d]
                    b_out[v] -= args.lr * db[v]

        if running_cnt > 0:
            avg = running_loss / running_cnt
            ppl = math.exp(min(20.0, avg))
            print(f"Epoch {epoch} done - avg loss/token {avg:.4f}  perplexity {ppl:.2f}")

    # Save head JSON
    head_path = args.save_head or "head.json"
    with open(head_path, "w", encoding="utf-8") as f:
        json.dump({"dim": args.dim, "vocab_size": vocab_size, "W_out": W_out, "b_out": b_out}, f)
    print(f"Saved head to {head_path}")

    # Launch chatbot
    bot = Chatbot(
        merges_path=args.merges,
        vocab_path=args.vocab,
        head_path=head_path,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        ff=args.ff,
        add_pe=args.add_pe,
    )

    history: List[str] = []
    if args.system.strip():
        history.append(f"System: {args.system.strip()}")

    print("Chatbot ready. Type your message and press Enter. Ctrl+C to exit.")
    stop_tok = args.stop_token.strip() or None
    interactive = sys.stdin.isatty()
    try:
        while True:
            if interactive:
                user = input("You: ").strip()
            else:
                # Non-interactive mode (e.g., piped stdin). Read one line without a prompt.
                line = sys.stdin.readline()
                if not line:
                    break  # EOF
                user = line.strip()
                if not user:
                    continue
            reply = bot.chat_once(
                history=history,
                user_message=user,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop_token=stop_tok,
                greedy=args.greedy,
            )
            print(f"Assistant: {reply}")
            history.append(f"User: {user}")
            history.append(f"Assistant: {reply}")
            if not interactive:
                break  # single-turn behavior for piped stdin
    except (KeyboardInterrupt, EOFError):
        if interactive:
            print("\nExiting.")


if __name__ == "__main__":
    main()
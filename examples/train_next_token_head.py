import argparse
import json
import math
import os
import random
from typing import List, Sequence, Tuple

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import TransformerEncoder, generate_causal_mask


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


def _outer(x: Sequence[float], g: Sequence[float]) -> List[List[float]]:
    # x [D], g [V] -> [V x D] if we accumulate as dW[v][d] += x[d]*g[v]
    V = len(g)
    D = len(x)
    out = [[0.0] * D for _ in range(V)]
    for v in range(V):
        gv = g[v]
        if gv == 0.0:
            continue
        row = out[v]
        for d in range(D):
            row[d] = x[d] * gv
    return out


def _adam_update(
    W: List[List[float]],
    b: List[float],
    dW: List[List[float]],
    db: List[float],
    mW: List[List[float]],
    vW: List[List[float]],
    mb: List[float],
    vb: List[float],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    t: int,
) -> None:
    # Adam per-parameter update
    V, D = len(W), (len(W[0]) if W else 0)

    b1t = 1.0 - beta1 ** t
    b2t = 1.0 - beta2 ** t

    for v in range(V):
        # weights
        mW_row = mW[v]
        vW_row = vW[v]
        dW_row = dW[v]
        W_row = W[v]
        for d in range(D):
            mW_row[d] = beta1 * mW_row[d] + (1.0 - beta1) * dW_row[d]
            vW_row[d] = beta2 * vW_row[d] + (1.0 - beta2) * (dW_row[d] * dW_row[d])
            mhat = mW_row[d] / b1t
            vhat = vW_row[d] / b2t
            W_row[d] -= lr * mhat / (math.sqrt(vhat) + eps)
        # bias
        mb[v] = beta1 * mb[v] + (1.0 - beta1) * db[v]
        vb[v] = beta2 * vb[v] + (1.0 - beta2) * (db[v] * db[v])
        mhat_b = mb[v] / b1t
        vhat_b = vb[v] / b2t
        b[v] -= lr * mhat_b / (math.sqrt(vhat_b) + eps)


def _sgd_update(W: List[List[float]], b: List[float], dW: List[List[float]], db: List[float], lr: float) -> None:
    V, D = len(W), (len(W[0]) if W else 0)
    for v in range(V):
        for d in range(D):
            W[v][d] -= lr * dW[v][d]
        b[v] -= lr * db[v]


def _zero_like_matrix(M: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[0.0] * (len(M[0]) if M else 0) for _ in range(len(M))]


def _zero_like_vector(v: Sequence[float]) -> List[float]:
    return [0.0 for _ in range(len(v))]


def _prepare_windows(ids: List[int], seq_len: int, stride: int) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) indices for windows of length seq_len.
    """
    out = []
    N = len(ids)
    i = 0
    while i + seq_len + 1 <= N:
        out.append((i, i + seq_len))
        i += stride
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a next-token linear head on top of a frozen Transformer encoder.")
    parser.add_argument("--corpus", type=str, default="allen.txt", help="Path to corpus text file.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to BPE merges.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to BPE vocab.")
    parser.add_argument("--dim", type=int, default=32, help="Model/embedding dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ff", type=int, default=64, help="FFN hidden size.")
    parser.add_argument("--add_pe", action="store_true", help="Add sinusoidal positional encodings.")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length for training windows.")
    parser.add_argument("--stride", type=int, default=32, help="Stride between training windows.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of passes over the data.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--adam", action="store_true", help="Use Adam optimizer instead of SGD.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--save_head", type=str, default="", help="Optional path to save trained head weights (JSON).")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1) Load tokenizer + vocab + corpus
    tok = BPETokenizer()
    tok.load(args.merges, args.vocab)

    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    tokens = tok.encode(corpus_text)
    if len(tokens) < args.seq_len + 1:
        raise ValueError("Corpus is too small for the chosen seq_len. Provide a larger corpus or reduce seq_len.")

    # 2) Embedding + (frozen) Transformer encoder
    emb = EmbeddingLayer.from_vocab_file(args.vocab, dim=args.dim, seed=42)
    vocab_size = emb.vocab_size

    encoder = TransformerEncoder(
        num_layers=args.layers,
        dim=args.dim,
        num_heads=args.heads,
        ff_hidden=args.ff,
        seed=args.seed,
    )

    pe = SinusoidalPositionalEncoding(dim=args.dim) if args.add_pe else None

    # Precompute ids for efficiency
    ids_all = emb.tokens_to_ids(tokens)

    # 3) Linear prediction head: logits_t = x_t @ W_out^T + b
    # We'll store W as [V x D] so logits = W @ x + b
    W_out = _init_matrix(vocab_size, args.dim, rng, "xavier_uniform")
    b_out = [0.0] * vocab_size

    # Adam state (if used)
    mW = _zero_like_matrix(W_out)
    vW = _zero_like_matrix(W_out)
    mb = _zero_like_vector(b_out)
    vb = _zero_like_vector(b_out)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step_t = 0

    # 4) Training windows
    windows = _prepare_windows(ids_all, args.seq_len, max(1, args.stride))
    if not windows:
        raise ValueError("No training windows prepared. Check seq_len/stride vs corpus length.")

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(windows)
        running_loss = 0.0
        running_tokens = 0
        for idx, (s, e) in enumerate(windows, start=1):
            # window ids: [s, e] inclusive for inputs length L=seq_len
            # targets: next-token ids: ids[s+1 : e+1]
            ids_win = ids_all[s : e + 1]  # length L
            targets = ids_all[s + 1 : e + 1 + 1]  # length L, last target will be unused (ensure bounds)
            # Ensure same length for inputs and targets positions (predict next token)
            if len(targets) > len(ids_win):
                targets = targets[: len(ids_win)]
            elif len(targets) < len(ids_win):
                ids_win = ids_win[: len(targets)]
            L = len(ids_win)
            if L < 2:
                continue

            # Build embeddings
            X = emb.embed_ids(ids_win)  # [L x D]
            if pe is not None:
                X = pe.add_to(X, offset=0)

            # Causal mask to avoid peeking forward
            mask = generate_causal_mask(L)

            # Run encoder (frozen)
            H = encoder(X, mask=mask)  # [L x D]

            # Compute loss and gradients for the head over positions 0..L-2 (predict next token)
            dW = [[0.0] * args.dim for _ in range(vocab_size)]
            db = [0.0] * vocab_size
            loss = 0.0
            count = 0

            for t in range(L - 1):
                x_t = H[t]  # [D]
                y_id = targets[t]  # scalar class id

                # logits = W @ x + b -> [V]
                logits = _matvec(W_out, x_t)
                for v in range(vocab_size):
                    logits[v] += b_out[v]

                probs = _softmax_vec(logits)
                # Cross-entropy loss
                p_y = probs[y_id] if 0 <= y_id < vocab_size else 0.0
                if p_y <= 0.0:
                    loss += 100.0  # guard against log(0) with a large penalty
                else:
                    loss += -math.log(p_y)
                count += 1

                # Gradient wrt logits: p - one_hot(y)
                grad = probs
                if 0 <= y_id < vocab_size:
                    grad = [g for g in grad]  # copy
                    grad[y_id] -= 1.0
                # Accumulate dW and db
                # dW[v][d] += x_t[d] * grad[v], db[v] += grad[v]
                for v in range(vocab_size):
                    gv = grad[v]
                    if gv == 0.0:
                        continue
                    db[v] += gv
                    row = dW[v]
                    for d in range(args.dim):
                        row[d] += x_t[d] * gv

            if count == 0:
                continue

            running_loss += loss
            running_tokens += count

            # Normalize gradients by tokens in window
            scale = 1.0 / count
            for v in range(vocab_size):
                db[v] *= scale
                row = dW[v]
                for d in range(args.dim):
                    row[d] *= scale

            # Optimizer step
            if args.adam:
                step_t += 1
                _adam_update(W_out, b_out, dW, db, mW, vW, mb, vb, lr=args.lr, beta1=beta1, beta2=beta2, eps=eps, t=step_t)
            else:
                _sgd_update(W_out, b_out, dW, db, lr=args.lr)

            # Periodic logging
            if idx % 50 == 0:
                avg_loss = running_loss / max(1, running_tokens)
                ppl = math.exp(min(20.0, avg_loss))  # clamp for stability in print
                print(f"Epoch {epoch} [{idx}/{len(windows)}] - avg loss/token: {avg_loss:.4f}  perplexity: {ppl:.2f}")

        # End of epoch metrics
        if running_tokens > 0:
            avg_loss = running_loss / running_tokens
            ppl = math.exp(min(20.0, avg_loss))
            print(f"Epoch {epoch} done. avg loss/token: {avg_loss:.4f}  perplexity: {ppl:.2f}")

    # Optional: save head
    if args.save_head:
        payload = {
            "dim": args.dim,
            "vocab_size": vocab_size,
            "W_out": W_out,
            "b_out": b_out,
        }
        with open(args.save_head, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"Saved head to {args.save_head}")

    # Quick demo: show top-5 predictions for the last window's final hidden state
    if windows:
        s, e = windows[-1]
        ids_win = ids_all[s : e + 1]
        X = emb.embed_ids(ids_win)
        if pe is not None:
            X = pe.add_to(X, offset=0)
        H = encoder(X, mask=generate_causal_mask(len(X)))
        x_last = H[-1]
        logits = _matvec(W_out, x_last)
        for v in range(vocab_size):
            logits[v] += b_out[v]
        probs = _softmax_vec(logits)
        # top-5
        top5 = sorted(range(vocab_size), key=lambda i: probs[i], reverse=True)[:5]
        inv_vocab = {i: t for t, i in emb.token_to_id.items()}
        if emb.unk_id not in inv_vocab:
            inv_vocab[emb.unk_id] = "<unk>"
        print("Top-5 next-token predictions after the last window:")
        for i in top5:
            tok = inv_vocab.get(i, "<unk>")
            print(f"  {tok:20s}  p={probs[i]:.4f}")


if __name__ == "__main__":
    main()
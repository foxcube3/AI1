import argparse
import json
import math
import random
from typing import List, Optional, Sequence

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import TransformerEncoder, generate_causal_mask


def _softmax_vec(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not logits:
        return []
    t = max(1e-6, float(temperature))
    m = max(logits)
    exps = [math.exp((x - m) / t) for x in logits]
    s = sum(exps)
    if s == 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [e / s for e in exps]


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


def _top_k_indices(probs: Sequence[float], k: int) -> List[int]:
    V = len(probs)
    if V == 0:
        return []
    k = max(1, min(k, V))
    return sorted(range(V), key=lambda i: probs[i], reverse=True)[:k]


def _sample_from_top_k(probs: Sequence[float], k: int, rng: random.Random) -> int:
    idxs = _top_k_indices(probs, k)
    mass = sum(probs[i] for i in idxs)
    if mass <= 0.0:
        return idxs[0]
    # Renormalize over top-k
    renorm = [probs[i] / mass for i in idxs]
    r = rng.random()
    c = 0.0
    for j, p in zip(idxs, renorm):
        c += p
        if r <= c:
            return j
    return idxs[-1]


class Chatbot:
    def __init__(
        self,
        merges_path: str,
        vocab_path: str,
        head_path: str,
        dim: int,
        layers: int,
        heads: int,
        ff: int,
        add_pe: bool,
        seed: int = 123,
    ) -> None:
        # Load next-token head
        with open(head_path, "r", encoding="utf-8") as f:
            head = json.load(f)
        W_out = head["W_out"]
        b_out = head["b_out"]
        head_dim = int(head["dim"])
        if head_dim != dim:
            raise ValueError(f"Head dim {head_dim} does not match model dim {dim}")

        self.W_out = W_out
        self.b_out = b_out

        # Tokenizer + embedding
        self.tok = BPETokenizer()
        self.tok.load(merges_path, vocab_path)
        self.emb = EmbeddingLayer.from_vocab_file(vocab_path, dim=dim, seed=42)

        # Encoder
        self.encoder = TransformerEncoder(
            num_layers=layers,
            dim=dim,
            num_heads=heads,
            ff_hidden=ff,
            seed=seed,
        )
        self.pe = SinusoidalPositionalEncoding(dim=dim) if add_pe else None

        # Id->token convenience
        self.inv_vocab = {i: t for t, i in self.emb.token_to_id.items()}
        if self.emb.unk_id not in self.inv_vocab:
            self.inv_vocab[self.emb.unk_id] = "<unk>"

        self._rng = random.Random(seed ^ 0xBEEF)

    def _forward_hidden(self, tokens: Sequence[str]) -> List[List[float]]:
        X = self.emb.embed_tokens(tokens)
        if self.pe is not None:
            X = self.pe.add_to(X, offset=0)
        mask = generate_causal_mask(len(tokens))
        return self.encoder(X, mask=mask)

    def _next_token_distribution(
        self,
        tokens: Sequence[str],
        temperature: float = 1.0,
    ) -> List[float]:
        H = self._forward_hidden(tokens)
        x_last = H[-1]
        logits = _matvec(self.W_out, x_last)
        for i in range(len(logits)):
            logits[i] += self.b_out[i]
        return _softmax_vec(logits, temperature=temperature)

    def generate(
        self,
        prompt_tokens: Sequence[str],
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_k: int = 20,
        stop_token: Optional[str] = None,
        greedy: bool = False,
    ) -> List[str]:
        tokens = list(prompt_tokens)
        V = len(self.inv_vocab)

        for _ in range(max_new_tokens):
            probs = self._next_token_distribution(tokens, temperature=temperature)
            if greedy:
                next_id = max(range(len(probs)), key=lambda i: probs[i])
            else:
                next_id = _sample_from_top_k(probs, min(top_k, V), self._rng)
            next_tok = self.inv_vocab.get(next_id, "<unk>")
            tokens.append(next_tok)
            if stop_token and next_tok == stop_token:
                break
        return tokens

    def chat_once(
        self,
        history: List[str],
        user_message: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        stop_token: Optional[str],
        greedy: bool = False,
    ) -> str:
        # Simple role-tagged prompt
        prompt_text = " ".join(history + [f"User: {user_message}", "Assistant:"])
        prompt_toks = self.tok.encode(prompt_text)
        out_tokens = self.generate(
            prompt_tokens=prompt_toks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_token=stop_token,
            greedy=greedy,
        )
        # Extract only the newly generated assistant tokens
        new_toks = out_tokens[len(prompt_toks):]
        return self.tok.decode(new_toks).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Console chatbot using BPETokenizer + TransformerEncoder + trained next-token head."
    )
    # Generation flags quick reference:
    # --max_new_tokens N  : Maximum number of tokens to generate for each assistant reply.
    # --temperature FLOAT : Softmax temperature for sampling (default 0.9). Lower -> more deterministic.
    # --top_k N           : Sample only from the top-K most probable tokens (default 20).
    # --greedy            : Use greedy decoding (argmax) instead of sampling (ignores top_k/temperature).
    # --stop_token TOK    : Stop generation when TOK is produced (e.g., "<eos>").
    # Conversation control:
    # --system TEXT       : Optional system prompt prepended to the conversation context.
    parser.add_argument("--head", type=str, required=True, help="Path to trained head JSON.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to BPE merges.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to BPE vocab.")
    parser.add_argument("--dim", type=int, default=32, help="Model/embedding dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ff", type=int, default=64, help="FFN hidden size.")
    parser.add_argument("--add_pe", action="store_true", help="Add sinusoidal positional encodings (must match training).")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Generation length per turn.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Softmax temperature.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument("--stop_token", type=str, default="", help="Optional token that stops generation, e.g., '<eos>'.")
    parser.add_argument("--system", type=str, default="", help="Optional system prompt that prefixes the conversation.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    args = parser.parse_args()

    bot = Chatbot(
        merges_path=args.merges,
        vocab_path=args.vocab,
        head_path=args.head,
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
    try:
        while True:
            user = input("You: ").strip()
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
            # Append to history
            history.append(f"User: {user}")
            history.append(f"Assistant: {reply}")
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
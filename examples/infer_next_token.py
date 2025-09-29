import argparse
import json
import math
from typing import List, Sequence, Optional, Set

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


def _post_process_probs(
    probs: Sequence[float],
    emb: EmbeddingLayer,
    allow_only: Optional[Set[str]] = None,
    ban_tokens: Optional[Set[str]] = None,
    exclude_pad_token: Optional[str] = None,
    min_prob: float = 0.0,
) -> List[float]:
    V = len(probs)
    if V == 0:
        return []
    out = list(probs)

    if allow_only:
        allow_ids = {emb.token_to_id.get(t, emb.unk_id) for t in allow_only}
        for i in range(V):
            if i not in allow_ids:
                out[i] = 0.0

    if ban_tokens:
        for t in ban_tokens:
            idx = emb.token_to_id.get(t, None)
            if idx is not None and 0 <= idx < V:
                out[idx] = 0.0

    if exclude_pad_token:
        idx = emb.token_to_id.get(exclude_pad_token, None)
        if idx is not None and 0 <= idx < V:
            out[idx] = 0.0

    if min_prob > 0.0:
        thr = float(min_prob)
        for i in range(V):
            if out[i] < thr:
                out[i] = 0.0

    s = sum(out)
    if s > 0.0:
        out = [x / s for x in out]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference: score next-token probabilities using a trained head.")
    parser.add_argument("--text", type=str, required=True, help="Prompt text to condition on.")
    parser.add_argument("--head", type=str, required=True, help="Path to trained head JSON (from train_next_token_head.py).")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to BPE merges.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to BPE vocab.")
    parser.add_argument("--dim", type=int, default=32, help="Model/embedding dimension (must match head).")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ff", type=int, default=64, help="FFN hidden size.")
    parser.add_argument("--add_pe", action="store_true", help="Add sinusoidal positional encodings (must match training).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for probability scaling.")
    parser.add_argument("--top_k", type=int, default=10, help="Show top-k predictions.")
    parser.add_argument("--candidates", type=str, default="", help="Comma-separated token candidates to score explicitly.")
    parser.add_argument("--allow_only", type=str, default="", help="Comma-separated tokens to allow; everything else is masked out.")
    parser.add_argument("--ban_tokens", type=str, default="", help="Comma-separated tokens to ban from output.")
    parser.add_argument("--exclude_pad", action="store_true", help="Exclude the pad token from output if provided via --pad_token.")
    parser.add_argument("--pad_token", type=str, default="", help="Pad token string to exclude when --exclude_pad is set.")
    parser.add_argument("--min_prob", type=float, default=0.0, help="Minimum probability threshold; lower values are zeroed then renormalized.")
    args = parser.parse_args()

    with open(args.head, "r", encoding="utf-8") as f:
        head = json.load(f)
    W_out = head["W_out"]
    b_out = head["b_out"]
    head_dim = int(head["dim"])
    if head_dim != args.dim:
        raise ValueError(f"Head dim {head_dim} does not match --dim {args.dim}")

    tok = BPETokenizer()
    tok.load(args.merges, args.vocab)
    emb = EmbeddingLayer.from_vocab_file(args.vocab, dim=args.dim, seed=42)

    encoder = TransformerEncoder(
        num_layers=args.layers,
        dim=args.dim,
        num_heads=args.heads,
        ff_hidden=args.ff,
        seed=123,
    )
    pe = SinusoidalPositionalEncoding(dim=args.dim) if args.add_pe else None

    tokens = tok.encode(args.text)
    if not tokens:
        print("No tokens from input text.")
        return
    X = emb.embed_tokens(tokens)
    if pe is not None:
        X = pe.add_to(X, offset=0)

    mask = generate_causal_mask(len(tokens))
    H = encoder(X, mask=mask)
    x_last = H[-1]

    logits = _matvec(W_out, x_last)
    for i in range(len(logits)):
        logits[i] += b_out[i]
    probs_raw = _softmax_vec(logits, temperature=args.temperature)

    allow_only = {t.strip() for t in args.allow_only.split(",") if t.strip()} if args.allow_only.strip() else None
    ban_tokens = {t.strip() for t in args.ban_tokens.split(",") if t.strip()} if args.ban_tokens.strip() else None
    exclude_pad_token = (args.pad_token.strip() or None) if args.exclude_pad else None
    probs = _post_process_probs(
        probs_raw,
        emb=emb,
        allow_only=allow_only,
        ban_tokens=ban_tokens,
        exclude_pad_token=exclude_pad_token,
        min_prob=args.min_prob,
    )

    inv_vocab = {i: t for t, i in emb.token_to_id.items()}
    if emb.unk_id not in inv_vocab:
        inv_vocab[emb.unk_id] = "<unk>"

    V = len(probs)
    top_k = max(1, min(args.top_k, V))
    top_idx = sorted(range(V), key=lambda i: probs[i], reverse=True)[:top_k]
    print(f"Prompt: {args.text}")
    print(f"Tokens: {tokens}")
    print("Top-k predictions:")
    for i in top_idx:
        print(f"  {inv_vocab.get(i, '<unk>'):20s}  p={probs[i]:.4f}")

    if args.candidates.strip():
        cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
        if cands:
            print("Candidates:")
            for c in cands:
                idx = emb.token_to_id.get(c, emb.unk_id)
                print(f"  {c:20s}  p={probs[idx]:.6f} (id={idx})")


if __name__ == "__main__":
    main()
import argparse
import json
import math
from typing import List, Sequence, Optional

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
    parser.add_argument("--top_k", type=int, default=10, help="Show top-k predictions.")
    parser.add_argument("--candidates", type=str, default="", help="Comma-separated token candidates to score explicitly.")
    args = parser.parse_args()

    # Load head
    with open(args.head, "r", encoding="utf-8") as f:
        head = json.load(f)
    W_out = head["W_out"]
    b_out = head["b_out"]
    head_dim = int(head["dim"])
    if head_dim != args.dim:
        raise ValueError(f"Head dim {head_dim} does not match --dim {args.dim}")

    # Load tokenizer + vocab
    tok = BPETokenizer()
    tok.load(args.merges, args.vocab)
    emb = EmbeddingLayer.from_vocab_file(args.vocab, dim=args.dim, seed=42)

    # Prepare encoder (frozen)
    encoder = TransformerEncoder(
        num_layers=args.layers,
        dim=args.dim,
        num_heads=args.heads,
        ff_hidden=args.ff,
        seed=123,
    )
    pe = SinusoidalPositionalEncoding(dim=args.dim) if args.add_pe else None

    # Encode prompt
    tokens = tok.encode(args.text)
    if not tokens:
        print("No tokens from input text.")
        return
    X = emb.embed_tokens(tokens)
    if pe is not None:
        X = pe.add_to(X, offset=0)

    # Causal mask and encoder forward
    mask = generate_causal_mask(len(tokens))
    H = encoder(X, mask=mask)
    x_last = H[-1]

    # Compute logits and probs
    logits = _matvec(W_out, x_last)
    for i in range(len(logits)):
        logits[i] += b_out[i]
    probs = _softmax_vec(logits)

    # Inverse vocab for printing
    inv_vocab = {i: t for t, i in emb.token_to_id.items()}
    if emb.unk_id not in inv_vocab:
        inv_vocab[emb.unk_id] = "<unk>"

    # Show top-k
    V = len(probs)
    top_k = max(1, min(args.top_k, V))
    top_idx = sorted(range(V), key=lambda i: probs[i], reverse=True)[:top_k]
    print(f"Prompt: {args.text}")
    print(f"Tokens: {tokens}")
    print("Top-k predictions:")
    for i in top_idx:
        print(f"  {inv_vocab.get(i, '<unk>'):20s}  p={probs[i]:.4f}")

    # Explicit candidates
    if args.candidates.strip():
        cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
        if cands:
            print("Candidates:")
            for c in cands:
                idx = emb.token_to_id.get(c, emb.unk_id)
                print(f"  {c:20s}  p={probs[idx]:.6f} (id={idx})")


if __name__ == "__main__":
    main()
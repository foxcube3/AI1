import argparse
import json
import math
import random
from typing import List, Optional, Sequence, Set, Tuple

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import TransformerEncoder, make_causal_mask_from_tokens


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


def _sample_top_k(probs: Sequence[float], k: int, rng: random.Random) -> int:
    V = len(probs)
    if V == 0:
        return 0
    k = max(1, min(k, V))
    top_idx = sorted(range(V), key=lambda j: probs[j], reverse=True)[:k]
    mass = sum(probs[j] for j in top_idx)
    if mass <= 0.0:
        weights = [1.0 / k] * k
    else:
        weights = [probs[j] / mass for j in top_idx]
    return rng.choices(top_idx, weights=weights, k=1)[0]


def _sample_top_p(probs: Sequence[float], p: float, rng: random.Random) -> Tuple[int, List[int]]:
    V = len(probs)
    if V == 0:
        return 0, []
    # Sort indices by descending probability
    idx_sorted = sorted(range(V), key=lambda j: probs[j], reverse=True)
    nucleus: List[int] = []
    acc = 0.0
    p = max(1e-6, min(float(p), 1.0))
    for j in idx_sorted:
        nucleus.append(j)
        acc += probs[j]
        if acc >= p:
            break
    mass = sum(probs[j] for j in nucleus)
    if mass <= 0.0:
        weights = [1.0 / len(nucleus)] * len(nucleus)
    else:
        weights = [probs[j] / mass for j in nucleus]
    chosen = rng.choices(nucleus, weights=weights, k=1)[0]
    return chosen, nucleus


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
    parser = argparse.ArgumentParser(description="Generate a sequence by iterative next-token decoding with a trained head.")
    parser.add_argument("--prompt", type=str, required=True, help="Initial text prompt to condition on.")
    parser.add_argument("--head", type=str, required=True, help="Path to trained head JSON (from train_next_token_head.py).")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to BPE merges.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to BPE vocab.")
    parser.add_argument("--dim", type=int, default=32, help="Model/embedding dimension (must match head).")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ff", type=int, default=64, help="FFN hidden size.")
    parser.add_argument("--add_pe", action="store_true", help="Add sinusoidal positional encodings (must match training).")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature.")
    parser.add_argument("--top_k", type=int, default=10, help="Sample from top-k tokens.")
    parser.add_argument("--top_p", type=float, default=0.0, help="Nucleus sampling cutoff (0 disables; 0<p<=1 enables).")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (argmax) instead of sampling.")
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stdout as they are generated.")
    parser.add_argument("--stop_token", type=str, default="", help="Optional stop token; generation stops when produced.")
    parser.add_argument("--system", type=str, default="", help="Optional system prompt prepended to the user prompt.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sampling.")
    parser.add_argument("--out", type=str, default="", help="Optional path to write final generated text.")
    parser.add_argument("--jsonl", type=str, default="", help="Optional path to write per-step JSON lines for generation.")
    parser.add_argument("--jsonl_include_all", action="store_true", help="Include full probability distribution per step in JSONL (large).")
    parser.add_argument("--max_total_tokens", type=int, default=0, help="Optional hard cap on total tokenized length (prompt + generated). 0 disables.")
    parser.add_argument("--max_total_chars", type=int, default=0, help="Optional hard cap on total character length (prompt + generated). 0 disables.")
    parser.add_argument("--preset", type=str, default="", choices=["deterministic", "balanced", "creative"], help="Decoding preset: deterministic|balanced|creative.")
    # Post-processing controls
    parser.add_argument("--allow_only", type=str, default="", help="Comma-separated tokens to allow; mask others.")
    parser.add_argument("--ban_tokens", type=str, default="", help="Comma-separated tokens to ban.")
    parser.add_argument("--exclude_pad", action="store_true", help="Exclude pad token by name provided via --pad_token.")
    parser.add_argument("--pad_token", type=str, default="", help="Pad token string to exclude when --exclude_pad is set.")
    parser.add_argument("--min_prob", type=float, default=0.0, help="Minimum probability threshold; lower values zeroed then renormalized.")
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

    rng = random.Random(args.seed)

    # Apply decoding presets (override relevant flags)
    if args.preset == "deterministic":
        args.greedy = True
        args.temperature = 0.7
        args.top_k = max(1, args.top_k)
        args.top_p = 0.0
    elif args.preset == "balanced":
        args.greedy = False
        args.temperature = 0.9
        args.top_k = 20
        args.top_p = 0.9
    elif args.preset == "creative":
        args.greedy = False
        args.temperature = 1.1
        args.top_k = 0  # prefer nucleus
        args.top_p = 0.92

    base_text = args.prompt if not args.system else f"{args.system}\n{args.prompt}"
    text = base_text.strip()

    # Prepare reverse vocab
    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {i: t for t, i in vocab.items()}
    if "<unk>" not in vocab:
        id_to_token[emb.unk_id] = "<unk>"

    allow_only = {t.strip() for t in args.allow_only.split(",") if t.strip()} if args.allow_only.strip() else None
    ban_tokens = {t.strip() for t in args.ban_tokens.split(",") if t.strip()} if args.ban_tokens.strip() else None
    exclude_pad_token = (args.pad_token.strip() or None) if args.exclude_pad else None

    jsonl_fp = None
    try:
        if args.jsonl.strip():
            jsonl_fp = open(args.jsonl, "w", encoding="utf-8")

        produced: List[str] = []
        if args.stream:
            # Print initial decoded prompt
            initial = tok.decode(tok.encode(text))
            if initial:
                print(initial, end=" ", flush=True)

        for step in range(max(0, args.max_new_tokens)):
            if args.max_total_chars > 0 and len(text) >= args.max_total_chars:
                break

            tokens = tok.encode(text)
            if not tokens:
                break
            if args.max_total_tokens > 0 and len(tokens) >= args.max_total_tokens:
                break

            X = emb.embed_tokens(tokens)
            if pe is not None:
                X = pe.add_to(X, offset=0)

            mask = make_causal_mask_from_tokens(tokens)
            H = encoder(X, mask=mask)
            x_last = H[-1]

            logits = _matvec(W_out, x_last)
            for i in range(len(logits)):
                logits[i] += b_out[i]

            probs_raw = _softmax_vec(logits, temperature=args.temperature)
            probs = _post_process_probs(
                probs_raw,
                emb=emb,
                allow_only=allow_only,
                ban_tokens=ban_tokens,
                exclude_pad_token=exclude_pad_token,
                min_prob=args.min_prob,
            )

            sampled_from = "greedy"
            top_idx = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)[:max(1, args.top_k)]
            if args.greedy:
                j = top_idx[0]
            else:
                if args.top_p and args.top_p > 0.0:
                    j, nucleus = _sample_top_p(probs, p=args.top_p, rng=rng)
                    sampled_from = "top_p"
                    top_idx = nucleus
                else:
                    j = _sample_top_k(probs, k=args.top_k, rng=rng)
                    sampled_from = "top_k"

            next_tok = id_to_token.get(j, "<unk>")
            if args.max_total_chars > 0:
                prospective = (text + " " + next_tok).strip()
                if len(prospective) > args.max_total_chars:
                    break

            should_stop = bool(args.stop_token and next_tok == args.stop_token)
            if jsonl_fp is not None:
                event = {
                    "step": step,
                    "prompt": text,
                    "next_token": next_tok,
                    "stop": should_stop,
                    "sampled_from": sampled_from,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "candidates": [
                        {"id": idx, "token": id_to_token.get(idx, "<unk>"), "p": probs[idx]}
                        for idx in top_idx
                    ],
                }
                if args.jsonl_include_all:
                    event["probs"] = [{"id": i, "token": id_to_token.get(i, "<unk>"), "p": probs[i]} for i in range(len(probs))]
                jsonl_fp.write(json.dumps(event) + "\n")

            if should_stop:
                break
            produced.append(next_tok)

            text = (text + " " + next_tok).strip()
            if args.stream:
                print(next_tok, end=" ", flush=True)

    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    final = tok.decode(tok.encode(text))
    if args.stream:
        print()  # newline after streaming
    print(final)
    if args.out.strip():
        with open(args.out, "w", encoding="utf-8") as f_out:
            f_out.write(final + "\n")


if __name__ == "__main__":
    main()
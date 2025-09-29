import argparse
import random
import time

from transformer_blocks import TransformerEncoder, generate_causal_mask


def make_random_sequence(seq_len: int, dim: int, seed: int = 0):
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(seq_len)]


def benchmark_once(encoder, X, mask=None):
    start = time.perf_counter()
    Y = encoder(X, mask=mask)
    end = time.perf_counter()
    return (end - start), Y


def main():
    parser = argparse.ArgumentParser(description="Benchmark the pure-Python Transformer encoder.")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length.")
    parser.add_argument("--dim", type=int, default=64, help="Model dimension.")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--layers", type=int, default=4, help="Number of encoder layers.")
    parser.add_argument("--ff", type=int, default=256, help="FFN hidden dimension.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed runs.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for model and input.")
    parser.add_argument("--causal", action="store_true", help="Use a causal mask.")
    args = parser.parse_args()

    # Build model and input
    encoder = TransformerEncoder(
        num_layers=args.layers,
        dim=args.dim,
        num_heads=args.heads,
        ff_hidden=args.ff,
        seed=args.seed,
    )
    X = make_random_sequence(args.seq_len, args.dim, seed=args.seed)
    mask = generate_causal_mask(args.seq_len) if args.causal else None

    # Warmup
    _ = encoder(X, mask=mask)

    # Timed runs
    times = []
    for _ in range(args.repeats):
        t, _ = benchmark_once(encoder, X, mask=mask)
        times.append(t)

    avg = sum(times) / len(times) if times else 0.0
    tokens = args.seq_len
    toks_per_s = tokens / avg if avg > 0 else 0.0

    print("Benchmark results (pure Python):")
    print(f"  seq_len={args.seq_len}, dim={args.dim}, heads={args.heads}, layers={args.layers}, ff={args.ff}")
    print(f"  repeats={args.repeats}, causal={args.causal}")
    print(f"  avg_time_per_forward = {avg:.6f} s")
    print(f"  tokens_per_second    = {toks_per_s:.2f}")


if __name__ == "__main__":
    main()
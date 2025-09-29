import argparse
import time

from transformer_blocks import TransformerEncoder, generate_causal_mask


def time_forward(encoder, X, mask=None):
    t0 = time.perf_counter()
    _ = encoder(X, mask=mask)
    t1 = time.perf_counter()
    return t1 - t0


def make_input(seq_len, dim):
    # simple deterministic input to avoid RNG cost
    return [[(i + 1) * 0.001 * (j + 1) for j in range(dim)] for i in range(seq_len)]


def run_grid(seq_lens, dim, heads, layers, ff, repeats, masked):
    print("seq_len,masked,avg_time_s,tokens_per_s")
    for L in seq_lens:
        encoder = TransformerEncoder(
            num_layers=layers, dim=dim, num_heads=heads, ff_hidden=ff, seed=123
        )
        X = make_input(L, dim)
        mask = generate_causal_mask(L) if masked else None
        # warmup
        _ = encoder(X, mask=mask)
        # timing
        times = []
        for _ in range(repeats):
            times.append(time_forward(encoder, X, mask=mask))
        avg = sum(times) / len(times) if times else 0.0
        tps = L / avg if avg > 0 else 0.0
        print(f"{L},{int(masked)},{avg:.6f},{tps:.2f}")


def parse_seq_grid(arg):
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Compare masked vs unmasked forward times across sequence lengths.")
    parser.add_argument("--seq_grid", type=str, default="8,16,32,64", help="Comma-separated sequence lengths.")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    seq_lens = parse_seq_grid(args.seq_grid)
    run_grid(seq_lens, args.dim, args.heads, args.layers, args.ff, args.repeats, masked=False)
    run_grid(seq_lens, args.dim, args.heads, args.layers, args.ff, args.repeats, masked=True)


if __name__ == "__main__":
    main()
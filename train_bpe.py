import argparse
from pathlib import Path
from bpe_tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a Byte-Pair Encoding tokenizer on a corpus.")
    parser.add_argument("--corpus", type=str, default="allen.txt", help="Path to corpus text file.")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Approximate target vocabulary size.")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum pair frequency to merge.")
    parser.add_argument("--output_prefix", type=str, default="bpe", help="Prefix for saved merges and vocab.")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    tokenizer = BPETokenizer()
    tokenizer.train(str(corpus_path), vocab_size=args.vocab_size, min_frequency=args.min_frequency)

    merges_path = f"{args.output_prefix}_merges.txt"
    vocab_path = f"{args.output_prefix}_vocab.json"
    tokenizer.save(merges_path, vocab_path)

    # Quick demo
    sample = "This is a sample sentence to tokenize."
    encoded = tokenizer.encode(sample)
    print("Sample:", sample)
    print("Encoded:", encoded)
    print(f"Saved merges to {merges_path}")
    print(f"Saved vocab to {vocab_path}")


if __name__ == "__main__":
    main()
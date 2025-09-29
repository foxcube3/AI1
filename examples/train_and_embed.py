import argparse
from pathlib import Path

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer


def main():
    parser = argparse.ArgumentParser(description="Train BPE and embed text in one pipeline.")
    parser.add_argument("--corpus", type=str, default="allen.txt", help="Path to corpus text file.")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Approximate target vocab size.")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum bigram frequency.")
    parser.add_argument("--output_prefix", type=str, default="bpe", help="Prefix for saved merges/vocab.")
    parser.add_argument("--dim", type=int, default=32, help="Embedding dimension.")
    parser.add_argument("--text", type=str, required=True, help="Text to tokenize and embed.")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    # Train tokenizer
    tok = BPETokenizer()
    tok.train(str(corpus_path), vocab_size=args.vocab_size, min_frequency=args.min_frequency)

    merges_path = f"{args.output_prefix}_merges.txt"
    vocab_path = f"{args.output_prefix}_vocab.json"
    tok.save(merges_path, vocab_path)

    # Build embedding layer from saved vocab
    emb = EmbeddingLayer.from_vocab_file(vocab_path, dim=args.dim, seed=42)

    # Tokenize and embed
    tokens = tok.encode(args.text)
    ids = emb.tokens_to_ids(tokens)
    vectors = emb.embed_tokens(tokens)

    print("Saved merges to:", merges_path)
    print("Saved vocab to:", vocab_path)
    print("Input:", args.text)
    print("Tokens:", tokens)
    print("Ids:", ids)
    print("First 3 dims of each vector:", [[round(x, 4) for x in row[:3]] for row in vectors])


if __name__ == "__main__":
    main()
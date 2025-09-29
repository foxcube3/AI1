import argparse

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding


def main():
    parser = argparse.ArgumentParser(description="Embed text and add sinusoidal positional encodings.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to merges file.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to vocab file.")
    parser.add_argument("--dim", type=int, default=32, help="Embedding dimension.")
    parser.add_argument("--text", type=str, required=True, help="Text to embed with positional encodings.")
    args = parser.parse_args()

    tok = BPETokenizer()
    tok.load(args.merges, args.vocab)

    emb = EmbeddingLayer.from_vocab_file(args.vocab, dim=args.dim, seed=42)
    pe = SinusoidalPositionalEncoding(dim=args.dim)

    tokens = tok.encode(args.text)
    base_vecs = emb.embed_tokens(tokens)
    vecs_with_pe = pe.add_to(base_vecs, offset=0)

    print("Input:", args.text)
    print("Tokens:", tokens)
    print("First 3 dims (base):", [[round(x, 4) for x in row[:3]] for row in base_vecs])
    print("First 3 dims (with PE):", [[round(x, 4) for x in row[:3]] for row in vecs_with_pe])


if __name__ == "__main__":
    main()
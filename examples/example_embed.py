import argparse

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer


def main():
    parser = argparse.ArgumentParser(description="Embed text using BPETokenizer + EmbeddingLayer.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to merges file.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to vocab file.")
    parser.add_argument("--dim", type=int, default=32, help="Embedding dimension.")
    parser.add_argument("--text", type=str, required=True, help="Text to embed.")
    args = parser.parse_args()

    # Load tokenizer and vocab
    tok = BPETokenizer()
    tok.load(args.merges, args.vocab)

    # Build embedding layer from vocab
    emb = EmbeddingLayer.from_vocab_file(args.vocab, dim=args.dim, seed=42)

    # Tokenize with BPE and embed
    tokens = tok.encode(args.text)
    ids = emb.tokens_to_ids(tokens)
    vectors = emb.embed_tokens(tokens)

    print("Input:", args.text)
    print("Tokens:", tokens)
    print("Ids:", ids)
    print("First 3 dims of each vector:", [[round(x, 4) for x in row[:3]] for row in vectors])


if __name__ == "__main__":
    main()
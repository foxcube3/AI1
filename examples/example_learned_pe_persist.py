import argparse
import os

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import LearnedPositionalEmbedding


def main():
    parser = argparse.ArgumentParser(description="Demo: save and reload LearnedPositionalEmbedding, then apply.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to merges file.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to vocab file.")
    parser.add_argument("--dim", type=int, default=32, help="Embedding dimension.")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length for learned PE.")
    parser.add_argument("--text", type=str, required=True, help="Text to embed with learned positional embeddings.")
    parser.add_argument("--pe_out", type=str, default="learned_pe.json", help="Path to save learned PE JSON.")
    args = parser.parse_args()

    # Load tokenizer and vocab
    tok = BPETokenizer()
    tok.load(args.merges, args.vocab)

    # Build token embedding layer
    emb = EmbeddingLayer.from_vocab_file(args.vocab, dim=args.dim, seed=42)

    # Create learned positional embeddings, then save to disk
    lpe = LearnedPositionalEmbedding(dim=args.dim, max_len=args.max_len, seed=123)
    lpe.save_weights(args.pe_out)

    # Load into a new instance to verify persistence
    lpe2 = LearnedPositionalEmbedding(dim=1, max_len=1)  # placeholder; will be overwritten by load
    lpe2.load_weights(args.pe_out)

    # Tokenize and embed text
    tokens = tok.encode(args.text)
    base_vecs = emb.embed_tokens(tokens)

    # Add positional encodings from the reloaded instance
    vecs_with_pe = lpe2.add_to(base_vecs, offset=0)

    print("Input:", args.text)
    print("Tokens:", tokens)
    print("Saved learned PE to:", os.path.abspath(args.pe_out))
    print("Reloaded dim/max_len:", lpe2.dim, lpe2.max_len)
    print("First 3 dims (base):", [[round(x, 4) for x in row[:3]] for row in base_vecs])
    print("First 3 dims (with learned PE):", [[round(x, 4) for x in row[:3]] for row in vecs_with_pe])


if __name__ == "__main__":
    main()
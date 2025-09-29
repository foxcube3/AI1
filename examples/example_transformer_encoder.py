import argparse

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import (
    TransformerEncoder,
    generate_causal_mask,
    make_causal_mask_from_tokens,
    make_padding_mask_from_tokens,
)


def main():
    parser = argparse.ArgumentParser(description="Run a small Transformer encoder over embedded tokens.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to merges file.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to vocab file.")
    parser.add_argument("--dim", type=int, default=32, help="Model/embedding dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ff", type=int, default=64, help="FFN hidden size.")
    parser.add_argument("--text", type=str, required=True, help="Text to encode and run through the encoder.")
    parser.add_argument(
        "--add_pe",
        action="store_true",
        help="If set, add sinusoidal positional encodings before the encoder.",
    )
    parser.add_argument(
        "--use_mask",
        type=str,
        default="none",
        choices=["none", "causal", "padding_from_tokens", "causal_from_tokens"],
        help="Type of mask to apply during attention.",
    )
    parser.add_argument(
        "--pad_token",
        type=str,
        default="<pad>",
        help="Pad token string when using *_from_tokens masking.",
    )
    args = parser.parse_args()

    # Load tokenizer and vocab
    tok = BPETokenizer()
    tok.load(args.merges, args.vocab)

    # Build embedding layer and encode tokens
    emb = EmbeddingLayer.from_vocab_file(args.vocab, dim=args.dim, seed=42)
    tokens = tok.encode(args.text)
    X = emb.embed_tokens(tokens)  # [T x D]

    if args.add_pe:
        pe = SinusoidalPositionalEncoding(dim=args.dim)
        X = pe.add_to(X, offset=0)

    # Prepare optional mask
    mask = None
    if args.use_mask == "causal":
        mask = generate_causal_mask(len(tokens))
    elif args.use_mask == "padding_from_tokens":
        mask = make_padding_mask_from_tokens(tokens, pad_token=args.pad_token)
    elif args.use_mask == "causal_from_tokens":
        mask = make_causal_mask_from_tokens(tokens, pad_token=args.pad_token)

    # Build transformer encoder
    encoder = TransformerEncoder(
        num_layers=args.layers,
        dim=args.dim,
        num_heads=args.heads,
        ff_hidden=args.ff,
        seed=123,
    )

    Y = encoder(X, mask=mask)  # [T x D]

    print("Input:", args.text)
    print("Tokens:", tokens)
    print("Sequence length:", len(tokens))
    print("Model dim:", args.dim)
    print("Layers/Heads/FF:", args.layers, args.heads, args.ff)
    print("Mask type:", args.use_mask)
    if mask is not None:
        print("Mask[0]:", mask[0] if mask else None)
    print("First 3 dims (input):", [[round(v, 4) for v in row[:3]] for row in X])
    print("First 3 dims (output):", [[round(v, 4) for v in row[:3]] for row in Y])


if __name__ == "__main__":
    main()
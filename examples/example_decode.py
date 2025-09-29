import argparse
from pathlib import Path

from bpe_tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Encode and decode text using a trained BPE tokenizer.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to merges file.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to vocab file.")
    parser.add_argument("--input", type=str, default="allen.txt", help="Path to input text file.")
    parser.add_argument("--out", type=str, default="examples/allen_decoded.txt", help="Path to write decoded output.")
    args = parser.parse_args()

    # Load tokenizer
    tok = BPETokenizer()
    tok.load(args.merges, args.vocab)

    # Read input
    inp_path = Path(args.input)
    text = inp_path.read_text(encoding="utf-8")

    # Encode and decode
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(decoded, encoding="utf-8")

    # Print summary
    print("Input file:", inp_path)
    print("First 120 chars of input:", text[:120].replace("\n", " "))
    print("Token count:", len(tokens))
    print("First 40 tokens:", tokens[:40])
    print("Decoded written to:", out_path)
    print("First 120 chars of decoded:", decoded[:120].replace("\n", " "))


if __name__ == "__main__":
    main()
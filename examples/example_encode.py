import argparse
from bpe_tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Encode text using a trained BPE tokenizer.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to merges file.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to vocab file.")
    parser.add_argument("--text", type=str, required=True, help="Text to encode.")
    args = parser.parse_args()

    tok = BPETokenizer()
    tok.load(args.merges, args.vocab)
    tokens = tok.encode(args.text)

    print("Input:", args.text)
    print("Tokens:", tokens)


if __name__ == "__main__":
    main()
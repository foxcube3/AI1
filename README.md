BPE Tokenizer (Byte-Pair Encoding)
[![CI](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml)

Overview
- A simple, self-contained BPE subword tokenizer implemented in Python.
- Includes a training script that learns merges from a text corpus (allen.txt by default), and a small example showing how to encode text.

Repository contents
- bpe_tokenizer.py — BPETokenizer class (train, encode, save/load).
- train_bpe.py — CLI to train and save merges and vocab.
- allen.txt — Example corpus to get started immediately.
- examples/example_encode.py — Example script to load a trained tokenizer and encode text.

Requirements
- Python 3.8+ (no external dependencies)

Quick start
1) Train the tokenizer
- python train_bpe.py
- Options:
  - --corpus: path to corpus file (default: allen.txt)
  - --vocab_size: approximate target vocabulary size (default: 1000)
  - --min_frequency: minimum bigram frequency for merges (default: 2)
  - --output_prefix: output prefix for saved files (default: bpe)
- Outputs:
  - bpe_merges.txt
  - bpe_vocab.json

2) Encode some text
- python examples/example_encode.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "This is a sample sentence."

Programmatic usage
- from bpe_tokenizer import BPETokenizer
- tok = BPETokenizer()
- tok.load("bpe_merges.txt", "bpe_vocab.json")
- tokens = tok.encode("Your text to tokenize here.")
- print(tokens)

Notes
- End-of-word marker </w> is used internally during training/encoding to avoid merges across word boundaries.
- The decode method returns a space-joined string of tokens (baseline), which is adequate for many modeling pipelines but not intended for perfect text reconstruction.
- vocab_size is an approximate target; actual number of merges learned depends on available frequent pairs and min_frequency.

License
- MIT or your preferred license (update as needed).

CI badge note
- Replace OWNER/REPO in the badge URL with your GitHub org/user and repository name after pushing to GitHub.
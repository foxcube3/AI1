BPE Tokenizer (Byte-Pair Encoding) + Embedding Layer
[![CI](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml)

Overview
- A simple, self-contained BPE subword tokenizer implemented in Python.
- A dependency-free EmbeddingLayer that is directly compatible with the tokenizer’s output (list of string tokens).
- Includes examples, tests, linting, and packaging via pyproject.toml.

Features
- BPETokenizer
  - Train on a corpus and save/load merges and vocab.
  - Encode text into subword tokens.
- EmbeddingLayer
  - Build from tokenizer vocab (token -> index).
  - Convert tokens to ids and to dense embedding vectors.
  - Handles OOV tokens via a dedicated <unk> index without mutating the saved vocab.
  - Save/load embedding weights to JSON.
  - No external dependencies.
- Tooling
  - Unit tests for both tokenizer and embedding.
  - Ruff and Flake8 linting configured via pyproject.toml.
  - GitHub Actions CI: lint, unit tests, smoke E2E pipeline, build wheels/sdist, optional publish on tag.

Repository contents
- bpe_tokenizer.py — BPETokenizer class (train, encode, save/load).
- embedding.py — EmbeddingLayer (tokens/ids/embeddings, save/load weights).
- train_bpe.py — CLI to train and save merges and vocab.
- allen.txt — Example corpus to get started immediately.
- examples/example_encode.py — Load a trained tokenizer and encode text.
- examples/example_embed.py — Load tokenizer + vocab, build embedding layer, and embed text.
- examples/train_and_embed.py — One-shot pipeline: train BPE then embed text.
- tests/test_bpe.py — Unit tests for BPETokenizer.
- tests/test_embedding.py — Unit tests for EmbeddingLayer.
- pyproject.toml — Packaging metadata and lint configuration (ruff + flake8).
- .github/workflows/python-tests.yml — CI workflow.

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

3) Embed text with the trained vocab
- python examples/example_embed.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "Allen allows ample analysis" --dim 32

4) One-shot training + embedding pipeline
- python examples/train_and_embed.py --corpus allen.txt --vocab_size 1000 --min_frequency 2 --output_prefix bpe --dim 32 --text "Allen allows ample analysis"

Programmatic usage
- Tokenizer
  - from bpe_tokenizer import BPETokenizer
  - tok = BPETokenizer()
  - tok.load("bpe_merges.txt", "bpe_vocab.json")
  - tokens = tok.encode("Your text to tokenize here.")
- Embedding
  - from embedding import EmbeddingLayer
  - emb = EmbeddingLayer.from_vocab_file("bpe_vocab.json", dim=64, seed=42)
  - ids = emb.tokens_to_ids(tokens)
  - vectors = emb.embed_tokens(tokens)

Notes
- End-of-word marker </w> is used internally during training/encoding to avoid merges across word boundaries; it’s not exposed in the final tokens.
- The decode method returns a space-joined string of tokens (baseline), which is adequate for many modeling pipelines but not intended for perfect text reconstruction.
- vocab_size is an approximate target; actual number of merges learned depends on available frequent pairs and min_frequency.
- Embedding OOV policy: tokens not present in vocab map to an internal <unk> index with its own embedding vector. This does not modify the saved vocab file.

Development
- Tests
  - python -m unittest discover -v
- Lint
  - ruff check .
  - flake8 .
- Build (wheel + sdist)
  - python -m pip install build
  - python -m build

CI
- GitHub Actions runs on push and PR:
  - Lints with ruff and flake8 (settings in pyproject.toml).
  - Runs unit tests (tests/test_*.py).
  - Smoke tests the train_and_embed pipeline.
  - Builds distributions and uploads them as artifacts.
- PyPI publish (optional)
  - Set PYPI_API_TOKEN repository secret.
  - Create and push a tag, e.g. v0.1.0:
    - git tag v0.1.0 && git push origin v0.1.0
  - The publish job will build and upload via twine.

License
- MIT or your preferred license (update as needed).

CI badge note
- Replace OWNER/REPO in the badge URL with your GitHub org/user and repository name after pushing to GitHub.
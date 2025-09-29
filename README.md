BPE Tokenizer (Byte-Pair Encoding) + Embedding Layer
[![CI](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml) [![PyPI](https://img.shields.io/pypi/v/bpe-tokenizer-embedding.svg)](https://pypi.org/project/bpe-tokenizer-embedding/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository contents](#repository-contents)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Examples](#examples)
- [API Reference](#api-reference)
  - [BPETokenizer](#bpetokenizer)
    - [train](#bpetokenizer-train)
    - [encode](#bpetokenizer-encode)
    - [decode](#bpetokenizer-decode)
    - [save](#bpetokenizer-save)
    - [load](#bpetokenizer-load)
  - [EmbeddingLayer](#embeddinglayer)
    - [__init__](#embeddinglayer-init)
    - [from_vocab_file](#embeddinglayer-from-vocab-file)
    - [tokens_to_ids](#embeddinglayer-tokens-to-ids)
    - [ids_to_tokens](#embeddinglayer-ids-to-tokens)
    - [embed_ids](#embeddinglayer-embed-ids)
    - [embed_tokens](#embeddinglayer-embed-tokens)
    - [embed_batch](#embeddinglayer-embed-batch)
    - [save_weights](#embeddinglayer-save-weights)
    - [load_weights](#embeddinglayer-load-weights)
  - [SinusoidalPositionalEncoding](#sinusoidalpositionalencoding)
    - [encode](#sinusoidalpositionalencoding-encode)
    - [add_to](#sinusoidalpositionalencoding-add-to)
  - [LearnedPositionalEmbedding](#learnedpositionalembedding)
    - [__init__](#learnedpositionalembedding-init)
    - [encode](#learnedpositionalembedding-encode)
    - [add_to](#learnedpositionalembedding-add-to)
- [Development](#development)
- [CI](#ci)
- [License](#license)

<a id="overview"></a>
Overview
- A simple, self-contained BPE subword tokenizer implemented in Python.
- A dependency-free EmbeddingLayer that is directly compatible with the tokenizer’s output (list of string tokens).
- Sinusoidal positional encodings for sequence modeling.
- Includes examples, tests, linting, and packaging via pyproject.toml.

<a id="features"></a>
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
- Positional Encoding
  - SinusoidalPositionalEncoding (Transformer-style).
  - LearnedPositionalEmbedding (trainable table up to a max length).
  - Generate PE matrices and add to embeddings without external deps.
- Core Transformer Blocks (dependency-free)
  - LayerNorm
  - MultiHeadSelfAttention (scaled dot-product attention with masking)
  - PositionwiseFeedForward (ReLU)
  - TransformerEncoderLayer (pre-norm, residuals)
  - TransformerEncoder (stacked encoder layers)
- Tooling
  - Unit tests for tokenizer, embedding, and positional encoding.
  - Ruff and Flake8 linting configured via pyproject.toml.
  - GitHub Actions CI: lint, unit tests, smoke E2E pipeline, build wheels/sdist, optional publish on tag.

<a id="repository-contents"></a>
Repository contents
- bpe_tokenizer.py — BPETokenizer class (train, encode, save/load).
- embedding.py — EmbeddingLayer (tokens/ids/embeddings, save/load weights).
- positional_encoding.py — Sinusoidal positional encoding utilities.
- transformer_blocks.py — Core Transformer blocks (LayerNorm, MHA, FFN, EncoderLayer, Encoder).
- train_bpe.py — CLI to train and save merges and vocab.
- allen.txt — Example corpus to get started immediately.
- examples/example_encode.py — Load a trained tokenizer and encode text.
- examples/example_embed.py — Load tokenizer + vocab, build embedding layer, and embed text.
- examples/example_embed_with_pe.py — Embed text and add sinusoidal positional encodings.
- examples/example_embed_with_learned_pe.py — Embed text and add learned positional embeddings.
- examples/example_transformer_encoder.py — End-to-end example running a Transformer encoder on embedded tokens.
- examples/train_and_embed.py — One-shot pipeline: train BPE then embed text.
- examples/benchmark_transformer.py — Pure-Python benchmark utility for the Transformer encoder.
- examples/benchmark_transformer_grid.py — Compare masked vs unmasked forward times across lengths.
- tests/test_bpe.py — Unit tests for BPETokenizer.
- tests/test_embedding.py — Unit tests for EmbeddingLayer.
- tests/test_positional.py — Unit tests for positional encoding.
- tests/test_transformer.py — Unit tests for Transformer blocks and masks.
- pyproject.toml — Packaging metadata and lint configuration (ruff + flake8).
- .github/workflows/python-tests.yml — CI workflow.

<a id="requirements"></a>
Requirements
- Python 3.8+ (no external dependencies)

<a id="quick-start"></a>
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
- Positional Encoding
  - from positional_encoding import SinusoidalPositionalEncoding
  - pe = SinusoidalPositionalEncoding(dim=64)
  - pe_vectors = pe.encode(len(vectors), offset=0)
  - vectors_with_pe = pe.add_to(vectors)

Notes
- End-of-word marker </w> is used internally during training/encoding to avoid merges across word boundaries; it’s not exposed in the final tokens.
- The decode method returns a space-joined string of tokens (baseline), which is adequate for many modeling pipelines but not intended for perfect text reconstruction.
- vocab_size is an approximate target; actual number of merges learned depends on available frequent pairs and min_frequency.
- Embedding OOV policy: tokens not present in vocab map to an internal <unk> index with its own embedding vector. This does not modify the saved vocab file.

<a id="development"></a>
Development
- Tests
  - python -m unittest discover -v
- Lint
  - ruff check .
  - flake8 .
- Build (wheel + sdist)
  - python -m pip install build
  - python -m build

<a id="ci"></a>
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

<a id="examples"></a>
Examples
- Encoding example: examples/example_encode.py
  - python examples/example_encode.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "This is a sample sentence."
- Embedding example: examples/example_embed.py
  - python examples/example_embed.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "Allen allows ample analysis" --dim 32
- Embedding + Sinusoidal Positional Encoding: examples/example_embed_with_pe.py
  - python examples/example_embed_with_pe.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "Allen allows ample analysis" --dim 32
- Embedding + Learned Positional Embedding: examples/example_embed_with_learned_pe.py
  - python examples/example_embed_with_learned_pe.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "Allen allows ample analysis" --dim 32 --max_len 512
- Transformer encoder over embedded tokens: examples/example_transformer_encoder.py
  - python examples/example_transformer_encoder.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "Allen allows ample analysis" --dim 32 --layers 2 --heads 4 --ff 64 --add_pe
- Benchmark the Transformer encoder (pure Python): examples/benchmark_transformer.py
  - python examples/benchmark_transformer.py --seq_len 64 --dim 64 --heads 8 --layers 4 --ff 256 --repeats 5 --causal
- Train + embed pipeline: examples/train_and_embed.py
  - python examples/train_and_embed.py --corpus allen.txt --vocab_size 1000 --min_frequency 2 --output_prefix bpe --dim 32 --text "Allen allows ample analysis"

<a id="api-reference"></a>
API Reference

<a id="bpetokenizer"></a>
### BPETokenizer (bpe_tokenizer.py)

<a id="bpetokenizer-train"></a>
#### train(corpus_path: str, vocab_size: int = 1000, min_frequency: int = 2) -> None
Learn merges from a corpus and build a vocabulary.

<a id="bpetokenizer-encode"></a>
#### encode(text: str) -> List[str]
Tokenize text into subword tokens.

<a id="bpetokenizer-decode"></a>
#### decode(tokens: Iterable[str]) -> str
Baseline decode by joining tokens with spaces.

<a id="bpetokenizer-save"></a>
#### save(merges_path: str, vocab_path: str) -> None
Save merges and vocab to disk.

<a id="bpetokenizer-load"></a>
#### load(merges_path: str, vocab_path: str) -> None
Load merges and vocab from disk.

<a id="embeddinglayer"></a>
### EmbeddingLayer (embedding.py)

<a id="embeddinglayer-init"></a>
#### __init__(vocab: Dict[str, int], dim: int, seed: Optional[int] = None, init: str = "xavier_uniform", unk_token: str = "<unk>")
Create an embedding table for a given vocab. Supports init schemes: zeros, uniform, xavier_uniform.

<a id="embeddinglayer-from-vocab-file"></a>
#### from_vocab_file(vocab_path: str, dim: int, seed: Optional[int] = None, init: str = "xavier_uniform", unk_token: str = "<unk>") -> EmbeddingLayer
Construct an embedding layer from a saved vocab JSON file.

<a id="embeddinglayer-tokens-to-ids"></a>
#### tokens_to_ids(tokens: Sequence[str]) -> List[int]
Map tokens to ids using an unk id for OOV tokens.

<a id="embeddinglayer-ids-to-tokens"></a>
#### ids_to_tokens(ids: Sequence[int]) -> List[str]
Reverse mapping; returns <unk> for unknown ids.

<a id="embeddinglayer-embed-ids"></a>
#### embed_ids(ids: Sequence[int]) -> List[List[float]]
Lookup embeddings by id sequence.

<a id="embeddinglayer-embed-tokens"></a>
#### embed_tokens(tokens: Sequence[str]) -> List[List[float]]
Convenience: tokens -> ids -> embeddings.

<a id="embeddinglayer-embed-batch"></a>
#### embed_batch(batch_tokens: Sequence[Sequence[str]]) -> List[List[List[float]]]
Batch embedding for multiple token sequences.

<a id="embeddinglayer-save-weights"></a>
#### save_weights(path: str) -> None
Save weights and minimal metadata to JSON.

<a id="embeddinglayer-load-weights"></a>
#### load_weights(path: str) -> None
Load weights and metadata from JSON.

<a id="sinusoidalpositionalencoding"></a>
### SinusoidalPositionalEncoding (positional_encoding.py)

<a id="sinusoidalpositionalencoding-encode"></a>
#### encode(length: int, offset: int = 0) -> List[List[float]]
Create a [length x dim] positional encoding matrix starting at the given offset.

<a id="sinusoidalpositionalencoding-add-to"></a>
#### add_to(embeddings: Sequence[Sequence[float]], offset: int = 0) -> List[List[float]]
Element-wise add positional encodings to an embedding sequence. Returns a new list.

<a id="learnedpositionalembedding"></a>
### LearnedPositionalEmbedding (positional_encoding.py)

<a id="learnedpositionalembedding-init"></a>
#### __init__(dim: int, max_len: int, init: str = "xavier_uniform", seed: Optional[int] = None)
Create a trainable positional embedding table of shape [max_len x dim]. Init schemes: zeros, uniform, xavier_uniform.

<a id="learnedpositionalembedding-encode"></a>
#### encode(length: int, offset: int = 0) -> List[List[float]]
Return a [length x dim] slice starting from position `offset`. Raises if offset+length > max_len.

<a id="learnedpositionalembedding-add-to"></a>
#### add_to(embeddings: Sequence[Sequence[float]], offset: int = 0) -> List[List[float]]
Element-wise add learned positional embeddings to an embedding sequence. Returns a new list.

#### save_weights(path: str) -> None
Save learned positional weights and metadata (dim, max_len) to JSON.

#### load_weights(path: str) -> None
Load learned positional weights and metadata from JSON.

<a id="transformerblocks"></a>
### Core Transformer Blocks (transformer_blocks.py)

<a id="layernorm"></a>
#### LayerNorm(dim: int, eps: float = 1e-5)
Per-position layer normalization. Callable on sequences X: List[List[float]] with shape [seq_len x dim].

<a id="mhsa"></a>
#### MultiHeadSelfAttention(dim: int, num_heads: int, seed: Optional[int] = None, init: str = "xavier_uniform")
- forward(X, mask=None) -> List[List[float]]
- X shape: [seq_len x dim]; mask shape: [seq_len x seq_len] with values <= 0 masked.
- Returns [seq_len x dim].

<a id="ffn"></a>
#### PositionwiseFeedForward(dim: int, hidden_dim: int, activation: str = "relu", seed: Optional[int] = None, init: str = "xavier_uniform")
- forward(X) -> List[List[float]]
- Applies two linear layers with ReLU, returns [seq_len x dim].

<a id="encoderlayer"></a>
#### TransformerEncoderLayer(dim: int, num_heads: int, ff_hidden: int, seed: Optional[int] = None, init: str = "xavier_uniform")
- forward(X, mask=None) -> List[List[float]]
- Pre-norm residual block: x = x + MHA(LN1(x)); x = x + FFN(LN2(x)).

<a id="encoder"></a>
#### TransformerEncoder(num_layers: int, dim: int, num_heads: int, ff_hidden: int, seed: Optional[int] = None, init: str = "xavier_uniform")
- forward(X, mask=None) -> List[List[float]]
- Stacked encoder layers.

#### generate_causal_mask(seq_len: int) -> List[List[float]]
- Utility that returns a lower-triangular mask with 1.0 allowed and 0.0 masked.

#### generate_padding_mask(seq_len: int, valid_len: int) -> List[List[float]]
- Masks out positions i or j >= valid_len (padding).

#### generate_causal_padding_mask(seq_len: int, valid_len: int) -> List[List[float]]
- Combines causal constraint (j <= i) with padding mask.

#### generate_padding_mask_from_flags(pad_flags: Sequence[bool]) -> List[List[float]]
- pad_flags[i] == True means position i is padding (masked). Returns [L x L] mask.

#### generate_causal_padding_mask_from_flags(pad_flags: Sequence[bool]) -> List[List[float]]
- Same as above but enforces causal constraint (j <= i).

#### build_flags_from_tokens(tokens: Sequence[str], pad_token: str = "<pad>") -> List[bool]
- Returns a per-token boolean list where True marks padding tokens.

#### generate_causal_masks_from_lengths(lengths: Sequence[int]) -> List[List[List[float]]]
- Convenience to create a batch of [L x L] causal+padding masks where L = max(lengths).

<a id="license"></a>
License
- MIT or your preferred license (update as needed).

CI badge note
- Replace OWNER/REPO in the badge URL with your GitHub org/user and repository name after pushing to GitHub.
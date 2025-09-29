BPE Tokenizer (Byte-Pair Encoding) + Embedding Layer
[![CI](https://github.com/foxcube3/AI1/actions/workflows/python-tests.yml/badge.svg)](https://github.com/foxcube3/AI1/actions/workflows/python-tests.yml) [![Manual Terminal](https://github.com/foxcube3/AI1/actions/workflows/manual-terminal.yml/badge.svg)](https://github.com/foxcube3/AI1/actions/workflows/manual-terminal.yml) [![Manual CI](https://github.com/foxcube3/AI1/actions/workflows/manual-ci.yml/badge.svg)](https://github.com/foxcube3/AI1/actions/workflows/manual-ci.yml) [![PyPI](https://img.shields.io/pypi/v/bpe-tokenizer-embedding.svg)](https://pypi.org/project/bpe-tokenizer-embedding/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Single-turn Chatbot](https://img.shields.io/badge/Chatbot-single--turn-blue)](#single-turn-chatbot) [![Manual Terminal Examples](https://img.shields.io/badge/CI-manual%20terminal%20examples-blue)](#ci-manual-terminal-examples)

Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository contents](#repository-contents)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples](#examples)
  - [Post-processing (inference) — quick reference](#post-processing-inference-quick-reference)
  - [Chatbot usage](#chatbot-usage)
  - [Single-turn chatbot quick runs](#single-turn-chatbot)
  - [End-to-end: Tokenizer + Embedding + Encoder](#end-to-end-encoder)
  - [Positional encodings in the encoder](#pe-in-encoder)
- [Simple training (next-token head)](#simple-training-next-token-head)
  - [End-to-end: next-token head](#end-to-end-next-token-head)
  - [Sequence generation (iterative decoding)](#sequence-generation)
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
  - [Inference Post-processing](#api-inference-post-processing)
  - [Transformer Blocks](#transformer-blocks)
    - [LayerNorm](#layernorm)
    - [MultiHeadSelfAttention](#multiheadselfattention)
    - [PositionwiseFeedForward](#positionwisefeedforward)
    - [TransformerEncoderLayer](#transformerencoderlayer)
    - [TransformerEncoder](#transformerencoder)
    - [Mask utilities](#mask-utilities)
- [Development](#development)
  - [Testing](#testing)
  - [Linting](#linting)
  - [Building and publishing](#building-and-publishing)
- [CI](#ci)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
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
- examples/example_learned_pe_persist.py — Demonstrate saving/loading learned positional embedding weights.
- examples/example_transformer_encoder.py — End-to-end example running a Transformer encoder on embedded tokens.
- examples/train_and_embed.py — One-shot pipeline: train BPE then embed text.
- examples/train_next_token_head.py — Train a next-token linear head on top of the frozen Transformer encoder.
- examples/infer_next_token.py — Inference utility for the trained next-token head.
- examples/train_then_infer.py — Train a next-token head and immediately run inference.
- examples/train_then_chatbot.py — Train a next-token head and launch the console chatbot in one command.
- examples/benchmark_transformer.py — Pure-Python benchmark utility for the Transformer encoder.
- examples/benchmark_transformer_grid.py — Compare masked vs unmasked forward times across lengths.
- tests/test_bpe.py — Unit tests for BPETokenizer.
- tests/test_embedding.py — Unit tests for EmbeddingLayer.
- tests/test_positional.py — Unit tests for positional encoding.
- tests/test_transformer.py — Unit tests for Transformer blocks and masks.
- tests/test_next_token_pipeline.py — Smoke test for training and inference pipeline.
- tests/test_train_then_chatbot.py — Smoke test for single-turn chatbot training + run.
- pyproject.toml — Packaging metadata and lint configuration (ruff + flake8).
- .github/workflows/python-tests.yml — Default CI workflow.
- .github/workflows/manual-terminal.yml — On-demand terminal runner workflow.
- .github/workflows/manual-ci.yml — On-demand full CI workflow.

<a id="requirements"></a>
Requirements
- Python 3.8+ (no external dependencies)

<a id="installation"></a>
Installation
- From PyPI:
  - pip install bpe-tokenizer-embedding
- From source (editable):
  - git clone https://github.com/OWNER/REPO.git
  - cd REPO
  - pip install -e .

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
- Human-readable decode: BPETokenizer.decode now applies a lightweight heuristic detokenizer to reinsert spaces and normalize punctuation so assistant/chatbot outputs read naturally. It’s still not intended for perfect text reconstruction, but is much more readable than raw subwords.
- vocab_size is an approximate target; actual number of merges learned depends on available frequent pairs and min_frequency.
- Embedding OOV policy: tokens not present in vocab map to an internal <unk> index with its own embedding vector. This does not modify the saved vocab file.

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
  - With masks:
    - --use_mask none|causal|padding_from_tokens|causal_from_tokens
    - --pad_token "<pad>"  (used by *_from_tokens options)
  - Examples:
    - Causal: python examples/example_transformer_encoder.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "Allen allows ample analysis" --dim 32 --layers 2 --heads 4 --ff 64 --use_mask causal
    - Padding from tokens: python examples/example_transformer_encoder.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "hello world <pad> <pad>" --dim 32 --layers 2 --heads 4 --ff 64 --use_mask padding_from_tokens --pad_token "<pad>"
    - Causal+padding from tokens: python examples/example_transformer_encoder.py --merges bpe_merges.txt --vocab bpe_vocab.json --text "hello world <pad> <pad>" --dim 32 --layers 2 --heads 4 --ff 64 --use_mask causal_from_tokens --pad_token "<pad>"
- Benchmark the Transformer encoder (pure Python): examples/benchmark_transformer.py
  - python examples/benchmark_transformer.py --seq_len 64 --dim 64 --heads 8 --layers 4 --ff 256 --repeats 5 --causal
- Train + embed pipeline: examples/train_and_embed.py
  - python examples/train_and_embed.py --corpus allen.txt --vocab_size 1000 --min_frequency 2 --output_prefix bpe --dim 32 --text "Allen allows ample analysis"
- Simple training (next-token head): examples/train_next_token_head.py
  - python examples/train_next_token_head.py --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --seq_len 32 --epochs 5 --adam --add_pe --save_head head.json
- Inference for trained head: examples/infer_next_token.py
  - Basic usage:
    - python examples/infer_next_token.py --text "Allen allows" --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --top_k 10
  - Post-processing options:
    - Temperature scaling:
      - --temperature 0.7
    - Allow only a set of tokens (others are masked to 0 then renormalized):
      - --allow_only "Allen,analysis"
    - Ban specific tokens:
      - --ban_tokens "<unk>,<pad>"
    - Exclude pad token by name:
      - --exclude_pad --pad_token "<pad>"
    - Minimum probability threshold (values below are zeroed then renormalized):
      - --min_prob 0.001
  - With explicit token candidates:
    - python examples/infer_next_token.py --text "Allen allows" --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --candidates "<pad>,Allen,analysis"
  - Example combining options:
    - python examples/infer_next_token.py --text "Allen allows" --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --temperature 0.8 --exclude_pad --pad_token "<pad>" --ban_tokens "<unk>" --top_k 10
- End-to-end (train then infer): examples/train_then_infer.py
  - python examples/train_then_infer.py --corpus allen.txt --prompt "Allen allows" --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --seq_len 32 --epochs 3 --adam --add_pe --top_k 10 --save_head head.json
- Console chatbot (uses trained head): examples/chatbot.py
  - Train a head first (see examples/train_next_token_head.py), then:
    - python examples/chatbot.py --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --max_new_tokens 32 --temperature 0.9 --top_k 20

<a id="chatbot-usage"></a>
Chatbot usage
- Prerequisite: Train a next-token head and save it to head.json (see examples/train_next_token_head.py).
- Basic command:
  - python examples/chatbot.py --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --max_new_tokens 32 --temperature 0.9 --top_k 20
- Decoding modes:
  - Greedy (deterministic): add --greedy to pick the argmax at each step.
  - Sampling (default): temperature + top_k control diversity and candidate set.
- Useful flags:
  - --max_new_tokens N  : Maximum tokens per assistant reply.
  - --temperature T     : Softmax temperature (lower -> more deterministic).
  - --top_k K           : Sample from the top-K tokens.
  - --greedy            : Use greedy decoding (ignores top_k/temperature).
  - --stop_token "<eos>": Stop generation when the token is produced.
  - --system "TEXT"     : Add a system prompt to guide behavior.
- Examples:
  - Greedy decoding:
    - python examples/chatbot.py --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --greedy --max_new_tokens 32
  - Use a stop token:
    - python examples/chatbot.py --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --stop_token "<eos>" --max_new_tokens 64
  - Add a system prompt:
    - python examples/chatbot.py --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --system "You are a helpful assistant." --max_new_tokens 48

Non-interactive/single-turn modes for train-then-chatbot
- examples/train_then_chatbot.py now supports running a clean, single-turn interaction without interleaved prompts:
  - --stdin: force non-interactive mode and read exactly one line from stdin (no "You:" prompt), then exit.
  - --single_turn: run exactly one turn and exit.
  - --prompt "TEXT": when used with --single_turn, provide the user message directly. If omitted, reads one line from stdin (or prompts once if interactive).
- Examples:
  - Single turn via explicit prompt:
    - python examples/train_then_chatbot.py --single_turn --prompt "Hello there" --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 16 --layers 1 --heads 2 --ff 32 --seq_len 8 --stride 8 --epochs 1 --lr 0.02 --add_pe --save_head head.json --max_new_tokens 8 --temperature 0.9 --top_k 5
  - Single turn via stdin (no prompt printed):
    - echo "Hello there" | python examples/train_then_chatbot.py --single_turn --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 16 --layers 1 --heads 2 --ff 32 --seq_len 8 --stride 8 --epochs 1 --lr 0.02 --add_pe --save_head head.json --max_new_tokens 8 --temperature 0.9 --top_k 5
  - Force non-interactive stdin mode:
    - echo "Hello there" | python examples/train_then_chatbot.py --stdin --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 16 --layers 1 --heads 2 --ff 32 --seq_len 8 --stride 8 --epochs 1 --lr 0.02 --add_pe --save_head head.json --max_new_tokens 8 --temperature 0.9 --top_k 5

Masking utilities (quick snippet)
```python
from transformer_blocks import (
    build_flags_from_tokens,
    generate_padding_mask_from_flags,
    make_causal_mask_from_tokens,
)
from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from transformer_blocks import TransformerEncoder

# Prepare tokens and embeddings
tok = BPETokenizer(); tok.load("bpe_merges.txt", "bpe_vocab.json")
tokens = tok.encode("hello world <pad> <pad>")
emb = EmbeddingLayer.from_vocab_file("bpe_vocab.json", dim=32, seed=42)
X = emb.embed_tokens(tokens)

# Build masks
flags = build_flags_from_tokens(tokens, pad_token="<pad>")
pad_mask = generate_padding_mask_from_flags(flags)             # padding-only
causal_pad_mask = make_causal_mask_from_tokens(tokens)        # causal + padding

# Run encoder with a mask
enc = TransformerEncoder(num_layers=2, dim=32, num_heads=4, ff_hidden=64, seed=123)
Y = enc(X, mask=causal_pad_mask)
```

<a id="simple-training-next-token-head"></a>
Simple training (next-token head)
- This repository remains dependency-free and does not implement full backprop through the Transformer.
- We provide a practical training example that learns a next-token linear head on top of the frozen encoder representations.

Train:
- python examples/train_next_token_head.py --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --seq_len 32 --stride 32 --epochs 5 --lr 0.01 --adam --add_pe --save_head head.json

What it does:
- Tokenizes the corpus with BPETokenizer.
- Embeds tokens and runs them through a frozen TransformerEncoder with a causal mask (optionally adds sinusoidal PE).
- Trains only a softmax linear head (W_out, b_out) to predict the next token id (cross-entropy), using SGD or Adam.

Outputs:
- Prints average loss/token and perplexity per epoch.
- Optionally saves the trained head weights to JSON with --save_head.

Limitations:
- The encoder and embeddings are not updated (frozen). Extending this to full end-to-end training would require implementing a full autodiff or manual gradients for all blocks, which is out of scope for this dependency-free demo.

Inference (next-token head):
- After training and saving head.json:
  - python examples/infer_next_token.py --text "Allen allows" --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --top_k 10
- To score explicit candidates:
  - python examples/infer_next_token.py --text "Allen allows" --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --candidates "<pad>,Allen,analysis"

End-to-end (train then infer):
- Single command to train and immediately run inference:
  - python examples/train_then_infer.py --corpus allen.txt --prompt "Allen allows" --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --seq_len 32 --epochs 3 --adam --add_pe --top_k 10 --save_head head.json

<a id="single-turn-chatbot"></a>
Single-turn chatbot quick runs (clean output)
- Train head then answer one message without printing "You:" or extra lines:
  - Using explicit prompt:
    - python examples/train_then_chatbot.py --single_turn --prompt "Hello there" --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 16 --layers 1 --heads 2 --ff 32 --seq_len 8 --stride 8 --epochs 1 --lr 0.02 --add_pe --save_head head.json --max_new_tokens 8 --temperature 0.9 --top_k 5
  - Read one line from stdin (no prompt printed):
    - echo "Hello there" | python examples/train_then_chatbot.py --single_turn --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 16 --layers 1 --heads 2 --ff 32 --seq_len 8 --stride 8 --epochs 1 --lr 0.02 --add_pe --save_head head.json --max_new_tokens 8 --temperature 0.9 --top_k 5
  - Force stdin mode:
    - echo "Hello there" | python examples/train_then_chatbot.py --stdin --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 16 --layers 1 --heads 2 --ff 32 --seq_len 8 --stride 8 --epochs 1 --lr 0.02 --add_pe --save_head head.json --max_new_tokens 8 --temperature 0.9 --top_k 5

<a id="end-to-end-next-token-head"></a>
End-to-end: next-token head (train then generate)
- This example trains a small next-token linear head on top of a frozen Transformer encoder, then generates the next token probabilities and samples output tokens.

Train a head:
```bash
python examples/train_next_token_head.py \
  --corpus allen.txt \
  --merges bpe_merges.txt --vocab bpe_vocab.json \
  --dim 32 --layers 2 --heads 4 --ff 64 \
  --seq_len 32 --stride 32 --epochs 3 --lr 0.01 \
  --add_pe \
  --save_head head.json
```

Run inference and sample tokens:
```bash
# Basic inference with top-k sampling and temperature
python examples/infer_next_token.py \
  --text "Allen allows" \
  --head head.json \
  --merges bpe_merges.txt --vocab bpe_vocab.json \
  --dim 32 --layers 2 --heads 4 --ff 64 \
  --add_pe \
  --top_k 10 \
  --temperature 0.8
```

Programmatic outline:
```python
import json
from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import TransformerEncoder, make_causal_mask_from_tokens

# Load tokenizer and vocab
tok = BPETokenizer(); tok.load("bpe_merges.txt", "bpe_vocab.json")
emb = EmbeddingLayer.from_vocab_file("bpe_vocab.json", dim=32, seed=42)

# Build frozen encoder
enc = TransformerEncoder(num_layers=2, dim=32, num_heads=4, ff_hidden=64, seed=123)

# Input text
text = "Allen allows"
tokens = tok.encode(text)
ids = emb.tokens_to_ids(tokens)
X = emb.embed_ids(ids)

# Optional positional encoding
pe = SinusoidalPositionalEncoding(dim=32)
X_pe = pe.add_to(X)

# Causal+padding mask
mask = make_causal_mask_from_tokens(tokens)

# Forward through encoder
H = enc(X_pe, mask=mask)  # [T x 32], use the last position for next-token prediction

# Load trained head weights
with open("head.json", "r", encoding="utf-8") as f:
    head = json.load(f)
W_out = head["W_out"]  # [dim x vocab_size]
b_out = head["b_out"]  # [vocab_size]

# Compute logits for the last position (pure Python matvec)
last = H[-1]
logits = [sum(last[i] * W_out[i][j] for i in range(len(last))) + b_out[j] for j in range(len(b_out))]

# Softmax with temperature
import math
T = 0.8
m = max(logits)
probs = [math.exp((x - m) / T) for x in logits]
s = sum(probs)
probs = [p / s for p in probs]

# Sample top-k
k = 10
top_idx = sorted(range(len(probs)), key=lambda j: probs[j], reverse=True)[:k]
# Renormalize within top-k
top_mass = sum(probs[j] for j in top_idx)
top_probs = [probs[j] / top_mass for j in top_idx]

import random
j = random.choices(top_idx, weights=top_probs, k=1)[0]

# Map back to token
# Build reverse vocab
import json
with open("bpe_vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
id_to_token = {i: t for t, i in vocab.items()}
token_out = id_to_token.get(j, "<unk>")
print("Sampled next token:", token_out)
```

<a id="sequence-generation"></a>
Sequence generation (iterative decoding)
- Generate multiple tokens by iteratively:
  1) Encoding the growing text
  2) Embedding + optional positional encoding
  3) Forwarding through the frozen encoder
  4) Predicting next token with the trained head
  5) Appending the sampled token to the text and repeating until a stop condition.

Greedy decoding:
```python
import json, math
from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import TransformerEncoder, make_causal_mask_from_tokens

# Setup
tok = BPETokenizer(); tok.load("bpe_merges.txt", "bpe_vocab.json")
emb = EmbeddingLayer.from_vocab_file("bpe_vocab.json", dim=32, seed=42)
enc = TransformerEncoder(num_layers=2, dim=32, num_heads=4, ff_hidden=64, seed=123)
pe = SinusoidalPositionalEncoding(dim=32)

with open("head.json", "r", encoding="utf-8") as f:
    head = json.load(f)
W_out, b_out = head["W_out"], head["b_out"]

id_to_token = {i: t for t, i in json.load(open("bpe_vocab.json", "r", encoding="utf-8")).items()}

def logits_for_text(text: str):
    tokens = tok.encode(text)
    X = emb.embed_tokens(tokens)
    X_pe = pe.add_to(X)
    H = enc(X_pe, mask=make_causal_mask_from_tokens(tokens))
    h_last = H[-1]
    logits = [sum(h_last[i] * W_out[i][j] for i in range(len(h_last))) + b_out[j] for j in range(len(b_out))]
    return tokens, logits

text = "Allen allows"
max_new = 16
stop_token = "<eos>"

for _ in range(max_new):
    tokens, logits = logits_for_text(text)
    # Greedy: argmax
    j = max(range(len(logits)), key=lambda idx: logits[idx])
    next_tok = id_to_token.get(j, "<unk>")
    if next_tok == stop_token:
        break
    # Append to text (best-effort with a space)
    text = (text + " " + next_tok).strip()

print("Final text:", tok.decode(tok.encode(text)))
```

Top-k sampling with temperature:
```python
import random, math

def sample_top_k(logits, k=10, temperature=0.8):
    m = max(logits)
    probs = [math.exp((x - m) / max(1e-6, temperature)) for x in logits]
    s = sum(probs)
    probs = [p / (s if s > 0 else 1.0) for p in probs]
    top_idx = sorted(range(len(probs)), key=lambda j: probs[j], reverse=True)[:k]
    mass = sum(probs[j] for j in top_idx)
    weights = [probs[j] / (mass if mass > 0 else 1.0) for j in top_idx]
    return random.choices(top_idx, weights=weights, k=1)[0]

text = "Allen allows"
for _ in range(16):
    _, logits = logits_for_text(text)
    j = sample_top_k(logits, k=10, temperature=0.9)
    next_tok = id_to_token.get(j, "<unk>")
    if next_tok in {"<eos>", "<pad>"}:
        break
    text = (text + " " + next_tok).strip()

print("Generated:", tok.decode(tok.encode(text)))
```

Notes:
- Detokenization is heuristic; tok.decode provides readable output but not exact reconstruction.
- Ensure consistent hyperparameters between training and inference: dim, layers, heads, ff, and whether positional encoding was added.
- Stop conditions can include max token count, encountering a special token (e.g., <eos>), or probability thresholds.

<a id="post-processing-inference-quick-reference"></a>
Post-processing (inference) — quick reference
- These options modify the next-token probability distribution produced during inference:
  - --temperature FLOAT
    - Scales softmax logits; lower values sharpen, higher values smooth.
  - --allow_only "tok1,tok2,..."
    - Only the listed tokens are allowed; others are set to 0 then renormalized.
  - --ban_tokens "tokA,tokB,..."
    - Listed tokens are set to 0 then renormalized.
  - --exclude_pad --pad_token "<pad>"
    - Removes the pad token from the output distribution (requires token string).
  - --min_prob FLOAT
    - Zeroes probabilities below the threshold and renormalizes.
- Examples:
  - python examples/infer_next_token.py --text "Allen allows" --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --temperature 0.8 --top_k 10
  - python examples/infer_next_token.py --text "hello <pad>" --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --exclude_pad --pad_token "<pad>"
  - python examples/infer_next_token.py --text "Allen allows" --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --allow_only "Allen,analysis"
  - python examples/infer_next_token.py --text "Allen allows" --head head.json --merges bpe_merges.txt --vocab bpe_vocab.json --dim 32 --layers 2 --heads 4 --ff 64 --add_pe --ban_tokens "<unk>,<pad>" --min_prob 0.001

<a id="api-inference-post-processing"></a>
API Reference — Inference Post-processing
- The inference utility (examples/infer_next_token.py) exposes CLI flags that apply post-processing to the softmax distribution:
  - Temperature: --temperature FLOAT
    - Applies softmax with temperature t (default 1.0).
  - Allow-only: --allow_only "comma,separated,tokens"
    - Masks all tokens except the listed ones, then renormalizes.
  - Ban tokens: --ban_tokens "comma,separated,tokens"
    - Masks listed tokens (probability set to 0), then renormalizes.
  - Exclude pad: --exclude_pad --pad_token "<pad>"
    - Removes the pad token (by string name) from the distribution.
  - Minimum probability: --min_prob FLOAT
    - Zeros out probabilities below threshold, then renormalizes.
- Notes:
  - Token names must match entries in the vocab file used by the embedding layer.
  - Unknown tokens map to the <unk> id; banning <unk> is allowed.
  - Renormalization occurs only if total remaining mass > 0; otherwise the raw distribution is returned unchanged.

<a id="transformer-blocks"></a>
API Reference — Transformer Blocks
- Pure-Python list-based linear algebra (no external dependencies). Sequences are [seq_len x dim].

Type signatures (Python-style):
```python
from typing import List, Optional, Sequence

class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5) -> None: ...
    def __call__(self, X: Sequence[Sequence[float]]) -> List[List[float]]: ...

class MultiHeadSelfAttention:
    def __init__(self, dim: int, num_heads: int,
                 *, seed: Optional[int] = None, init: str = "xavier_uniform") -> None: ...
    def __call__(self, X: Sequence[Sequence[float]],
                 mask: Optional[Sequence[Sequence[float]]] = None) -> List[List[float]]: ...

class PositionwiseFeedForward:
    def __init__(self, dim: int, hidden_dim: int,
                 *, activation: str = "relu", seed: Optional[int] = None, init: str = "xavier_uniform") -> None: ...
    def __call__(self, X: Sequence[Sequence[float]]) -> List[List[float]]: ...

class TransformerEncoderLayer:
    def __init__(self, dim: int, num_heads: int, ff_hidden: int,
                 *, seed: Optional[int] = None, init: str = "xavier_uniform") -> None: ...
    def __call__(self, X: Sequence[Sequence[float]],
                 mask: Optional[Sequence[Sequence[float]]] = None) -> List[List[float]]: ...

class TransformerEncoder:
    def __init__(self, num_layers: int, dim: int, num_heads: int, ff_hidden: int,
                 *, seed: Optional[int] = None, init: str = "xavier_uniform") -> None: ...
    def __call__(self, X: Sequence[Sequence[float]],
                 mask: Optional[Sequence[Sequence[float]]] = None) -> List[List[float]]: ...

def generate_causal_mask(seq_len: int) -> List[List[float]]: ...
def generate_padding_mask(seq_len: int, valid_len: int) -> List[List[float]]: ...
def generate_causal_padding_mask(seq_len: int, valid_len: int) -> List[List[float]]: ...
def generate_causal_masks_from_lengths(lengths: Sequence[int]) -> List[List[List[float]]]: ...
def generate_padding_mask_from_flags(pad_flags: Sequence[bool]) -> List[List[float]]: ...
def generate_causal_padding_mask_from_flags(pad_flags: Sequence[bool]) -> List[List[float]]: ...
def build_flags_from_tokens(tokens: Sequence[str], pad_token: str = "<pad>") -> List[bool]: ...
def make_causal_mask_from_tokens(tokens: Sequence[str], pad_token: str = "<pad>") -> List[List[float]]: ...
def make_padding_mask_from_tokens(tokens: Sequence[str], pad_token: str = "<pad>") -> List[List[float]]: ...
```

<a id="layernorm"></a>
LayerNorm
- Purpose: Per-position layer normalization.
- Behavior:
  - y = (x - mean) / sqrt(var + eps) * gamma + beta
  - Deterministic fallback for near-constant vectors to avoid division by zero.
- Minimal example:
```python
from transformer_blocks import LayerNorm

ln = LayerNorm(dim=8)
X = [[0.1 * (i + j) for j in range(8)] for i in range(4)]
Y = ln(X)  # shape [4 x 8]
```

<a id="multiheadselfattention"></a>
MultiHeadSelfAttention
- Purpose: Scaled dot-product attention with multi-head splitting.
- Notes:
  - dim must be divisible by num_heads.
  - Uses W_q, W_k, W_v, W_o plus biases.
- Minimal example:
```python
from transformer_blocks import MultiHeadSelfAttention, generate_causal_mask

mha = MultiHeadSelfAttention(dim=16, num_heads=4, seed=42)
X = [[0.01 * (i + j) for j in range(16)] for i in range(6)]
mask = generate_causal_mask(len(X))  # optional
Y = mha(X, mask=mask)  # shape [6 x 16]
```

<a id="positionwisefeedforward"></a>
PositionwiseFeedForward
- Purpose: Two-layer MLP with ReLU applied position-wise.
- Minimal example:
```python
from transformer_blocks import PositionwiseFeedForward

ffn = PositionwiseFeedForward(dim=16, hidden_dim=64, seed=1)
X = [[0.02 * (i + j) for j in range(16)] for i in range(5)]
Y = ffn(X)  # shape [5 x 16]
```

<a id="transformerencoderlayer"></a>
TransformerEncoderLayer
- Architecture (pre-norm residual):
  - x = x + MHA(LN1(x))
  - x = x + FFN(LN2(x))
- Minimal example:
```python
from transformer_blocks import TransformerEncoderLayer, generate_causal_mask

layer = TransformerEncoderLayer(dim=32, num_heads=4, ff_hidden=64, seed=123)
X = [[0.01 * (i + j) for j in range(32)] for i in range(10)]
mask = generate_causal_mask(len(X))
Y = layer(X, mask=mask)  # shape [10 x 32]
```

<a id="transformerencoder"></a>
TransformerEncoder
- Stack of TransformerEncoderLayer blocks with deterministic per-layer seeds.
- Minimal example:
```python
from transformer_blocks import TransformerEncoder, generate_causal_mask

enc = TransformerEncoder(num_layers=3, dim=32, num_heads=4, ff_hidden=64, seed=7)
X = [[0.01 * (i + j) for j in range(32)] for i in range(12)]
mask = generate_causal_mask(len(X))
Y = enc(X, mask=mask)  # shape [12 x 32]
```

<a id="mask-utilities"></a>
Mask utilities
- Quick examples:
```python
from transformer_blocks import (
    generate_causal_mask,
    generate_padding_mask,
    generate_causal_padding_mask,
    build_flags_from_tokens,
    generate_padding_mask_from_flags,
    generate_causal_padding_mask_from_flags,
    make_causal_mask_from_tokens,
    make_padding_mask_from_tokens,
)

L = 5
causal = generate_causal_mask(L)
padding = generate_padding_mask(seq_len=L, valid_len=3)
causal_pad = generate_causal_padding_mask(seq_len=L, valid_len=3)

tokens = ["hello", "world", "<pad>", "<pad>", "<pad>"]
flags = build_flags_from_tokens(tokens)  # [False, False, True, True, True]
pad_mask = generate_padding_mask_from_flags(flags)
causal_pad_mask = generate_causal_padding_mask_from_flags(flags)

causal_pad_from_tokens = make_causal_mask_from_tokens(tokens)
pad_from_tokens = make_padding_mask_from_tokens(tokens)
```

<a id="end-to-end-encoder"></a>
End-to-end quick start: Tokenizer + Embedding + Encoder
- Minimal runnable snippet to go from raw text to encoder outputs.
```python
from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import TransformerEncoder, make_causal_mask_from_tokens

# Load trained tokenizer (use your paths)
tok = BPETokenizer(); tok.load("bpe_merges.txt", "bpe_vocab.json")

# Encode text and build embeddings
text = "Allen allows ample analysis"
tokens = tok.encode(text)
emb = EmbeddingLayer.from_vocab_file("bpe_vocab.json", dim=32, seed=42)
X = emb.embed_tokens(tokens)

# Optional sinusoidal positional encoding
pe = SinusoidalPositionalEncoding(dim=32)
X_pe = pe.add_to(X)

# Build a simple encoder and a causal+padding mask directly from tokens
enc = TransformerEncoder(num_layers=2, dim=32, num_heads=4, ff_hidden=64, seed=123)
mask = make_causal_mask_from_tokens(tokens)  # '<pad>' tokens will be masked if present

# Forward pass
Y = enc(X_pe, mask=mask)
print("Encoder output shape:", len(Y), "x", len(Y[0]))
```

<a id="pe-in-encoder"></a>
Using positional encodings in the encoder
- Sinusoidal vs learned positional encodings.

Sinusoidal positional encoding:
```python
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import TransformerEncoder

# Assume X is [seq_len x dim] embeddings
dim = 64
pe = SinusoidalPositionalEncoding(dim=dim)
X_with_pe = pe.add_to(X)  # offset=0 by default

enc = TransformerEncoder(num_layers=3, dim=dim, num_heads=8, ff_hidden=256, seed=7)
Y = enc(X_with_pe)  # optionally pass a mask
```

Learned positional embedding:
```python
from positional_encoding import LearnedPositionalEmbedding
from transformer_blocks import TransformerEncoder

# Assume X is [seq_len x dim] embeddings
dim = 64
seq_len = len(X)
lpe = LearnedPositionalEmbedding(dim=dim, max_len=512, seed=1, init="xavier_uniform")
X_with_lpe = lpe.add_to(X, offset=0)  # raises if seq_len exceeds max_len

enc = TransformerEncoder(num_layers=3, dim=dim, num_heads=8, ff_hidden=256, seed=7)
Y = enc(X_with_lpe)
```

Persisting and reloading learned positional embedding:
```python
from positional_encoding import LearnedPositionalEmbedding

lpe = LearnedPositionalEmbedding(dim=64, max_len=512, seed=1)
lpe.save_weights("learned_pe.json")

lpe2 = LearnedPositionalEmbedding(dim=1, max_len=1)  # dummy; will be overwritten
lpe2.load_weights("learned_pe.json")
assert lpe2.dim == 64 and lpe2.max_len == 512
```

<a id="development"></a>
Development
- Create and activate a virtual environment (recommended).
- Install in editable mode:
  - pip install -e .
- Install optional dev tools:
  - pip install ruff flake8 pytest build

<a id="testing"></a>
Testing
- Run the unittest suite:
  - python -m unittest discover tests -v
- Or with pytest (if installed):
  - python -m pytest -q

<a id="linting"></a>
Linting
- Ruff (configured in pyproject.toml):
  - ruff check .
- Flake8:
  - flake8 .

<a id="building-and-publishing"></a>
Building and publishing
- Build wheels/sdist (requires python-build):
  - python -m build
- Publish to PyPI (requires twine and appropriate credentials):
  - twine upload dist/*

<a id="ci"></a>
CI
- Default CI: .github/workflows/python-tests.yml runs on push/PR across Python 3.9–3.11
  - Lint (ruff, flake8)
  - Unit tests
  - Smoke E2E examples (train_and_embed, learned PE persistence, encoder example, next-token head training/inference, train-then-chatbot single turn)
  - Builds wheels/sdist and uploads artifacts
  - Optional publish on tagged builds

<a id="ci-manual-terminal-examples"></a>
Manual, built-in terminal (on-demand)
- Two additional workflows you can trigger from the Actions tab:
  1) Manual Terminal — arbitrary command runner
     - File: .github/workflows/manual-terminal.yml
     - Inputs:
       - python-version (default 3.11)
       - cmd (default runs unit tests)
     - Usage:
       - Actions → Manual Terminal → Run workflow → adjust cmd as needed
       - Output is captured and uploaded as terminal-output.txt artifact
     - Examples:
       - cmd: python -m pytest -q
       - cmd: python examples/train_then_chatbot.py --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 16 --layers 1 --heads 2 --ff 32 --seq_len 16 --stride 16 --epochs 1 --lr 0.01 --add_pe --save_head head.json --max_new_tokens 16 --temperature 0.9 --top_k 10 --system "You are a helpful assistant." --greedy
       - Clean single-turn chatbot with prompt:
         - cmd: python examples/train_then_chatbot.py --single_turn --prompt "Hello from CI" --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 16 --layers 1 --heads 2 --ff 32 --seq_len 8 --stride 8 --epochs 1 --lr 0.02 --add_pe --save_head head.json --max_new_tokens 8 --temperature 0.9 --top_k 5
       - Clean single-turn chatbot via stdin (no prompt):
         - cmd: echo "Hello from CI" \| python examples/train_then_chatbot.py --single_turn --corpus allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --dim 16 --layers 1 --heads 2 --ff 32 --seq_len 8 --stride 8 --epochs 1 --lr 0.02 --add_pe --save_head head.json --max_new_tokens 8 --temperature 0.9 --top_k 5
  2) Manual CI (Tests + Chatbot) — curated pipeline on demand
     - File: .github/workflows/manual-ci.yml
     - Inputs:
       - python-version (default 3.11)
     - Runs the same lint/tests/smoke steps as default CI and uploads artifacts:
       - ci_bpe_merges.txt, ci_bpe_vocab.json
       - ci_learned_pe.json
       - ci_head.json, ci_head_chat.json
       - Built distributions in dist/

Tips
- Prefer Manual CI when you want a full validation in one click.
- Use Manual Terminal for ad-hoc commands and debugging (output collected as artifact).

Quick links
- Default CI: [python-tests.yml](https://github.com/foxcube3/AI1/actions/workflows/python-tests.yml)
- Manual Terminal: [manual-terminal.yml](https://github.com/foxcube3/AI1/actions/workflows/manual-terminal.yml)
- Manual CI: [manual-ci.yml](https://github.com/foxcube3/AI1/actions/workflows/manual-ci.yml)

<a id="contributing"></a>
Contributing
- We welcome PRs for bug fixes, improvements, examples, and docs.
- Guidelines:
  - Keep code dependency-free (pure Python standard library).
  - Follow existing style and minimal, meaningful comments.
  - Add or update unit tests under tests/ for new functionality.
  - Run lint and tests locally:
    - ruff check .
    - flake8 .
    - python -m unittest discover tests -v
  - For features touching examples/, include a short usage snippet in README or the example's docstring.
  - CI will run lint, tests, and smoke examples on pull requests.
- Release process:
  - Bump version in pyproject.toml.
  - Build distributions: python -m build
  - Publish via CI or twine after tagging.

<a id="troubleshooting"></a>
Troubleshooting
- Dimension mismatches:
  - Ensure EmbeddingLayer(dim) matches TransformerEncoder(dim).
  - Positional encodings must use the same dim as embeddings; add_to will raise on mismatch.
- Mask shapes:
  - Masks must be [seq_len x seq_len]. Using tokens-based helpers is safer:
    - make_causal_mask_from_tokens(tokens)
    - make_padding_mask_from_tokens(tokens)
- Vocab consistency:
  - Use the same vocab file for embedding and inference. Unknown tokens map to <unk>.
  - If you ban <unk> during inference, confirm it exists in the vocab or is allocated internally.
- Learned positional embedding max_len:
  - LearnedPositionalEmbedding.encode/add_to will raise if offset+length exceeds max_len. Increase max_len or trim sequences.
- Next-token head alignment:
  - Train and infer with matching encoder hyperparameters (dim, layers, heads, ff).
  - If you trained with --add_pe, use the same PE during inference for consistent hidden states.
- Performance tips:
  - Reduce dim, layers, heads, ff for quicker runs.
  - Shorten seq_len/stride and epochs in examples for faster CI or local smoke tests.
- Determinism:
  - Setting seeds on EmbeddingLayer, Transformer blocks, and examples yields repeatable outputs, but sampling (top_k/temperature) is stochastic unless you also fix random module state.
- Decoding readability:
  - BPETokenizer.decode applies a heuristic detokenizer for human-readable output. It's not exact reconstruction but avoids "A l l e n" artifacts.

<a id="license"></a>
License
- MIT License. See the LICENSE file for details.
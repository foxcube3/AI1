BPE Tokenizer (Byte-Pair Encoding) + Embedding Layer
[![CI](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml) [![PyPI](https://img.shields.io/pypi/v/bpe-tokenizer-embedding.svg)](https://pypi.org/project/bpe-tokenizer-embedding/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository contents](#repository-contents)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Examples](#examples)
  - [Post-processing (inference) — quick reference](#post-processing-inference-quick-reference)
  - [Chatbot usage](#chatbot-usage)
- [Simple training (next-token head)](#simple-training-next-token-head)
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
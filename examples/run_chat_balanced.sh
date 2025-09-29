#!/usr/bin/env bash
# One-click balanced chatbot run
# Usage:
#   ./examples/run_chat_balanced.sh [HEAD_JSON] [EXTRA_ARGS...]
# Examples:
#   ./examples/run_chat_balanced.sh head.json
#   ./examples/run_chat_balanced.sh head.json --system "You are a helpful assistant." --stream

set -euo pipefail

HEAD_JSON=${1:-head.json}
shift || true

python examples/chatbot.py \
  --head "$HEAD_JSON" \
  --merges bpe_merges.txt \
  --vocab bpe_vocab.json \
  --dim 32 \
  --layers 2 \
  --heads 4 \
  --ff 64 \
  --add_pe \
  --preset balanced \
  "$@"
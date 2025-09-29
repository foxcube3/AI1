#!/usr/bin/env bash
# One-click deterministic chatbot run (greedy, lower temperature)
# Usage:
#   ./examples/run_chat_deterministic.sh [HEAD_JSON] [EXTRA_ARGS...]
# Example:
#   ./examples/run_chat_deterministic.sh head.json

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
  --preset deterministic \
  "$@"
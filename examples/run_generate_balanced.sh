#!/usr/bin/env bash
# One-click balanced sequence generation (iterative decoding)
# Usage:
#   ./examples/run_generate_balanced.sh "PROMPT" [HEAD_JSON] [EXTRA_ARGS...]
# Examples:
#   ./examples/run_generate_balanced.sh "Allen allows" head.json
#   ./examples/run_generate_balanced.sh "Hello world" head.json --out out.txt --jsonl gen.jsonl

set -euo pipefail

PROMPT=${1:-}
if [ -z "$PROMPT" ]; then
  echo "Error: PROMPT is required. Usage: ./examples/run_generate_balanced.sh \"PROMPT\" [HEAD_JSON] [EXTRA_ARGS...]"
  exit 1
fi

HEAD_JSON=${2:-head.json}
shift || true
shift || true

python examples/generate_sequence.py \
  --prompt "$PROMPT" \
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
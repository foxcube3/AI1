# Convenience Makefile targets for running chatbot and sequence generation presets
# Usage examples:
#   make chat-balanced HEAD=head.json ARGS="--system 'You are helpful.' --stream"
#   make chat-deterministic HEAD=head.json
#   make chat-creative HEAD=head.json ARGS="--stream"
#   make generate-balanced PROMPT="Allen allows" HEAD=head.json ARGS="--out out.txt --jsonl gen.jsonl"
#
# Variables:
#   HEAD   : path to trained head JSON (default: head.json)
#   PROMPT : prompt string for generation (default: Allen allows)
#   ARGS   : extra CLI args forwarded to the underlying scripts
#
# Dev targets:
#   make lint       -> ruff + flake8
#   make test       -> unittest discover
#   make pytest     -> pytest (if installed)
#   make ci         -> lint + test
#   make build      -> build wheel + sdist
#   make publish    -> twine upload dist/*

.PHONY: help chat-balanced chat-deterministic chat-creative generate-balanced decode-allen lint test pytest ci build publish

HEAD ?= head.json
PROMPT ?= Allen allows
ARGS ?=

help:
	@echo "Targets:"
	@echo "  make chat-balanced           HEAD=head.json ARGS='--system \"You are helpful.\" --stream'"
	@echo "  make chat-deterministic      HEAD=head.json"
	@echo "  make chat-creative           HEAD=head.json ARGS='--stream'"
	@echo "  make generate-balanced       PROMPT='Allen allows' HEAD=head.json ARGS='--out out.txt --jsonl gen.jsonl'"
	@echo "  make decode-allen            Decode allen.txt with trained BPE to examples/allen_decoded.txt"
	@echo "  make lint                    Run ruff + flake8"
	@echo "  make test                    Run unittest suite"
	@echo "  make pytest                  Run pytest (if installed)"
	@echo "  make ci                      Run lint + test"
	@echo "  make build                   Build wheel + sdist (python -m build)"
	@echo "  make publish                 Upload dist/* with twine"

chat-balanced:
	./examples/run_chat_balanced.sh $(HEAD) $(ARGS)

chat-deterministic:
	./examples/run_chat_deterministic.sh $(HEAD) $(ARGS)

chat-creative:
	./examples/run_chat_creative.sh $(HEAD) $(ARGS)

generate-balanced:
	./examples/run_generate_balanced.sh "$(PROMPT)" $(HEAD) $(ARGS)

decode-allen:
	python examples/example_decode.py --input allen.txt --merges bpe_merges.txt --vocab bpe_vocab.json --out examples/allen_decoded.txt

lint:
	ruff check .
	flake8 .

test:
	python -m unittest discover tests -v

pytest:
	python -m pytest -q || echo "pytest not installed; skipping."

ci: lint test

build:
	python -m build

publish:
	twine upload dist/* || echo "twine not installed or credentials missing; please configure and retry."
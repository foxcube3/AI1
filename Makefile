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

.PHONY: help chat-balanced chat-deterministic chat-creative generate-balanced

HEAD ?= head.json
PROMPT ?= Allen allows
ARGS ?=

help:
	@echo "Targets:"
	@echo "  make chat-balanced           HEAD=head.json ARGS='--system \"You are helpful.\" --stream'"
	@echo "  make chat-deterministic      HEAD=head.json"
	@echo "  make chat-creative           HEAD=head.json ARGS='--stream'"
	@echo "  make generate-balanced       PROMPT='Allen allows' HEAD=head.json ARGS='--out out.txt --jsonl gen.jsonl'"

chat-balanced:
	./examples/run_chat_balanced.sh $(HEAD) $(ARGS)

chat-deterministic:
	./examples/run_chat_deterministic.sh $(HEAD) $(ARGS)

chat-creative:
	./examples/run_chat_creative.sh $(HEAD) $(ARGS)

generate-balanced:
	./examples/run_generate_balanced.sh "$(PROMPT)" $(HEAD) $(ARGS)
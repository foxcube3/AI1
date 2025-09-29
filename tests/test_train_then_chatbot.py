import os
import sys
import builtins


def test_train_then_chatbot_single_turn(tmp_path):
    # Prepare output head path in temp directory
    head_path = tmp_path / "head.json"

    # Build argv for a quick training run
    argv = [
        "examples/train_then_chatbot.py",
        "--corpus", "allen.txt",
        "--merges", "bpe_merges.txt",
        "--vocab", "bpe_vocab.json",
        "--dim", "16",
        "--layers", "1",
        "--heads", "2",
        "--ff", "32",
        "--seq_len", "16",
        "--stride", "16",
        "--epochs", "1",
        "--lr", "0.01",
        "--add_pe",
        "--save_head", str(head_path),
        "--max_new_tokens", "4",
        "--temperature", "0.9",
        "--top_k", "5",
        "--system", "Test system prompt",
        "--stop_token", "<eos>",
        "--greedy",
    ]

    # Monkeypatch sys.argv and input() to simulate a single user turn then exit
    old_argv = sys.argv
    sys.argv = argv
    input_calls = {"count": 0}

    def fake_input(prompt=""):
        # Return a single user message, then simulate Ctrl+C to exit
        if input_calls["count"] == 0:
            input_calls["count"] += 1
            return "Hello there"
        raise KeyboardInterrupt

    old_input = builtins.input
    builtins.input = fake_input
    try:
        # Import and run main from the script
        import examples.train_then_chatbot as script
        script.main()
    finally:
        # Restore patched globals
        builtins.input = old_input
        sys.argv = old_argv

    # Verify the trained head was saved
    assert head_path.exists(), "Expected trained head.json to be created"

    # Basic sanity: file should contain minimal JSON keys
    import json
    with open(head_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "dim" in data and "W_out" in data and "b_out" in data, "Head JSON missing required keys"
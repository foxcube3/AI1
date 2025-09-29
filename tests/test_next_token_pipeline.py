import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class TestNextTokenPipeline(unittest.TestCase):
    def test_train_then_infer_smoke(self):
        # Use the included BPE assets and corpus
        merges = "bpe_merges.txt"
        vocab = "bpe_vocab.json"
        corpus = "allen.txt"

        # Ensure assets exist
        self.assertTrue(os.path.exists(merges), "Missing merges file")
        self.assertTrue(os.path.exists(vocab), "Missing vocab file")
        self.assertTrue(os.path.exists(corpus), "Missing corpus file")

        # Train a tiny head and save to a temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            head_path = os.path.join(tmpdir, "head.json")

            # Run training for 1 epoch with small seq_len to keep runtime short
            train_argv = [
                "train_next_token_head.py",
                "--corpus", corpus,
                "--merges", merges,
                "--vocab", vocab,
                "--dim", "16",
                "--layers", "1",
                "--heads", "2",
                "--ff", "32",
                "--seq_len", "8",
                "--stride", "8",
                "--epochs", "1",
                "--lr", "0.02",
                "--save_head", head_path,
                # No --adam and no --add_pe for speed
            ]
            # Import here to avoid import-time side effects on sys.argv
            from examples import train_next_token_head as train_mod

            # Capture stdout to avoid noisy test logs
            with patch.object(sys, "argv", train_argv):
                with redirect_stdout(io.StringIO()):
                    train_mod.main()

            # Validate head file
            self.assertTrue(os.path.exists(head_path), "Head file not created")
            with open(head_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("W_out", payload)
            self.assertIn("b_out", payload)
            self.assertIn("dim", payload)
            self.assertIsInstance(payload["W_out"], list)
            self.assertIsInstance(payload["b_out"], list)
            self.assertEqual(int(payload["dim"]), 16)

            # Now run inference using the saved head
            infer_argv = [
                "infer_next_token.py",
                "--text", "Allen allows",
                "--head", head_path,
                "--merges", merges,
                "--vocab", vocab,
                "--dim", "16",
                "--layers", "1",
                "--heads", "2",
                "--ff", "32",
                "--top_k", "5",
                # Match training flags (no add_pe)
            ]
            from examples import infer_next_token as infer_mod
            with patch.object(sys, "argv", infer_argv):
                with redirect_stdout(io.StringIO()):
                    infer_mod.main()


if __name__ == "__main__":
    unittest.main()
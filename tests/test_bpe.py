import os
import tempfile
import unittest
from pathlib import Path

from bpe_tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_path = Path("allen.txt")
        if not cls.corpus_path.exists():
            # Create a minimal fallback corpus if allen.txt isn't present
            cls.corpus_path.write_text(
                "Allen walked down the alley. Allen always allows ample analysis along ancient avenues.",
                encoding="utf-8",
            )

    def test_train_produces_merges_and_vocab(self):
        tok = BPETokenizer()
        tok.train(str(self.corpus_path), vocab_size=200, min_frequency=2)
        self.assertTrue(len(tok.merges) > 0, "Expected some merges to be learned")
        self.assertTrue(len(tok.vocab) > 0, "Expected a non-empty vocab")

    def test_encode_basic(self):
        tok = BPETokenizer()
        tok.train(str(self.corpus_path), vocab_size=200, min_frequency=2)
        tokens = tok.encode("Allen allows ample analysis")
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)

    def test_save_and_load(self):
        tok = BPETokenizer()
        tok.train(str(self.corpus_path), vocab_size=200, min_frequency=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            merges_path = os.path.join(tmpdir, "merges.txt")
            vocab_path = os.path.join(tmpdir, "vocab.json")
            tok.save(merges_path, vocab_path)

            tok2 = BPETokenizer()
            tok2.load(merges_path, vocab_path)

            self.assertEqual(len(tok.merges), len(tok2.merges))
            self.assertEqual(set(tok.vocab.keys()), set(tok2.vocab.keys()))


if __name__ == "__main__":
    unittest.main()
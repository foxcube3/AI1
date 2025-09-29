import os
import tempfile
import unittest
from pathlib import Path

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer


class TestEmbeddingLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_path = Path("allen.txt")
        if not cls.corpus_path.exists():
            cls.corpus_path.write_text(
                "Allen walked down the alley. Allen always allows ample analysis along ancient avenues.",
                encoding="utf-8",
            )

    def test_basic_tokens_to_ids_and_embed(self):
        vocab = {"Allen": 0, "allows": 1, "ample": 2, "analysis": 3}
        emb = EmbeddingLayer(vocab=vocab, dim=8, seed=123)

        tokens = ["Allen", "allows", "UNKNOWN_TOKEN"]
        ids = emb.tokens_to_ids(tokens)

        # Known ids should map correctly; unknown maps to unk_id
        self.assertEqual(ids[0], vocab["Allen"])
        self.assertEqual(ids[1], vocab["allows"])
        self.assertEqual(ids[2], emb.unk_id)

        # Embedding vectors should have correct dimensionality
        vecs = emb.embed_tokens(tokens)
        self.assertEqual(len(vecs), len(tokens))
        self.assertTrue(all(len(v) == 8 for v in vecs))

    def test_deterministic_with_seed(self):
        vocab = {"a": 0, "b": 1}
        emb1 = EmbeddingLayer(vocab=vocab, dim=16, seed=42)
        emb2 = EmbeddingLayer(vocab=vocab, dim=16, seed=42)

        self.assertEqual(emb1.weights, emb2.weights)

    def test_save_and_load_weights(self):
        vocab = {"a": 0, "b": 1}
        emb = EmbeddingLayer(vocab=vocab, dim=12, seed=7)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "emb.json")
            emb.save_weights(path)

            emb2 = EmbeddingLayer(vocab=vocab, dim=12, seed=0)  # will be overwritten by load
            emb2.load_weights(path)

            self.assertEqual(emb2.dim, 12)
            self.assertEqual(len(emb2.weights), emb.vocab_size)
            self.assertEqual(emb.weights, emb2.weights)

    def test_with_bpe_tokenizer_integration(self):
        # Train a small tokenizer then ensure embedding can consume its vocab
        tok = BPETokenizer()
        tok.train(str(self.corpus_path), vocab_size=200, min_frequency=2)

        emb = EmbeddingLayer(tok.vocab, dim=10, seed=1)
        text = "Allen allows ample analysis"
        tokens = tok.encode(text)
        ids = emb.tokens_to_ids(tokens)
        vecs = emb.embed_tokens(tokens)

        self.assertEqual(len(ids), len(tokens))
        self.assertEqual(len(vecs), len(tokens))
        self.assertTrue(all(len(v) == 10 for v in vecs))


if __name__ == "__main__":
    unittest.main()
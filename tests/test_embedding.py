import os
import tempfile
import unittest
from pathlib import Path
import math

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

    def test_initialization_modes(self):
        vocab = {"x": 0, "y": 1, "z": 2}
        dim = 20

        # zeros
        emb_zero = EmbeddingLayer(vocab=vocab, dim=dim, seed=1, init="zeros")
        self.assertTrue(all(all(val == 0.0 for val in row) for row in emb_zero.weights))

        # uniform
        emb_uni = EmbeddingLayer(vocab=vocab, dim=dim, seed=1, init="uniform")
        limit_uni = 1.0 / math.sqrt(dim)
        all_in_range_uni = all(all(-limit_uni <= v <= limit_uni for v in row) for row in emb_uni.weights)
        self.assertTrue(all_in_range_uni)
        # not all zeros
        self.assertTrue(any(any(v != 0.0 for v in row) for row in emb_uni.weights))

        # xavier_uniform
        emb_xav = EmbeddingLayer(vocab=vocab, dim=dim, seed=1, init="xavier_uniform")
        limit_xav = math.sqrt(6.0 / (dim + dim))
        all_in_range_xav = all(all(-limit_xav <= v <= limit_xav for v in row) for row in emb_xav.weights)
        self.assertTrue(all_in_range_xav)
        # ranges should generally differ from plain uniform when dim>0
        self.assertNotEqual(round(limit_uni, 6), round(limit_xav, 6))

    def test_unk_token_handling(self):
        # Case 1: vocab does not include <unk>
        vocab = {"a": 0, "b": 1}
        emb = EmbeddingLayer(vocab=vocab, dim=4, seed=0)
        self.assertGreaterEqual(emb.unk_id, 0)
        self.assertNotIn("<unk>", vocab)  # original vocab unchanged
        ids = emb.tokens_to_ids(["a", "c"])
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[1], emb.unk_id)

        # Case 2: vocab includes <unk>
        vocab2 = {"a": 0, "<unk>": 99}
        emb2 = EmbeddingLayer(vocab=vocab2, dim=4, seed=0)
        self.assertEqual(emb2.unk_id, 99)
        ids2 = emb2.tokens_to_ids(["missing"])
        self.assertEqual(ids2[0], 99)


if __name__ == "__main__":
    unittest.main()
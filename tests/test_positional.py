import unittest
import math

from positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEmbedding


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    def test_shapes_and_determinism(self):
        dim = 12
        pe = SinusoidalPositionalEncoding(dim=dim)
        out1 = pe.encode(5)
        out2 = pe.encode(5)
        self.assertEqual(len(out1), 5)
        self.assertTrue(all(len(row) == dim for row in out1))
        self.assertEqual(out1, out2)  # deterministic

    def test_values_monotonicity(self):
        # Check that encodings for different positions differ
        dim = 8
        pe = SinusoidalPositionalEncoding(dim=dim)
        a = pe.encode(1, offset=0)[0]
        b = pe.encode(1, offset=1)[0]
        self.assertNotEqual(a, b)

    def test_known_small_dim(self):
        # For dim=2, PE[p,0]=sin(p/base^(0))=sin(p), PE[p,1]=cos(p)
        base = 10000.0
        pe = SinusoidalPositionalEncoding(dim=2, base=base)
        row = pe.encode(1, offset=2)[0]
        self.assertAlmostEqual(row[0], math.sin(2.0 / (base ** 0.0)), places=7)
        self.assertAlmostEqual(row[1], math.cos(2.0 / (base ** 0.0)), places=7)

    def test_add_to(self):
        dim = 6
        pe = SinusoidalPositionalEncoding(dim=dim)
        embeddings = [[0.1] * dim, [0.2] * dim, [0.3] * dim]
        out = pe.add_to(embeddings, offset=10)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(len(row) == dim for row in out))
        # Ensure it's a sum (not identical to either input alone)
        self.assertNotEqual(out[0], embeddings[0])
        self.assertNotEqual(out[0], pe.encode(1, offset=10)[0])

    def test_dim_mismatch_raises(self):
        pe = SinusoidalPositionalEncoding(dim=4)
        with self.assertRaises(ValueError):
            pe.add_to([[0.0, 0.0, 0.0]], offset=0)  # dim=3 vs expected 4


class TestLearnedPositionalEmbedding(unittest.TestCase):
    def test_shapes_and_range(self):
        dim = 16
        max_len = 50
        pe = LearnedPositionalEmbedding(dim=dim, max_len=max_len, seed=123, init="uniform")
        out = pe.encode(10, offset=5)
        self.assertEqual(len(out), 10)
        self.assertTrue(all(len(row) == dim for row in out))
        # Ensure consistent slice equals weights slice
        self.assertEqual(out[0], pe.weights[5])
        self.assertEqual(out[-1], pe.weights[14])

    def test_add_to_and_dim_check(self):
        dim = 8
        max_len = 20
        pe = LearnedPositionalEmbedding(dim=dim, max_len=max_len, seed=1)
        embeddings = [[0.1] * dim for _ in range(4)]
        out = pe.add_to(embeddings, offset=3)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(len(row) == dim for row in out))
        # Mismatch
        with self.assertRaises(ValueError):
            pe.add_to([[0.0, 0.0]], offset=0)

    def test_bounds(self):
        dim = 4
        max_len = 5
        pe = LearnedPositionalEmbedding(dim=dim, max_len=max_len)
        with self.assertRaises(ValueError):
            pe.encode(6)  # exceeds max_len
        with self.assertRaises(ValueError):
            pe.encode(3, offset=4)  # 4..6 exceeds max_len


if __name__ == "__main__":
    unittest.main()
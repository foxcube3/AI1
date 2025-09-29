import math
import unittest

from transformer_blocks import (
    LayerNorm,
    MultiHeadSelfAttention,
    PositionwiseFeedForward,
    TransformerEncoderLayer,
    TransformerEncoder,
)


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


class TestLayerNorm(unittest.TestCase):
    def test_per_position_normalization(self):
        dim = 6
        ln = LayerNorm(dim)
        X = [
            [1.0, 2.0, 3.0, 4.0, -1.0, -2.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ]
        Y = ln(X)
        # Means approx 0, variances approx 1 for each row
        for row in Y:
            m = sum(row) / dim
            var = sum((x - m) * (x - m) for x in row) / dim
            self.assertAlmostEqual(m, 0.0, places=6)
            self.assertAlmostEqual(var, 1.0, places=5)


class TestMultiHeadSelfAttention(unittest.TestCase):
    def test_shapes_and_determinism(self):
        dim = 8
        heads = 2
        mha1 = MultiHeadSelfAttention(dim, heads, seed=42)
        mha2 = MultiHeadSelfAttention(dim, heads, seed=42)
        X = [[0.1 * (i + j) for j in range(dim)] for i in range(4)]

        Y1 = mha1(X)
        Y2 = mha2(X)

        self.assertEqual(len(Y1), len(X))
        self.assertTrue(all(len(row) == dim for row in Y1))
        self.assertEqual(Y1, Y2)

    def test_masking_with_identity_weights(self):
        # Use identity weights/biases so attention is computed on original X.
        dim = 4
        heads = 2  # divides dim
        mha = MultiHeadSelfAttention(dim, heads, seed=0)
        mha.W_q = eye(dim)
        mha.W_k = eye(dim)
        mha.W_v = eye(dim)
        mha.W_o = eye(dim)
        mha.b_q = [0.0] * dim
        mha.b_k = [0.0] * dim
        mha.b_v = [0.0] * dim
        mha.b_o = [0.0] * dim

        # Two tokens with distinct vectors
        x0 = [1.0, 0.0, 0.0, 0.0]
        x1 = [0.0, 1.0, 0.0, 0.0]
        X = [x0, x1]

        # Causal mask: token 0 cannot see token 1
        mask = [
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        Y_masked = mha(X, mask=mask)
        # With mask, output at position 0 should equal v0 (since only attends to itself)
        # Since W_v and W_o are identity, expect exactly x0
        self.assertEqual([round(v, 6) for v in Y_masked[0]], [1.0, 0.0, 0.0, 0.0])

        # Without mask, attention would consider both tokens; ensure it's not exactly x0
        Y_unmasked = mha(X, mask=None)
        self.assertNotEqual([round(v, 6) for v in Y_unmasked[0]], [1.0, 0.0, 0.0, 0.0])


class TestPositionwiseFeedForward(unittest.TestCase):
    def test_relu_identity_when_weights_are_identity(self):
        dim = 6
        ffn = PositionwiseFeedForward(dim, hidden_dim=dim, seed=0)
        # Force identity weights and zero bias: output = ReLU(X)
        ffn.W1 = eye(dim)
        ffn.W2 = eye(dim)
        ffn.b1 = [0.0] * dim
        ffn.b2 = [0.0] * dim

        X = [
            [1.0, -2.0, 0.5, -0.1, 3.0, -4.0],
            [-1.0, 2.0, -0.5, 0.1, -3.0, 4.0],
        ]
        Y = ffn(X)
        expected = [
            [1.0, 0.0, 0.5, 0.0, 3.0, 0.0],
            [0.0, 2.0, 0.0, 0.1, 0.0, 4.0],
        ]
        self.assertEqual([[round(v, 6) for v in row] for row in Y], expected)


class TestTransformerEncoderLayer(unittest.TestCase):
    def test_identity_when_submodules_zero(self):
        dim = 8
        heads = 2
        layer = TransformerEncoderLayer(dim, heads, ff_hidden=8, seed=0)

        # Zero-out attention and FFN so residual adds zeros -> identity mapping
        z = zeros(dim, dim)
        layer.mha.W_q = z
        layer.mha.W_k = z
        layer.mha.W_v = z
        layer.mha.W_o = z
        layer.mha.b_q = [0.0] * dim
        layer.mha.b_k = [0.0] * dim
        layer.mha.b_v = [0.0] * dim
        layer.mha.b_o = [0.0] * dim

        layer.ffn.W1 = zeros(dim, layer.ffn.hidden_dim)
        layer.ffn.W2 = zeros(layer.ffn.hidden_dim, dim)
        layer.ffn.b1 = [0.0] * layer.ffn.hidden_dim
        layer.ffn.b2 = [0.0] * dim

        X = [[0.1 * (i + j) for j in range(dim)] for i in range(5)]
        Y = layer(X)
        self.assertEqual([[round(v, 8) for v in row] for row in Y],
                         [[round(v, 8) for v in row] for row in X])

    def test_shapes_and_determinism(self):
        dim = 12
        heads = 3
        layer1 = TransformerEncoderLayer(dim, heads, ff_hidden=24, seed=123)
        layer2 = TransformerEncoderLayer(dim, heads, ff_hidden=24, seed=123)
        X = [[(i + 1) * 0.01 * (j + 1) for j in range(dim)] for i in range(7)]
        Y1 = layer1(X)
        Y2 = layer2(X)
        self.assertEqual(len(Y1), len(X))
        self.assertTrue(all(len(row) == dim for row in Y1))
        self.assertEqual(Y1, Y2)


class TestTransformerEncoder(unittest.TestCase):
    def test_stack_identity_when_zeroed(self):
        dim = 8
        enc = TransformerEncoder(num_layers=3, dim=dim, num_heads=2, ff_hidden=16, seed=7)
        # Zero-out all layers (MHA + FFN)
        for layer in enc.layers:
            z = zeros(dim, dim)
            layer.mha.W_q = z
            layer.mha.W_k = z
            layer.mha.W_v = z
            layer.mha.W_o = z
            layer.mha.b_q = [0.0] * dim
            layer.mha.b_k = [0.0] * dim
            layer.mha.b_v = [0.0] * dim
            layer.mha.b_o = [0.0] * dim

            layer.ffn.W1 = zeros(dim, layer.ffn.hidden_dim)
            layer.ffn.W2 = zeros(layer.ffn.hidden_dim, dim)
            layer.ffn.b1 = [0.0] * layer.ffn.hidden_dim
            layer.ffn.b2 = [0.0] * dim

        X = [[(i + j) * 0.05 for j in range(dim)] for i in range(4)]
        Y = enc(X)
        self.assertEqual([[round(v, 8) for v in row] for row in Y],
                         [[round(v, 8) for v in row] for row in X])

    def test_mask_flow(self):
        dim = 8
        enc = TransformerEncoder(num_layers=2, dim=dim, num_heads=2, ff_hidden=16, seed=1)
        X = [[0.1 * (i + j) for j in range(dim)] for i in range(5)]
        # simple lower-triangular mask
        n = len(X)
        mask = [[1.0 if j <= i else 0.0 for j in range(n)] for i in range(n)]
        # Ensure it runs and returns correct shape
        Y = enc(X, mask=mask)
        self.assertEqual(len(Y), n)
        self.assertTrue(all(len(row) == dim for row in Y))


if __name__ == "__main__":
    unittest.main()
import math
import unittest

from transformer_blocks import (
    LayerNorm,
    MultiHeadSelfAttention,
    PositionwiseFeedForward,
    TransformerEncoderLayer,
    TransformerEncoder,
    generate_causal_mask,
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

    def test_various_head_configs_and_invalid(self):
        dim = 12
        for h in [1, 2, 3, 4, 6, 12]:
            mha = MultiHeadSelfAttention(dim, h, seed=1)
            X = [[0.01 * (i + j) for j in range(dim)] for i in range(5)]
            Y = mha(X)
            self.assertEqual(len(Y), len(X))
            self.assertTrue(all(len(row) == dim for row in Y))
        # invalid heads (not dividing dim) should raise
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(dim, 5)
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(dim, 0)

    def test_numerical_stability_large_values(self):
        dim = 8
        heads = 2
        mha = MultiHeadSelfAttention(dim, heads, seed=2)
        # Large magnitude inputs
        X = [[(i + 1) * (j + 1) * 1e3 for j in range(dim)] for i in range(4)]
        Y = mha(X)
        # Ensure outputs are finite
        for row in Y:
            for v in row:
                self.assertTrue(math.isfinite(v))


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
        mask = generate_causal_mask(n)
        # Ensure it runs and returns correct shape
        Y = enc(X, mask=mask)
        self.assertEqual(len(Y), n)
        self.assertTrue(all(len(row) == dim for row in Y))


 class TestMasks(unittest.TestCase):
    def test_generate_causal_mask(self):
        for n in [0, 1, 2, 5]:
            m = generate_causal_mask(n)
            self.assertEqual(len(m), n)
            for i in range(n):
                self.assertEqual(len(m[i]), n)
                for j in range(n):
                    expected = 1.0 if j <= i else 0.0
                    self.assertEqual(m[i][j], expected)
        with self.assertRaises(ValueError):
            generate_causal_mask(-1)

    def test_padding_and_causal_padding_masks(self):
        from transformer_blocks import (
            generate_padding_mask,
            generate_causal_padding_mask,
            generate_causal_masks_from_lengths,
            generate_padding_mask_from_flags,
            generate_causal_padding_mask_from_flags,
            build_flags_from_tokens,
        )

        # padding-only mask
        m = generate_padding_mask(seq_len=5, valid_len=3)
        expected = [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self.assertEqual(m, [[float(v) for v in row] for row in expected])

        # padding-only from flags (same as valid_len=3 -> flags [F,F,F,T,T])
        flags = [False, False, False, True, True]
        m_flags = generate_padding_mask_from_flags(flags)
        self.assertEqual(m_flags, m)

        # causal+padding
        mc = generate_causal_padding_mask(seq_len=5, valid_len=3)
        expected_c = [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self.assertEqual(mc, [[float(v) for v in row] for row in expected_c])

        # causal+padding from flags
        mc_flags = generate_causal_padding_mask_from_flags(flags)
        self.assertEqual(mc_flags, mc)

        # build flags from tokens
        tokens = ["a", "b", "<pad>", "<pad>"]
        flags_from_tokens = build_flags_from_tokens(tokens)
        self.assertEqual(flags_from_tokens, [False, False, True, True])
        # custom pad token
        tokens2 = ["a", "b", "<x>", "<x>"]
        flags2 = build_flags_from_tokens(tokens2, pad_token="<x>")
        self.assertEqual(flags2, [False, False, True, True])

        # make causal mask directly from tokens should match causal mask from flags
        from transformer_blocks import make_causal_mask_from_tokens
        mc_from_tokens = make_causal_mask_from_tokens(tokens)
        mc_from_flags = generate_causal_padding_mask_from_flags(flags_from_tokens)
        self.assertEqual(mc_from_tokens, mc_from_flags)

        # lengths helper
        masks = generate_causal_masks_from_lengths([0, 2, 3])
        self.assertEqual(len(masks), 3)
        self.assertEqual(len(masks[0]), 3)  # max length
        self.assertEqual(len(masks[0][0]), 3)
        # first mask with length 0 should be all zeros
        self.assertTrue(all(all(v == 0.0 for v in row) for row in masks[0]))

        # invalid args
        with self.assertRaises(ValueError):
            generate_padding_mask(3, 4)
        with self.assertRaises(ValueError):
            generate_causal_padding_mask(3, -1)


 if __name__ == "__main__":
    unittest.main()
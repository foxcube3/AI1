from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple


# --------- Small list-based linear algebra helpers (no external deps) ---------


def _matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    """Matrix multiply: A [m x n] x B [n x p] -> [m x p]."""
    if not A or not B:
        return []
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    if n != n2:
        raise ValueError(f"matmul: inner dims mismatch {n} vs {n2}")
    out: List[List[float]] = [[0.0] * p for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(p):
                out[i][j] += aik * Bk[j]
    return out


def _matvec(A: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    """Matrix-vector multiply: A [m x n] x x [n] -> [m]."""
    if not A:
        return []
    m, n = len(A), len(A[0])
    if len(x) != n:
        raise ValueError(f"matvec: dims mismatch {n} vs {len(x)}")
    out = [0.0] * m
    for i in range(m):
        s = 0.0
        Ai = A[i]
        for k in range(n):
            s += Ai[k] * x[k]
        out[i] = s
    return out


def _transpose(A: Sequence[Sequence[float]]) -> List[List[float]]:
    if not A:
        return []
    m, n = len(A), len(A[0])
    out: List[List[float]] = [[0.0] * m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            out[j][i] = A[i][j]
    return out


def _add_bias(rows: Sequence[Sequence[float]], b: Sequence[float]) -> List[List[float]]:
    if not rows:
        return []
    if len(rows[0]) != len(b):
        raise ValueError(f"bias add: dim mismatch {len(rows[0])} vs {len(b)}")
    out: List[List[float]] = []
    for r in rows:
        out.append([rj + bj for rj, bj in zip(r, b)])
    return out


def _relu(rows: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[(x if x > 0.0 else 0.0) for x in r] for r in rows]


def _softmax(rows: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for r in rows:
        if not r:
            out.append([])
            continue
        m = max(r)
        exps = [math.exp(x - m) for x in r]
        s = sum(exps)
        if s == 0.0:
            out.append([1.0 / len(r)] * len(r))
        else:
            out.append([e / s for e in exps])
    return out


def _split_heads(X: Sequence[Sequence[float]], num_heads: int) -> List[List[List[float]]]:
    """
    Split [seq_len x dim] into [num_heads x seq_len x head_dim].
    """
    if not X:
        return [[] for _ in range(num_heads)]
    seq_len, dim = len(X), len(X[0])
    if dim % num_heads != 0:
        raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
    head_dim = dim // num_heads
    out: List[List[List[float]]] = []
    for h in range(num_heads):
        start = h * head_dim
        end = start + head_dim
        out.append([row[start:end] for row in X])
    return out


def _combine_heads(heads: Sequence[Sequence[Sequence[float]]]) -> List[List[float]]:
    """
    Combine [num_heads x seq_len x head_dim] -> [seq_len x (num_heads*head_dim)].
    """
    if not heads:
        return []
    num_heads = len(heads)
    seq_len = len(heads[0])
    head_dim = len(heads[0][0]) if seq_len > 0 else 0
    out: List[List[float]] = []
    for t in range(seq_len):
        row: List[float] = []
        for h in range(num_heads):
            row.extend(heads[h][t])
        out.append(row)
    return out


def _scaled_dot_product_attention(
    Q: Sequence[Sequence[float]],
    K: Sequence[Sequence[float]],
    V: Sequence[Sequence[float]],
    mask: Optional[Sequence[Sequence[float]]] = None,
) -> List[List[float]]:
    """
    Q: [seq_len x d_k], K: [seq_len x d_k], V: [seq_len x d_v]
    Returns: [seq_len x d_v]
    """
    d_k = len(Q[0]) if Q else 0
    # scores = Q @ K^T
    scores = _matmul(Q, _transpose(K))
    scale = 1.0 / math.sqrt(max(1, d_k))
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            scores[i][j] *= scale
            if mask is not None:
                # mask[i][j] == 0.0 means disallow; use large negative
                if mask[i][j] <= 0.0:
                    scores[i][j] = -1e30
    attn = _softmax(scores)
    return _matmul(attn, V)


def _init_matrix(rows: int, cols: int, rng: random.Random, scheme: str) -> List[List[float]]:
    s = scheme.lower()
    if s == "zeros":
        return [[0.0 for _ in range(cols)] for _ in range(rows)]
    elif s in {"uniform", "xavier_uniform"}:
        if s == "uniform":
            limit = 1.0 / math.sqrt(cols if cols > 0 else 1)
        else:
            limit = math.sqrt(6.0 / float((rows + cols) if (rows + cols) > 0 else 1))
        return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]
    else:
        raise ValueError(f"Unknown init scheme: {scheme}")


def _init_vector(size: int, val: float = 0.0) -> List[float]:
    return [val for _ in range(size)]


# ---------------------------- Core building blocks ----------------------------


class LayerNorm:
    """
    Minimal layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
    Operates on a single vector or a sequence (applied per position).
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.eps = eps
        self.gamma: List[float] = [1.0] * dim
        self.beta: List[float] = [0.0] * dim

    def _normalize_vec(self, x: Sequence[float]) -> List[float]:
        if len(x) != self.dim:
            raise ValueError(f"LayerNorm dim mismatch: expected {self.dim}, got {len(x)}")
        m = sum(x) / self.dim
        var = sum((xi - m) * (xi - m) for xi in x) / self.dim
        denom = 1.0 / math.sqrt(var + self.eps)
        return [(xi - m) * denom * g + b for xi, g, b in zip(x, self.gamma, self.beta)]

    def __call__(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self._normalize_vec(row) for row in X]


class MultiHeadSelfAttention:
    """
    Dependency-free multi-head self-attention with list-based math.

    Input/Output shape: [seq_len x dim]
    - num_heads must divide dim.
    - Optional boolean/float mask [seq_len x seq_len]; values <= 0.0 are treated as masked.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        seed: Optional[int] = None,
        init: str = "xavier_uniform",
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide dim")
        self.dim = dim
        self.num_heads = num_heads
        self._rng = random.Random(seed)
        # Single shared projection matrices for all heads (standard implementation).
        self.W_q: List[List[float]] = _init_matrix(dim, dim, self._rng, init)
        self.W_k: List[List[float]] = _init_matrix(dim, dim, self._rng, init)
        self.W_v: List[List[float]] = _init_matrix(dim, dim, self._rng, init)
        self.W_o: List[List[float]] = _init_matrix(dim, dim, self._rng, init)
        self.b_q: List[float] = _init_vector(dim, 0.0)
        self.b_k: List[float] = _init_vector(dim, 0.0)
        self.b_v: List[float] = _init_vector(dim, 0.0)
        self.b_o: List[float] = _init_vector(dim, 0.0)

    def forward(
        self,
        X: Sequence[Sequence[float]],
        mask: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[List[float]]:
        """
        X: [seq_len x dim]; mask: [seq_len x seq_len] with <=0 as masked
        """
        if not X:
            return []
        if len(X[0]) != self.dim:
            raise ValueError(f"Input dim mismatch: expected {self.dim}, got {len(X[0])}")

        # Projections
        Q = _add_bias(_matmul(X, self.W_q), self.b_q)
        K = _add_bias(_matmul(X, self.W_k), self.b_k)
        V = _add_bias(_matmul(X, self.W_v), self.b_v)

        # Split into heads
        Qh = _split_heads(Q, self.num_heads)  # [H x T x Dh]
        Kh = _split_heads(K, self.num_heads)
        Vh = _split_heads(V, self.num_heads)

        # Attention per head
        head_outs: List[List[List[float]]] = []
        for h in range(self.num_heads):
            head_outs.append(_scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], mask=mask))

        # Combine heads and project out
        combined = _combine_heads(head_outs)  # [T x D]
        out = _add_bias(_matmul(combined, self.W_o), self.b_o)
        return out

    __call__ = forward


class PositionwiseFeedForward:
    """
    FFN: FFN(x) = W2 * act(W1 * x + b1) + b2
    Operates on [seq_len x dim]; hidden_dim is the inner dimension.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        activation: str = "relu",
        seed: Optional[int] = None,
        init: str = "xavier_uniform",
    ) -> None:
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.activation = activation.lower()
        self._rng = random.Random(seed)

        self.W1: List[List[float]] = _init_matrix(dim, hidden_dim, self._rng, init)
        self.b1: List[float] = _init_vector(hidden_dim, 0.0)
        self.W2: List[List[float]] = _init_matrix(hidden_dim, dim, self._rng, init)
        self.b2: List[float] = _init_vector(dim, 0.0)

    def forward(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        if not X:
            return []
        if len(X[0]) != self.dim:
            raise ValueError(f"FFN input dim mismatch: expected {self.dim}, got {len(X[0])}")
        H = _add_bias(_matmul(X, self.W1), self.b1)
        if self.activation == "relu":
            H = _relu(H)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        Y = _add_bias(_matmul(H, self.W2), self.b2)
        return Y

    __call__ = forward


class TransformerEncoderLayer:
    """
    A single Transformer encoder layer with pre-norm architecture:

    x = x + MHA(LN1(x))
    x = x + FFN(LN2(x))

    Shapes are list-based with X: [seq_len x dim].
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_hidden: int,
        *,
        seed: Optional[int] = None,
        init: str = "xavier_uniform",
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        # Seed splitting for deterministic construction
        rng_base = random.Random(seed).randint(0, 2**31 - 1)
        self.ln1 = LayerNorm(dim)
        self.mha = MultiHeadSelfAttention(dim, num_heads, seed=rng_base, init=init)
        self.ln2 = LayerNorm(dim)
        self.ffn = PositionwiseFeedForward(dim, ff_hidden, seed=rng_base ^ 0x9E3779B1, init=init)

    def forward(
        self,
        X: Sequence[Sequence[float]],
        mask: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[List[float]]:
        if not X:
            return []
        if len(X[0]) != self.dim:
            raise ValueError(f"EncoderLayer input dim mismatch: expected {self.dim}, got {len(X[0])}")

        # Pre-norm + residual around attention
        x1 = self.ln1(X)
        attn_out = self.mha(x1, mask=mask)
        x_res1: List[List[float]] = [[a + b for a, b in zip(xi, ai)] for xi, ai in zip(X, attn_out)]

        # Pre-norm + residual around FFN
        x2 = self.ln2(x_res1)
        ffn_out = self.ffn(x2)
        x_res2: List[List[float]] = [[a + b for a, b in zip(xi, fi)] for xi, fi in zip(x_res1, ffn_out)]
        return x_res2

    __call__ = forward


class TransformerEncoder:
    """
    Stack of TransformerEncoderLayer blocks.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        ff_hidden: int,
        *,
        seed: Optional[int] = None,
        init: str = "xavier_uniform",
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.dim = dim
        self.layers: List[TransformerEncoderLayer] = []
        # Derive a simple sequence of seeds for determinism
        base_rng = random.Random(seed)
        for i in range(num_layers):
            layer_seed = base_rng.randint(0, 2**31 - 1)
            self.layers.append(
                TransformerEncoderLayer(dim, num_heads, ff_hidden, seed=layer_seed, init=init)
            )

    def forward(
        self,
        X: Sequence[Sequence[float]],
        mask: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[List[float]]:
        out = list(list(row) for row in X)  # shallow copy to avoid mutating input
        for layer in self.layers:
            out = layer(out, mask=mask)
        return out

    __call__ = forward
from __future__ import annotations

import json
import math
import random
from typing import List, Optional, Sequence


class SinusoidalPositionalEncoding:
    """
    Dependency-free sinusoidal positional encoding (Transformer-style).

    For position p and dimension i:
      PE[p, 2i]   = sin(p / (base ** (2i / dim)))
      PE[p, 2i+1] = cos(p / (base ** (2i / dim)))

    - dim: embedding dimension (must be > 0)
    - base: scaling base (typically 10000.0)
    """

    def __init__(self, dim: int, *, base: float = 10000.0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.base = float(base)

        # Precompute denominators for efficiency: base ** (2i/dim)
        self._div_terms: List[float] = [
            self.base ** (2.0 * (i // 2) / self.dim) for i in range(self.dim)
        ]

    def encode(self, length: int, *, offset: int = 0) -> List[List[float]]:
        """
        Create a [length x dim] positional encoding matrix starting at position `offset`.
        """
        if length < 0:
            raise ValueError("length must be non-negative")

        dim = self.dim
        out: List[List[float]] = []
        for p in range(offset, offset + length):
            row = [0.0] * dim
            for i in range(0, dim, 2):
                angle = p / self._div_terms[i]
                row[i] = math.sin(angle)
                if i + 1 < dim:
                    row[i + 1] = math.cos(angle)
            out.append(row)
        return out

    def add_to(self, embeddings: Sequence[Sequence[float]], *, offset: int = 0) -> List[List[float]]:
        """
        Add positional encodings to an existing embedding sequence.
        Returns a new list (does not mutate input).
        """
        if not embeddings:
            return []
        length = len(embeddings)
        dim = self.dim
        pe = self.encode(length, offset=offset)
        out: List[List[float]] = []
        for vec, pos in zip(embeddings, pe):
            if len(vec) != dim:
                raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {len(vec)}")
            out.append([a + b for a, b in zip(vec, pos)])
        return out


class LearnedPositionalEmbedding:
    """
    Learned positional embeddings with a fixed maximum length.

    - dim: embedding dimension (>0)
    - max_len: maximum supported sequence length (>0)
    - init: 'zeros' | 'uniform' | 'xavier_uniform'
    - seed: optional seed for deterministic initialization

    Methods mirror SinusoidalPositionalEncoding: encode(length, offset) and add_to(embeddings, offset).
    """

    def __init__(
        self,
        dim: int,
        max_len: int,
        *,
        init: str = "xavier_uniform",
        seed: Optional[int] = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.dim = dim
        self.max_len = max_len
        self._rng = random.Random(seed)
        self.weights: List[List[float]] = self._init_weights(max_len, dim, init)

    def _init_weights(self, rows: int, cols: int, init: str) -> List[List[float]]:
        init_l = init.lower()
        if init_l == "zeros":
            return [[0.0 for _ in range(cols)] for _ in range(rows)]
        elif init_l in {"uniform", "xavier_uniform"}:
            if init_l == "uniform":
                limit = 1.0 / math.sqrt(cols)
            else:
                limit = math.sqrt(6.0 / (cols + cols))
            return [[self._rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]
        else:
            raise ValueError(f"Unknown init scheme: {init}")

    def encode(self, length: int, *, offset: int = 0) -> List[List[float]]:
        """
        Return [length x dim] learned positional embeddings starting at position `offset`.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        end = offset + length
        if end > self.max_len:
            raise ValueError(f"Requested positions up to {end-1}, but max_len is {self.max_len}")
        return [self.weights[p] for p in range(offset, end)]

    def add_to(self, embeddings: Sequence[Sequence[float]], *, offset: int = 0) -> List[List[float]]:
        """
        Add learned positional embeddings to an existing embedding sequence.
        """
        if not embeddings:
            return []
        length = len(embeddings)
        dim = self.dim
        pe = self.encode(length, offset=offset)
        out: List[List[float]] = []
        for vec, pos in zip(embeddings, pe):
            if len(vec) != dim:
                raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {len(vec)}")
            out.append([a + b for a, b in zip(vec, pos)])
        return out

    # -------- Persistence --------

    def save_weights(self, path: str) -> None:
        """
        Save learned positional embedding weights and metadata to JSON.
        """
        payload = {
            "dim": self.dim,
            "max_len": self.max_len,
            "weights": self.weights,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load_weights(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.dim = int(payload["dim"])
        self.max_len = int(payload["max_len"])
        self.weights = payload["weights"]
        if any(len(row) != self.dim for row in self.weights):
            raise ValueError("Loaded positional weights have inconsistent dimensions")
        if len(self.weights) != self.max_len:
            # Keep safety check but allow correcting max_len to actual length for robustness
            self.max_len = len(self.weights)
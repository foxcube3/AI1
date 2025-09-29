from __future__ import annotations

import math
from typing import List, Sequence


class SinusoidalPositionalEncoding:
    """
    Dependency-free sinusoidal positional encoding (Transformer-style).

    For position p and dimension i:
      PE[p, 2i]   = sin(p / (base ** (2i / dim)))
      PE[p, 2i+1] = cos(p / (base ** (2i / dim)))

    - dim: embedding dimension (must be &gt; 0)
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
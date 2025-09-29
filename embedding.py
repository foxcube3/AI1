from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


class EmbeddingLayer:
    """
    A minimal embedding layer compatible with BPETokenizer output.

    - Accepts a vocabulary mapping token -> index (e.g., BPETokenizer.vocab)
    - Maps lists of tokens (as produced by BPETokenizer.encode) to:
        * token ids (integers)
        * dense embedding vectors
    - No external dependencies.

    Notes:
    - Unknown tokens (not present in vocab) are mapped to a dedicated <unk> index
      allocated internally. This does not modify the original vocab file on disk.
    - Weights are stored as python lists of floats to avoid external deps.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        dim: int,
        *,
        seed: Optional[int] = None,
        init: str = "xavier_uniform",
        unk_token: str = "<unk>",
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")

        # Copy vocab to avoid accidental external mutation
        self.token_to_id: Dict[str, int] = dict(vocab)
        self.dim = dim
        self.unk_token = unk_token

        # Reserve an internal index for UNK if not present
        if self.unk_token in self.token_to_id:
            self.unk_id = self.token_to_id[self.unk_token]
            self.has_external_unk = True
        else:
            self.unk_id = max(self.token_to_id.values(), default=-1) + 1
            self.has_external_unk = False

        self.vocab_size = max(self.token_to_id.values(), default=-1) + 1
        if self.unk_id >= self.vocab_size:
            # Extend logical size to include our unk slot
            self.vocab_size = self.unk_id + 1

        # Initialize RNG
        self._rng = random.Random(seed)

        # Initialize weights
        self.weights: List[List[float]] = self._init_weights(self.vocab_size, dim, init)

        # If vocab didn't include unk, ensure it has initialized weights as the last row
        # (already covered by _init_weights given vocab_size includes unk slot).

    def _init_weights(self, rows: int, cols: int, init: str) -> List[List[float]]:
        init = init.lower()
        if init == "zeros":
            return [[0.0 for _ in range(cols)] for _ in range(rows)]
        elif init in {"uniform", "xavier_uniform"}:
            if init == "uniform":
                limit = 1.0 / math.sqrt(cols)
            else:
                # Xavier/Glorot uniform
                # fan_in = fan_out = cols for embeddings
                limit = math.sqrt(6.0 / (cols + cols))
            return [[self._rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]
        else:
            raise ValueError(f"Unknown init scheme: {init}")

    # -------- Token/id utilities --------

    def tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        """
        Map tokens to ids, using unk_id for OOV tokens.
        """
        lookup = self.token_to_id
        unk = self.unk_id
        return [lookup.get(t, unk) for t in tokens]

    def ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        """
        Best-effort reverse mapping. If unk was not in the original vocab, it won't
        be present in the reverse map and will appear as <unk>.
        """
        id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}
        if not self.has_external_unk:
            id_to_token[self.unk_id] = self.unk_token
        return [id_to_token.get(i, self.unk_token) for i in ids]

    # -------- Forward APIs --------

    def embed_ids(self, ids: Sequence[int]) -> List[List[float]]:
        """
        Lookup embeddings for a sequence of ids.
        """
        W = self.weights
        vocab_size = len(W)
        dim = self.dim
        out: List[List[float]] = []
        for i in ids:
            if 0 <= i < vocab_size:
                out.append(W[i])
            else:
                # Out of range -> map to unk
                out.append(W[self.unk_id])
        return out

    def embed_tokens(self, tokens: Sequence[str]) -> List[List[float]]:
        """
        Convenience: tokens -> ids -> embeddings.
        """
        return self.embed_ids(self.tokens_to_ids(tokens))

    def embed_batch(self, batch_tokens: Sequence[Sequence[str]]) -> List[List[List[float]]]:
        """
        Batch embedding for a list of token sequences.
        """
        return [self.embed_tokens(seq) for seq in batch_tokens]

    # -------- Persistence --------

    def save_weights(self, path: str) -> None:
        """
        Save weights and minimal metadata to JSON. Keeps things simple and dependency-free.
        """
        payload = {
            "dim": self.dim,
            "unk_token": self.unk_token,
            "unk_id": self.unk_id,
            "has_external_unk": self.has_external_unk,
            "weights": self.weights,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load_weights(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.dim = int(payload["dim"])
        self.unk_token = str(payload["unk_token"])
        self.unk_id = int(payload["unk_id"])
        self.has_external_unk = bool(payload["has_external_unk"])
        self.weights = payload["weights"]
        # Sanity
        self.vocab_size = len(self.weights)
        if any(len(row) != self.dim for row in self.weights):
            raise ValueError("Loaded weights have inconsistent dimensions")

    # -------- Factory helpers --------

    @classmethod
    def from_vocab_file(
        cls,
        vocab_path: str,
        dim: int,
        *,
        seed: Optional[int] = None,
        init: str = "xavier_uniform",
        unk_token: str = "<unk>",
    ) -> "EmbeddingLayer":
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab, dim=dim, seed=seed, init=init, unk_token=unk_token)


# Simple demo utility when run as a script
def _demo() -> None:
    """
    Minimal demo:
    - expects bpe_vocab.json in current directory
    - reads a line of text, tokenizes with naive whitespace split (since BPETokenizer is separate)
    - prints ids and first 3 dims of embeddings
    """
    vocab_path = os.getenv("BPE_VOCAB", "bpe_vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Vocab file not found: {vocab_path}")
        return

    emb = EmbeddingLayer.from_vocab_file(vocab_path, dim=8, seed=42)
    text = "Allen allows ample analysis"
    tokens = text.split()
    ids = emb.tokens_to_ids(tokens)
    vecs = emb.embed_tokens(tokens)
    print("Tokens:", tokens)
    print("Ids:", ids)
    print("Embeddings[0..2 dims]:", [[round(x, 4) for x in v[:3]] for v in vecs])


if __name__ == "__main__":
    _demo()
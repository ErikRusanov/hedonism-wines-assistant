"""Deterministic BM25-style sparse encoder shared by indexing (I-3) and query (I-5).

The same fitted encoder MUST be used on both sides — index and query — or the
sparse channel suffers train/serve skew. I-3 is the only place the encoder is
fitted and persisted (``data/sparse_encoder.json``); I-5 loads that exact file.

Determinism is non-negotiable: token -> dimension uses a stable ``blake2b`` hash
(never Python's salted ``hash()``, which differs between processes), and the IDF
table is persisted, so encoding a text yields identical vectors across runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# BM25 term-frequency saturation constant. TF contributes with diminishing
# returns: tf * (k1 + 1) / (tf + k1). No document-length normalisation — the
# "passport" texts are short and uniform, so length bias is negligible.
_K1: Final = 1.5

# Tokens shorter than this carry no retrieval signal (single letters/digits).
_MIN_TOKEN_LEN: Final = 2

_TOKEN_SPLIT: Final = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on any non-alphanumeric run; drop length-1 tokens."""
    return [tok for tok in _TOKEN_SPLIT.split(text.lower()) if len(tok) >= _MIN_TOKEN_LEN]


def _hash_token(token: str, vocab_size: int) -> int:
    """Map a token to a stable sparse dimension via blake2b (process-independent)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % vocab_size


@dataclass(frozen=True, slots=True)
class SparseEncoder:
    """Encodes text into a sparse ``(indices, values)`` vector using BM25 weights.

    ``idf`` is the fitted inverse-document-frequency table keyed by token. When it
    is empty (no ``fit`` was run) the encoder falls back to saturated raw term
    frequencies, which keeps it usable fully offline and in unit tests.
    """

    idf: dict[str, float] = field(default_factory=dict)
    vocab_size: int = 2**20

    @classmethod
    def fit(cls, corpus: Iterable[str]) -> SparseEncoder:
        """Build the IDF table from a corpus of texts (the cards' embedding text)."""
        document_freq: Counter[str] = Counter()
        n_docs = 0
        for text in corpus:
            n_docs += 1
            document_freq.update(set(_tokenize(text)))
        # BM25 IDF: ln(1 + (N - n + 0.5) / (n + 0.5)) — always positive.
        idf = {
            token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in document_freq.items()
        }
        return cls(idf=idf)

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """Return parallel ``(indices, values)`` lists; duplicate indices summed."""
        weights: dict[int, float] = {}
        for token, tf in Counter(_tokenize(text)).items():
            tf_saturated = tf * (_K1 + 1.0) / (tf + _K1)
            # Fitted: unseen terms have no document support (idf 0). No fit:
            # weight by saturated TF alone so the encoder works offline.
            idf = self.idf.get(token, 0.0) if self.idf else 1.0
            weight = tf_saturated * idf
            if weight == 0.0:
                continue
            index = _hash_token(token, self.vocab_size)
            weights[index] = weights.get(index, 0.0) + weight
        indices = list(weights.keys())
        values = [weights[index] for index in indices]
        return indices, values

    def save(self, path: str | Path) -> None:
        """Persist the fitted IDF so the query side (I-5) loads the same encoder."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"vocab_size": self.vocab_size, "idf": self.idf}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SparseEncoder:
        """Load a previously :meth:`save`-d encoder."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(idf=data["idf"], vocab_size=data["vocab_size"])

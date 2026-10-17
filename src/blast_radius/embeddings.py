"""Text -> vector, with a real model when one is installed and a deterministic fallback when not.

The fallback matters more than it looks. FalkorDB's vector index is a real feature of the database
and the demo has to exercise it, but a demo that cannot start because a 90 MB model download failed
on conference wifi is worse than one with approximate embeddings. So:

* `fastembed` installed -> real sentence embeddings (BAAI/bge-small-en-v1.5, 384 dims).
* not installed -> a hashed bag-of-trigrams projection into the same 384 dims.

Both are L2-normalised, so cosine similarity behaves and the *same* index, the same query and the
same Cypher serve either one. Which is in use is reported by `describe()` and must be stated in the
README -- the hashed vectors retain enough lexical signal to rank "pricing discount 5xx" above
"cart cache eviction", which is all the recall demo needs, but they are not semantic and claiming
otherwise would be dishonest.

The dimension is fixed at config time because the vector index declares it at creation: switching
backends without re-creating the index would silently mismatch.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Any, Sequence

from .config import SETTINGS

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


# --------------------------------------------------------------------------------------
# Real backend, if available
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _fastembed_model() -> Any | None:
    try:
        from fastembed import TextEmbedding  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001  -- not installed, or no network on first use
        return None
    try:
        return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception:  # noqa: BLE001  -- model download failed; fall back rather than crash
        return None


def backend() -> str:
    return "fastembed:BAAI/bge-small-en-v1.5" if _fastembed_model() is not None else "hashed-trigram"


def is_semantic() -> bool:
    """True only when a real model is doing the work. The README should print this."""
    return _fastembed_model() is not None


# --------------------------------------------------------------------------------------
# Fallback backend
# --------------------------------------------------------------------------------------

def _hashed(text: str, dim: int) -> list[float]:
    """Deterministic hashed projection of word unigrams and character trigrams.

    Unigrams carry the topical words; character trigrams give partial-match tolerance
    ("pricing" vs "prices"), which a pure unigram hash would miss entirely.
    """
    vector = [0.0] * dim
    words = _tokens(text)

    def bump(feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * weight

    for word in words:
        bump(f"w:{word}", 1.0)
        padded = f"^{word}$"
        for i in range(len(padded) - 2):
            bump(f"t:{padded[i:i + 3]}", 0.35)
    for a, b in zip(words, words[1:]):
        bump(f"b:{a}_{b}", 0.6)

    return vector


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        # An all-zero vector would make cosine similarity undefined, and the index rejects it.
        # One deterministic non-zero component keeps empty text harmless.
        out = [0.0] * len(vector)
        out[0] = 1.0
        return out
    return [v / norm for v in vector]


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def embed(text: str, dim: int | None = None) -> list[float]:
    """One embedding, L2-normalised, always `dim` long."""
    dimension = dim or SETTINGS.embedding_dim
    model = _fastembed_model()
    if model is not None:
        try:
            vector = list(next(iter(model.embed([text or ""]))))
            if len(vector) == dimension:
                return _normalise([float(v) for v in vector])
            # A model whose width disagrees with the index is a configuration error, not something
            # to paper over by truncating: fall back loudly rather than corrupt the index.
        except Exception:  # noqa: BLE001
            pass
    return _normalise(_hashed(text or "", dimension))


def embed_many(texts: Sequence[str], dim: int | None = None) -> list[list[float]]:
    dimension = dim or SETTINGS.embedding_dim
    model = _fastembed_model()
    if model is not None:
        try:
            vectors = [list(v) for v in model.embed(list(texts))]
            if all(len(v) == dimension for v in vectors):
                return [_normalise([float(x) for x in v]) for v in vectors]
        except Exception:  # noqa: BLE001
            pass
    return [embed(text, dimension) for text in texts]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Only for tests and diagnostics -- real similarity is computed by FalkorDB's index."""
    return sum(x * y for x, y in zip(a, b))


def describe() -> dict[str, Any]:
    return {
        "backend": backend(),
        "semantic": is_semantic(),
        "dimension": SETTINGS.embedding_dim,
        "note": (
            "Real sentence embeddings."
            if is_semantic()
            else "Deterministic hashed fallback: lexical, not semantic. "
                 "Install fastembed for real embeddings."
        ),
    }

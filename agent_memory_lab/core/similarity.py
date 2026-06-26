"""Zero-dependency, deterministic text embedding + cosine.

We use signed feature hashing so the whole lab runs offline and reproducibly
(no model download, no numpy). Lexical overlap drives similarity, which is
enough for the synthetic StaleBench workloads. Swap in a real embedder via the
``mlx`` provider for production-grade retrieval.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[a-z0-9\+#]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def embed_hash(text: str, dim: int = 256) -> list[float]:
    """Deterministic signed-hashing embedding, L2-normalized."""
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors (assumed L2-normalized => cosine similarity)."""
    return sum(x * y for x, y in zip(a, b))

"""Deterministic, offline provider — the lab's default."""

from __future__ import annotations

from .base import Embedder, LLM
from ..core.similarity import embed_hash


class MockEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        return embed_hash(text, dim=self.dim)


class MockLLM(LLM):
    """No real generation needed: the deterministic extractor is rule-based,
    and StaleBench answers are read straight from retrieved triples. This stub
    exists only so the ``mlx`` <-> ``mock`` interface stays symmetric."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - unused in mock path
        return ""

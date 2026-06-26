"""Provider abstraction: an embedder + an LLM.

Two providers ship:
  * ``mock`` — deterministic, zero-dependency, offline. Used by tests and the
    default benchmark so results are reproducible and CI-friendly.
  * ``mlx`` — real models on Apple Silicon (Metal) via mlx-lm + mlx-embeddings.

Selecting a provider only changes the *quality* of embedding/extraction, never
the memory semantics — so the staleness story is provider-independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Embedder(ABC):
    """Text -> vector, with per-instance memoization for determinism + speed."""

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        v = self._cache.get(text)
        if v is None:
            v = self._embed(text)
            self._cache[text] = v
        return v

    @abstractmethod
    def _embed(self, text: str) -> list[float]:
        ...


class LLM(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...


@dataclass
class Provider:
    name: str
    embedder: Embedder
    llm: LLM


def get_provider(name: str = "mock") -> Provider:
    name = (name or "mock").lower()
    if name == "mock":
        from .mock import MockEmbedder, MockLLM
        return Provider("mock", MockEmbedder(), MockLLM())
    if name == "mlx":
        from .mlx_provider import MLXEmbedder, MLXLLM
        return Provider("mlx", MLXEmbedder(), MLXLLM())
    if name == "mlx-hash":
        # real MLX LLM (extraction + answering) + deterministic hash embeddings,
        # so retrieval needs no embedding-model download.
        from .mock import MockEmbedder
        from .mlx_provider import MLXLLM
        return Provider("mlx-hash", MockEmbedder(), MLXLLM())
    raise ValueError(f"unknown provider {name!r} (use 'mock', 'mlx', or 'mlx-hash')")

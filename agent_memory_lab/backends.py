"""Backend registry — each backend is a composition of the 4 modules.

This IS the ablation playground: a backend is nothing but a choice of
{extractor, storage, maintainer, retriever}. Toggle one part to see its effect
(e.g. `bitemporal-nomaint` keeps the time-aware reader but removes maintenance,
and staleness comes right back — proving maintenance is the cause).
"""

from __future__ import annotations

from typing import Callable, Union

from .core.interfaces import MemorySystem
from .modules import (
    TripleStore, SqliteTripleStore, KuzuTripleStore,
    TemplateExtractor, LLMExtractor,
    EmbeddingRetriever, TimeAwareRetriever,
    NoOpMaintainer, KeyOverwriteMaintainer, BiTemporalMaintainer,
)
from .providers import Provider, get_provider

# default audit-trail bound for the "bounded" variant (forces forgetting in demos)
DEFAULT_CAPACITY = 6


def _extractor(provider: Provider):
    if provider.name == "mlx":
        return LLMExtractor(provider.llm)
    return TemplateExtractor()


def _build(name, provider, storage, maintainer, retriever) -> MemorySystem:
    return MemorySystem(
        name=name,
        extractor=_extractor(provider),
        storage=storage,
        maintainer=maintainer,
        retriever=retriever,
        embedder=provider.embedder,
        llm=provider.llm,
    )


_FACTORIES: dict[str, Callable[[str, Provider], MemorySystem]] = {
    # append-only RAG (recency-agnostic): the canonical stale baseline
    "append-only": lambda p_name, p: _build(
        p_name, p, TripleStore(), NoOpMaintainer(),
        EmbeddingRetriever(collapse_recent=False)),
    # recency-aware append-only: fixes simple overwrites, still fails retraction
    "append-recency": lambda p_name, p: _build(
        p_name, p, TripleStore(), NoOpMaintainer(),
        EmbeddingRetriever(collapse_recent=True)),
    # Mem0-style key overwrite: current + tiny, but loses history
    "key-overwrite": lambda p_name, p: _build(
        p_name, p, TripleStore(), KeyOverwriteMaintainer(),
        EmbeddingRetriever(collapse_recent=True)),
    # MiniMemState: GEM revision + forgetting on a bi-temporal graph
    "bitemporal": lambda p_name, p: _build(
        p_name, p, TripleStore(), BiTemporalMaintainer(),
        TimeAwareRetriever()),
    # bounded audit trail (forgetting policy on)
    "bitemporal-bounded": lambda p_name, p: _build(
        p_name, p, TripleStore(), BiTemporalMaintainer(capacity=DEFAULT_CAPACITY),
        TimeAwareRetriever()),
    # ablation: time-aware reader but maintenance OFF -> staleness returns
    "bitemporal-nomaint": lambda p_name, p: _build(
        p_name, p, TripleStore(), NoOpMaintainer(),
        TimeAwareRetriever()),
    # same bi-temporal semantics, persisted in SQLite (stdlib, on-disk capable)
    "bitemporal-sqlite": lambda p_name, p: _build(
        p_name, p, SqliteTripleStore(), BiTemporalMaintainer(),
        TimeAwareRetriever()),
    # same again, on the Kùzu embedded property-graph engine (needs [kuzu] extra)
    "bitemporal-kuzu": lambda p_name, p: _build(
        p_name, p, KuzuTripleStore(), BiTemporalMaintainer(),
        TimeAwareRetriever()),
}

# the default comparison set printed by `bench`
BACKENDS = ["append-only", "append-recency", "key-overwrite", "bitemporal"]
ALL_BACKENDS = list(_FACTORIES.keys())


def build_backend(name: str, provider: Union[Provider, str, None] = None) -> MemorySystem:
    if name not in _FACTORIES:
        raise ValueError(f"unknown backend {name!r}. choices: {ALL_BACKENDS}")
    if provider is None or isinstance(provider, str):
        provider = get_provider(provider or "mock")
    return _FACTORIES[name](name, provider)

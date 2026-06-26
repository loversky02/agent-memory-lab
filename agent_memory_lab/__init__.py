"""Agent Memory Lab.

A runnable lab for the paper *Are We Ready For An Agent-Native Memory System?*
(arXiv:2606.24775). It treats agent memory as a DATA SYSTEM with four swappable
modules — storage, extraction, retrieval, maintenance — and measures the
unsolved primitive the paper highlights: **invalidation** (retracting a fact
that was true last week).

Four facets, one repo:
  * Ablation playground  -> `MemorySystem` composed from 4 modules
  * MiniMemState         -> the `BiTemporalGraph` backend (GEM revision+forgetting)
  * StaleBench           -> temporal-fact generator + invalidation metrics
  * Cost/correctness      -> `Profiler` + dashboard
"""

from .core.interfaces import MemoryItem, Triple, MemorySystem
from .core.profiler import Profiler
from .backends import build_backend, BACKENDS

__version__ = "0.1.0"

__all__ = [
    "MemoryItem",
    "Triple",
    "MemorySystem",
    "Profiler",
    "build_backend",
    "BACKENDS",
    "__version__",
]

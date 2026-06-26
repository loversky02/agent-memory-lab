"""Adapters that map the paper's real benchmarks into StaleBench `Episode`s.

LoCoMo and LongMemEval are natural-language conversation benchmarks, so the
*extraction* and *answering* steps need a real LLM — use `--provider mlx`. The
adapters (and their parsing tests) run offline against bundled sample fixtures;
point `--path` at a full download to run the real thing.
"""

from .base import load_dataset, nl_probe, LOADERS
from . import locomo, longmemeval   # noqa: F401  (registers the loaders)
from .run import run_dataset

__all__ = ["load_dataset", "run_dataset", "nl_probe", "LOADERS"]

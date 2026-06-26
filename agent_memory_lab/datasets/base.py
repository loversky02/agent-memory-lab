"""Dataset loader registry + helpers shared by the adapters."""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

from ..stalebench.generator import Turn, Probe, Episode

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

LOADERS: dict[str, Callable[[list], list[Episode]]] = {}


def register(name: str):
    def deco(fn):
        LOADERS[name] = fn
        return fn
    return deco


def nl_probe(question: str, answer, subject: str = "user",
             meta: Optional[dict] = None) -> Probe:
    return Probe(
        subject=subject, predicate="", query=question, kind="nl",
        gold=(str(answer) if answer is not None else None), meta=meta or {},
    )


def make_episode(subject: str, texts: list[str], probes: list[Probe],
                 meta: dict) -> Episode:
    turns = [Turn(idx=i, text=t, subject=subject, kind="dialog")
             for i, t in enumerate(texts)]
    return Episode(subject=subject, turns=turns, probes=probes,
                   ask_turn=len(turns), meta={**meta, "nl": True})


def _read(name: str, path: Optional[str]):
    if path is None:
        path = os.path.join(FIXTURES, f"{name}_sample.json")
    with open(path) as fh:
        return json.load(fh)


def load_dataset(name: str, path: Optional[str] = None,
                 limit: Optional[int] = None) -> list[Episode]:
    from . import locomo, longmemeval  # noqa: F401  (ensure registration)
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}. choices: {sorted(LOADERS)}")
    episodes = LOADERS[name](_read(name, path))
    return episodes[:limit] if limit is not None else episodes

"""Run an NL dataset across backends and score answer accuracy.

Each probe's predicted answer comes from `MemorySystem.answer_nl` (LLM reads the
retrieved memories). Scoring is normalized substring containment of the gold
answer — a lenient recall metric standard for open-ended memory QA. Meaningful
only with a real LLM (`--provider mlx`); the mock provider returns empty answers.
"""

from __future__ import annotations

import sys
from typing import Optional, Union

from ..backends import build_backend
from ..providers import Provider, get_provider
from .base import load_dataset


def _norm(s: Optional[str]) -> str:
    return " ".join(s.lower().split()) if s else ""


def score(gold: Optional[str], predicted: Optional[str]) -> bool:
    g, p = _norm(gold), _norm(predicted)
    return bool(g) and bool(p) and g in p


def run_dataset(
    name: str,
    path: Optional[str] = None,
    provider: Union[Provider, str, None] = "mock",
    backends: Optional[list[str]] = None,
    limit: Optional[int] = None,
    max_probes: Optional[int] = None,
) -> list[dict]:
    provider = provider if isinstance(provider, Provider) else get_provider(provider or "mock")
    episodes = load_dataset(name, path, limit)
    backends = backends or ["bitemporal", "append-only"]
    rows = []
    for bname in backends:
        try:
            correct = total = 0
            for ep in episodes:
                system = build_backend(bname, provider)
                system.ingest(ep.turns)
                probes = ep.probes[:max_probes] if max_probes else ep.probes
                for pr in probes:
                    pred = system.answer_nl(pr.query, ep.ask_turn)
                    total += 1
                    correct += int(score(pr.gold, pred))
            rows.append({
                "backend": bname, "dataset": name, "episodes": len(episodes),
                "probes": total,
                "answer_accuracy": round(correct / total, 4) if total else 0.0,
            })
        except Exception as exc:
            print(f"[skip] backend {bname!r}: {exc}", file=sys.stderr)
    return rows

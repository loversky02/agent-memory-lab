"""Run StaleBench across backends and collect the cost+correctness table."""

from __future__ import annotations

from typing import Optional, Union

from ..backends import build_backend
from ..providers import Provider, get_provider
from ..modules.extractors import QUERIES, render_set
from .generator import Turn, Episode, generate_episode
from .metrics import evaluate_probe, summarize


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _provider(p: Union[Provider, str, None]) -> Provider:
    return p if isinstance(p, Provider) else get_provider(p or "mock")


def run_episode(system, episode: Episode) -> dict:
    system.ingest(episode.turns)
    records = [evaluate_probe(system, pr, episode.ask_turn) for pr in episode.probes]
    return {
        "records": records,
        "prof": system.prof.snapshot(),
        "size": system.storage.size(),
    }


def measure_invalidation_latency(name: str, provider: Union[Provider, str, None] = None) -> Optional[int]:
    """Turns after an overwrite until the system stops returning the old value.

    editor: Vim (turn 0) -> VS Code (turn 3). We probe at now = 3,4,5,...
    Returns 0 if the update is immediate, None if the old value never clears.
    """
    provider = _provider(provider)
    overwrite_turn = 3
    turns = [
        Turn(0, render_set("editor", "Vim"), kind="set", predicate="editor", value="Vim"),
        Turn(1, "Filler chit-chat about lunch."),
        Turn(2, "Filler chit-chat about the weather."),
        Turn(3, render_set("editor", "VS Code"), kind="set", predicate="editor", value="VS Code"),
        Turn(4, "Filler chit-chat about coffee."),
        Turn(5, "Filler chit-chat about traffic."),
    ]
    system = build_backend(name, provider)
    system.ingest(turns)
    for now in range(overwrite_turn, len(turns) + 1):
        if system.answer(QUERIES["editor"], "editor", now) != "Vim":
            return now - overwrite_turn
    return None


def _bench_one(name, provider, n_episodes, seed0, scenario, chain_len, filler_ratio) -> dict:
    recs: list[dict] = []
    sizes, scans, writes, removes, closes, forgets = [], [], [], [], [], []
    for e in range(n_episodes):
        ep = generate_episode(seed0 + e, scenario=scenario,
                              chain_len=chain_len, filler_ratio=filler_ratio)
        system = build_backend(name, provider)
        res = run_episode(system, ep)
        recs.extend(res["records"])
        sizes.append(res["size"])
        p = res["prof"]
        scans.append(p["avg_scanned"]); writes.append(p["writes"])
        removes.append(p["removes"]); closes.append(p["closes"])
        forgets.append(p["forgets"])
    return {"backend": name, **summarize(recs),
            "invalidation_latency": measure_invalidation_latency(name, provider),
            "avg_store_size": round(_mean(sizes), 2),
            "avg_scanned": round(_mean(scans), 2),
            "writes": round(_mean(writes), 2),
            "removes": round(_mean(removes), 2),
            "closes": round(_mean(closes), 2),
            "forgets": round(_mean(forgets), 2)}


def run_benchmark(
    backends: list[str],
    provider: Union[Provider, str, None] = None,
    n_episodes: int = 20,
    seed0: int = 0,
    scenario: str = "all",
    chain_len: int = 3,
    filler_ratio: float = 0.5,
) -> list[dict]:
    provider = _provider(provider)
    rows: list[dict] = []
    for name in backends:
        try:
            rows.append(_bench_one(name, provider, n_episodes, seed0,
                                   scenario, chain_len, filler_ratio))
        except Exception as exc:  # e.g. optional backend (kuzu) not installed
            import sys
            print(f"[skip] backend {name!r}: {exc}", file=sys.stderr)
    return rows

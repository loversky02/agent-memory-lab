"""StaleBench — a synthetic temporal-fact workload.

We script a stream of conversation turns in which user attributes are *updated*
(SET v1 -> v2 -> v3) and sometimes *retracted* (the fact stops being true with
no replacement). Filler turns add noise. At the end we probe each attribute and
record three things per probe:

  * gold       — the value that is true NOW (None if retracted)
  * superseded — values that USED to be true (returning any of these == stale)
  * target     — for retractions, the value that must no longer be returned

This is the dynamic-knowledge-update setting the paper says end-to-end F1
quietly fails. Everything is seeded => fully reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..modules.extractors import PREDICATES, QUERIES, render_set, render_retract

FILLERS = [
    "The weather has been lovely this whole week.",
    "I grabbed a coffee this morning before work.",
    "We watched a long movie last night.",
    "My plants needed watering again today.",
    "Traffic was surprisingly light on the way back.",
    "I finally cleaned out my inbox.",
    "Lunch was great, thanks for asking.",
]


@dataclass
class Turn:
    idx: int
    text: str
    subject: str = "user"
    kind: str = "filler"            # set | retract | filler
    predicate: Optional[str] = None
    value: Optional[str] = None


@dataclass
class Probe:
    subject: str
    predicate: str
    query: str
    kind: str                       # current | history
    gold: Optional[str]
    superseded: list[str] = field(default_factory=list)
    target: Optional[str] = None    # retracted value that must not resurface
    meta: dict = field(default_factory=dict)   # dataset extras (category, id, ...)


@dataclass
class Episode:
    subject: str
    turns: list[Turn]
    probes: list[Probe]
    ask_turn: int
    meta: dict = field(default_factory=dict)


def _scenario_retracts(rng: random.Random, predicates: list[str], scenario: str) -> set[str]:
    if scenario == "basic":
        return set()
    if scenario == "retract":
        # retract roughly half (at least one)
        k = max(1, len(predicates) // 2)
        return set(rng.sample(predicates, k))
    # "all": each predicate retracted with prob 0.5
    return {p for p in predicates if rng.random() < 0.5}


def generate_episode(
    seed: int,
    predicates: Optional[list[str]] = None,
    chain_len: int = 3,
    scenario: str = "all",
    filler_ratio: float = 0.5,
    subject: str = "user",
) -> Episode:
    rng = random.Random(seed)
    preds = list(predicates) if predicates else list(PREDICATES.keys())
    retracted = _scenario_retracts(rng, preds, scenario)

    # plan each predicate's ordered events
    decks: dict[str, list[tuple[str, str]]] = {}
    for p in preds:
        n = min(chain_len, len(PREDICATES[p]))
        values = rng.sample(PREDICATES[p], n)
        events = [("set", v) for v in values]
        if p in retracted:
            events.append(("retract", values[-1]))
        decks[p] = events

    # interleave events (preserve per-predicate order), sprinkle filler turns
    turns: list[Turn] = []
    event_turns: dict[str, list[tuple[int, str, str]]] = {p: [] for p in preds}
    idx = 0
    while any(decks[p] for p in preds):
        live = [p for p in preds if decks[p]]
        p = rng.choice(live)
        if rng.random() < filler_ratio:
            turns.append(Turn(idx=idx, text=rng.choice(FILLERS), subject=subject))
            idx += 1
        kind, value = decks[p].pop(0)
        text = render_set(p, value) if kind == "set" else render_retract(p, value)
        turns.append(Turn(idx=idx, text=text, subject=subject, kind=kind,
                          predicate=p, value=value))
        event_turns[p].append((idx, kind, value))
        idx += 1

    ask_turn = idx
    probes = _build_probes(preds, event_turns, subject)
    return Episode(subject=subject, turns=turns, probes=probes, ask_turn=ask_turn,
                   meta={"seed": seed, "scenario": scenario,
                         "retracted": sorted(retracted)})


def _build_probes(preds, event_turns, subject) -> list[Probe]:
    probes: list[Probe] = []
    for p in preds:
        evs = event_turns[p]
        sets = [v for (_, k, v) in evs if k == "set"]
        retracted = evs[-1][1] == "retract"
        current = None if retracted else sets[-1]
        # everything that was ever true but isn't the current value
        superseded = [v for v in sets if v != current]
        target = sets[-1] if retracted else None
        probes.append(Probe(
            subject=subject, predicate=p, query=QUERIES[p], kind="current",
            gold=current, superseded=superseded, target=target,
        ))
        # history probe only makes sense when there is a stable current value
        if not retracted and len(sets) >= 2:
            probes.append(Probe(
                subject=subject, predicate=p,
                query=f"Which {p} did I use before {current}?",
                kind="history", gold=sets[-2],
            ))
    return probes


def controlled_episode() -> Episode:
    """A tiny, fully-deterministic episode for tests/demos (no RNG).

    editor:    Vim -> VS Code            (update; current = VS Code)
    language:  Python -> Rust, retracted (current = None; Rust must not resurface)
    """
    turns = [
        Turn(0, render_set("editor", "Vim"), kind="set", predicate="editor", value="Vim"),
        Turn(1, FILLERS[0]),
        Turn(2, render_set("language", "Python"), kind="set", predicate="language", value="Python"),
        Turn(3, render_set("editor", "VS Code"), kind="set", predicate="editor", value="VS Code"),
        Turn(4, FILLERS[1]),
        Turn(5, render_set("language", "Rust"), kind="set", predicate="language", value="Rust"),
        Turn(6, render_retract("language", "Rust"), kind="retract", predicate="language", value="Rust"),
    ]
    probes = [
        Probe("user", "editor", QUERIES["editor"], "current", "VS Code", ["Vim"]),
        Probe("user", "editor", "Which editor did I use before VS Code?", "history", "Vim"),
        Probe("user", "language", QUERIES["language"], "current", None, ["Python", "Rust"], target="Rust"),
    ]
    return Episode("user", turns, probes, ask_turn=7, meta={"controlled": True})

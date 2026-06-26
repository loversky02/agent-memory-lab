"""Metrics that look BELOW the final answer (the paper's whole point).

  staleness_rate   fraction of current-probes answered with a superseded /
                   retracted value — the "hallucination of the past" rate.
  update_em        substring-exact-match of the current answer vs gold (the
                   number the paper reports, e.g. Zep ~44.4).
  history_recall   can the system still answer "what did I use before?" —
                   tests whether maintenance preserved a temporal trail.
  invalidation_latency (see runner) turns until an overwrite takes effect.
"""

from __future__ import annotations

from .generator import Probe


def evaluate_probe(system, probe: Probe, now_t: int) -> dict:
    if probe.kind == "history":
        predicted = system.recall_previous(probe.predicate, now_t, probe.subject)
        stale = False
    else:
        predicted = system.answer(probe.query, probe.predicate, now_t)
        stale = predicted is not None and predicted in set(probe.superseded)
    return {
        "predicate": probe.predicate,
        "kind": probe.kind,
        "predicted": predicted,
        "gold": probe.gold,
        "stale": bool(stale),
        "correct": predicted == probe.gold,
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate per-probe records into headline rates."""
    cur = [r for r in records if r["kind"] == "current"]
    hist = [r for r in records if r["kind"] == "history"]

    def _rate(rows, key):
        return sum(1 for r in rows if r[key]) / len(rows) if rows else 0.0

    return {
        "n_current": len(cur),
        "n_history": len(hist),
        "staleness_rate": round(_rate(cur, "stale"), 4),
        "update_em": round(_rate(cur, "correct"), 4),
        "history_recall": round(_rate(hist, "correct"), 4),
    }

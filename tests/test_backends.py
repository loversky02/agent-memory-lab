"""Behavioral contract for each backend on the controlled episode.

These assertions encode the paper's qualitative findings:
  * append-only returns stale facts and can't answer history
  * key-overwrite is current + tiny but amnesiac
  * the bi-temporal graph is current AND keeps an auditable trail
  * removing maintenance (but keeping the time-aware reader) brings staleness back
"""

from agent_memory_lab.backends import build_backend
from agent_memory_lab.stalebench import controlled_episode
from agent_memory_lab.stalebench.metrics import summarize
from agent_memory_lab.stalebench.runner import run_episode


def _run(name):
    system = build_backend(name, "mock")
    res = run_episode(system, controlled_episode())
    return summarize(res["records"]), system


def test_append_only_is_stale_and_amnesiac():
    s, _ = _run("append-only")
    assert s["staleness_rate"] > 0        # retraction guarantees at least one stale
    assert s["history_recall"] == 0.0     # no temporal trail to query


def test_key_overwrite_current_but_no_history():
    s, system = _run("key-overwrite")
    assert s["staleness_rate"] == 0.0
    assert s["update_em"] == 1.0
    assert s["history_recall"] == 0.0
    assert system.storage.size() <= 2     # keeps only current values


def test_bitemporal_correct_and_keeps_history():
    s, system = _run("bitemporal")
    assert s["staleness_rate"] == 0.0
    assert s["update_em"] == 1.0
    assert s["history_recall"] == 1.0
    assert system.prof.closes >= 2        # revision (edge invalidation) happened


def test_ablation_no_maintenance_brings_staleness_back():
    s, _ = _run("bitemporal-nomaint")
    assert s["staleness_rate"] > 0        # time-aware reader, but nothing invalidated

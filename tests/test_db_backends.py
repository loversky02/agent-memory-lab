"""The bi-temporal contract must hold identically across storage engines."""

import pytest

from agent_memory_lab.backends import build_backend
from agent_memory_lab.core.interfaces import MemoryItem
from agent_memory_lab.stalebench import controlled_episode
from agent_memory_lab.stalebench.metrics import summarize
from agent_memory_lab.stalebench.runner import run_episode


@pytest.mark.parametrize("name", ["bitemporal", "bitemporal-sqlite", "bitemporal-kuzu"])
def test_db_backends_match_inmemory(name):
    try:
        system = build_backend(name, "mock")
    except RuntimeError as exc:           # optional engine (kuzu) not installed
        pytest.skip(f"{name} unavailable: {exc}")
    s = summarize(run_episode(system, controlled_episode())["records"])
    assert s["staleness_rate"] == 0.0
    assert s["update_em"] == 1.0
    assert s["history_recall"] == 1.0
    assert system.prof.closes >= 2        # revision happened in the DB too


def test_sqlite_persists_to_file(tmp_path):
    from agent_memory_lab.modules.sqlite_storage import SqliteTripleStore
    path = str(tmp_path / "mem.db")
    store = SqliteTripleStore(path)
    store.add(MemoryItem(0, "user", "editor", "Vim", "set Vim", 0, 0))
    store.close(0, 3)
    del store
    reopened = SqliteTripleStore(path)    # survives across "process" boundary
    assert reopened.size() == 1
    item = reopened.items()[0]
    assert item.object == "Vim" and item.t_valid_end == 3

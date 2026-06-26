from agent_memory_lab.stalebench.metrics import summarize


def test_summarize_rates():
    recs = [
        {"kind": "current", "stale": True, "correct": False},
        {"kind": "current", "stale": False, "correct": True},
        {"kind": "history", "stale": False, "correct": True},
    ]
    s = summarize(recs)
    assert s["n_current"] == 2 and s["n_history"] == 1
    assert s["staleness_rate"] == 0.5
    assert s["update_em"] == 0.5
    assert s["history_recall"] == 1.0


def test_summarize_empty():
    s = summarize([])
    assert s["staleness_rate"] == 0.0 and s["history_recall"] == 0.0

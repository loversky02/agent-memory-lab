import pytest

from agent_memory_lab.datasets import load_dataset, run_dataset
from agent_memory_lab.datasets.run import score


def test_locomo_parses():
    eps = load_dataset("locomo")
    assert len(eps) == 1
    ep = eps[0]
    assert ep.meta["nl"] is True
    assert len(ep.turns) == 4
    gold = {p.query: p.gold for p in ep.probes}
    assert gold["What editor does Alice use now?"] == "VS Code"
    assert gold["What editor did Alice originally use?"] == "Vim"


def test_longmemeval_parses():
    eps = load_dataset("longmemeval")
    assert len(eps) == 1
    ep = eps[0]
    assert len(ep.turns) == 4
    pr = ep.probes[0]
    assert pr.gold == "Berlin"
    assert pr.meta["type"] == "knowledge-update"


def test_limit_zero():
    assert load_dataset("longmemeval", limit=0) == []


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("nope")


def test_score_contains():
    assert score("Berlin", "I believe the user lives in Berlin now.")
    assert not score("Berlin", "They live in Hanoi.")
    assert not score("Berlin", None)


def test_run_dataset_smoke_mock():
    rows = run_dataset("locomo", provider="mock", backends=["bitemporal"])
    assert rows and rows[0]["backend"] == "bitemporal"
    assert "answer_accuracy" in rows[0]

from agent_memory_lab.stalebench import generate_episode, controlled_episode


def _sig(ep):
    return ([t.text for t in ep.turns],
            [(p.predicate, p.kind, p.gold, tuple(p.superseded)) for p in ep.probes])


def test_determinism_same_seed():
    assert _sig(generate_episode(1)) == _sig(generate_episode(1))


def test_different_seed_differs():
    assert [t.text for t in generate_episode(1).turns] != \
           [t.text for t in generate_episode(2).turns]


def test_superseded_excludes_current():
    ep = generate_episode(3, scenario="basic")
    for p in ep.probes:
        if p.kind == "current" and p.gold is not None:
            assert p.gold not in p.superseded


def test_controlled_structure():
    cur = {p.predicate: p for p in controlled_episode().probes if p.kind == "current"}
    assert cur["editor"].gold == "VS Code" and "Vim" in cur["editor"].superseded
    assert cur["language"].gold is None and cur["language"].target == "Rust"

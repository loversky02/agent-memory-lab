from agent_memory_lab.modules.extractors import (
    TemplateExtractor, render_set, render_retract,
)

ex = TemplateExtractor()


def test_extract_set():
    t = ex.extract(render_set("editor", "Vim"))
    assert len(t) == 1
    assert t[0].predicate == "editor" and t[0].object == "Vim"
    assert not t[0].is_retraction


def test_extract_retraction():
    t = ex.extract(render_retract("language", "Rust"))
    assert len(t) == 1
    assert t[0].predicate == "language" and t[0].is_retraction
    assert t[0].object is None


def test_filler_yields_nothing():
    assert ex.extract("The weather has been lovely this week.") == []


def test_multiword_value():
    assert ex.extract(render_set("editor", "VS Code"))[0].object == "VS Code"
    assert ex.extract(render_set("role", "tech lead"))[0].object == "tech lead"

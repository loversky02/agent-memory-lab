from agent_memory_lab.core.similarity import cosine
from agent_memory_lab.providers import get_provider


def test_mock_embedder_deterministic():
    e = get_provider("mock").embedder
    assert e.embed("hello world") == e.embed("hello world")
    assert abs(cosine(e.embed("hello world"), e.embed("hello world")) - 1.0) < 1e-9


def test_similar_beats_dissimilar():
    e = get_provider("mock").embedder
    base = e.embed("I switched my main editor to Vim.")
    near = e.embed("I switched my main editor to Zed.")
    far = e.embed("The weather has been lovely this week.")
    assert cosine(base, near) > cosine(base, far)

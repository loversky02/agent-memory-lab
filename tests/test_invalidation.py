from agent_memory_lab.stalebench.runner import measure_invalidation_latency


def test_append_only_never_invalidates():
    assert measure_invalidation_latency("append-only", "mock") is None


def test_recency_and_lifecycle_invalidate_immediately():
    assert measure_invalidation_latency("append-recency", "mock") == 0
    assert measure_invalidation_latency("key-overwrite", "mock") == 0
    assert measure_invalidation_latency("bitemporal", "mock") == 0

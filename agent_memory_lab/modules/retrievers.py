"""Retrieval module: query + now -> ranked items.

`EmbeddingRetriever` is classic read-only RAG. Two flavors:
  * collapse_recent=False — return everything ever stored, ranked by similarity.
    With nothing to disambiguate versions, a stale fact can win ("hallucination
    of the past").
  * collapse_recent=True — a recency heuristic: keep only the newest fact per
    (subject, predicate) at read time. Fixes simple overwrites, but a *retracted*
    fact still has a live positive version, so retraction staleness remains.

`TimeAwareRetriever` filters to edges valid at `now_t` first — the only thing
that handles both overwrite AND retraction, because invalidation is recorded in
valid-time by the maintainer.
"""

from __future__ import annotations

from ..core.interfaces import MemoryItem, Retriever, Storage
from ..core.similarity import cosine


def _rank(query: str, cands: list[MemoryItem], k: int, embedder,
          collapse_recent: bool) -> list[MemoryItem]:
    if collapse_recent:
        latest: dict[tuple, MemoryItem] = {}
        for c in cands:
            key = (c.subject, c.predicate)
            cur = latest.get(key)
            if cur is None or c.t_ingested > cur.t_ingested:
                latest[key] = c
        cands = list(latest.values())
    qv = embedder.embed(query)
    scored = [(cosine(qv, embedder.embed(c.text)), c) for c in cands]
    # recency as the tie-break when similarities are equal
    scored.sort(key=lambda s: (-s[0], -s[1].t_ingested))
    return [c for _, c in scored[:k]]


class EmbeddingRetriever(Retriever):
    def __init__(self, collapse_recent: bool = False) -> None:
        self.collapse_recent = collapse_recent

    def retrieve(self, query, k, now_t, store: Storage, embedder, prof):
        cands = [i for i in store.items() if i.object is not None and not i.is_retraction]
        prof.retrievals += 1
        prof.items_scanned += len(cands)
        return _rank(query, cands, k, embedder, self.collapse_recent)


class TimeAwareRetriever(Retriever):
    """Top-k restricted to edges valid at `now_t` (bi-temporal read)."""

    def retrieve(self, query, k, now_t, store: Storage, embedder, prof):
        cands = [i for i in store.query(valid_at=now_t) if i.object is not None]
        prof.retrievals += 1
        prof.items_scanned += len(cands)
        return _rank(query, cands, k, embedder, collapse_recent=False)

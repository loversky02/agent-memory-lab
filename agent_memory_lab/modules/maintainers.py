"""Maintenance module — the lifecycle step the paper says decides correctness.

Three strategies, increasing in capability:

  NoOpMaintainer        append-only. Stores everything, never invalidates.
                        => returns stale facts ("hallucinations of the past").
  KeyOverwriteMaintainer  Mem0-style. Overwrite by (subject,predicate) key.
                        => stays current and tiny, but destroys history.
  BiTemporalMaintainer   GEM operators: `revision` (close the contradicted edge
                        by setting valid-time) + `forgetting` (bounded trim of
                        closed edges). => current AND auditable history.

The paper's headline — *localized maintenance beats global reorganization* — is
exactly the move from append-only to local edge revision.
"""

from __future__ import annotations

from typing import Optional

from ..core.interfaces import MemoryItem, Maintainer, Storage
from ..core.profiler import Profiler


class NoOpMaintainer(Maintainer):
    name = "append-only"

    def integrate(self, new_items, store: Storage, now_t: int, prof: Profiler) -> None:
        for it in new_items:
            store.add(it)
            prof.writes += 1


class KeyOverwriteMaintainer(Maintainer):
    name = "key-overwrite"

    def integrate(self, new_items, store: Storage, now_t: int, prof: Profiler) -> None:
        for it in new_items:
            stale = store.query(subject=it.subject, predicate=it.predicate,
                                include_retractions=True)
            for old in stale:
                store.remove(old.id)
                prof.removes += 1
            if not it.is_retraction:        # a retraction simply clears the key
                store.add(it)
                prof.writes += 1


class BiTemporalMaintainer(Maintainer):
    name = "bitemporal-graph"

    def __init__(self, capacity: Optional[int] = None) -> None:
        # capacity bounds the audit trail; None = keep all history.
        self.capacity = capacity

    def integrate(self, new_items, store: Storage, now_t: int, prof: Profiler) -> None:
        for it in new_items:
            open_pos = store.query(subject=it.subject, predicate=it.predicate,
                                   open_only=True)  # open positive edges
            if it.is_retraction:
                # forgetting/retraction: close every open edge for this key.
                for e in open_pos:
                    store.close(e.id, now_t)
                    prof.closes += 1
                store.add(it)               # keep tombstone for provenance
                prof.writes += 1
                continue

            if any(e.object == it.object for e in open_pos):
                continue                    # dedupe: cheap local no-op
            # revision: close the contradicted edge(s), then open the new one.
            for e in open_pos:
                if e.object != it.object:
                    store.close(e.id, now_t)
                    prof.closes += 1
            store.add(it)
            prof.writes += 1

        self._forget(store, prof)

    def _forget(self, store: Storage, prof: Profiler) -> None:
        if self.capacity is None:
            return
        while store.size() > self.capacity:
            closed = [i for i in store.items() if i.t_valid_end is not None]
            if not closed:
                break                       # never drop an open (current) edge
            victim = min(closed, key=lambda i: i.t_valid_end)
            store.remove(victim.id)
            prof.forgets += 1

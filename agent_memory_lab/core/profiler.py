"""Operational-cost profiler.

The paper's core complaint: end-to-end F1 hides cost. So every memory operation
is counted here — writes, removes, edge closes (revision), forgets, and the
number of items scanned per retrieval — plus wall-clock for ingest/retrieve.
These feed the Cost-vs-Correctness view.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Profiler:
    writes: int = 0          # items appended to storage
    removes: int = 0         # items hard-deleted (key overwrite)
    closes: int = 0          # edges invalidated via valid-time (GEM revision)
    forgets: int = 0         # closed edges dropped by forgetting policy
    retrievals: int = 0      # number of retrieve() calls
    items_scanned: int = 0   # cumulative candidates scanned across retrievals
    ingest_seconds: float = 0.0
    retrieve_seconds: float = 0.0

    def reset(self) -> None:
        for f in (
            "writes", "removes", "closes", "forgets",
            "retrievals", "items_scanned",
        ):
            setattr(self, f, 0)
        self.ingest_seconds = 0.0
        self.retrieve_seconds = 0.0

    @property
    def avg_scanned(self) -> float:
        return self.items_scanned / self.retrievals if self.retrievals else 0.0

    def snapshot(self) -> dict:
        d = asdict(self)
        d["avg_scanned"] = round(self.avg_scanned, 3)
        return d

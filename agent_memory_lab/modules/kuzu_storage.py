"""Kùzu-backed triple store — an embedded property-graph engine.

Facts are stored as nodes in Kùzu (an embedded graph database) and queried with
Cypher. Same bi-temporal contract as the other stores, so it slots into the
`bitemporal-kuzu` backend unchanged. Requires `pip install "agent-memory-lab[kuzu]"`.
Lazily imported so the rest of the lab runs without it.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from ..core.interfaces import MemoryItem, Storage

_RETURN = ("f.id, f.subject, f.predicate, f.object, f.text, f.t_ingested, "
           "f.t_valid_start, f.t_valid_end, f.is_retraction, f.source_turn")

_SCHEMA = (
    "CREATE NODE TABLE IF NOT EXISTS Fact("
    "id INT64, subject STRING, predicate STRING, object STRING, text STRING, "
    "t_ingested INT64, t_valid_start INT64, t_valid_end INT64, "
    "is_retraction BOOLEAN, source_turn INT64, PRIMARY KEY(id))"
)


class KuzuTripleStore(Storage):
    def __init__(self, path: Optional[str] = None) -> None:
        try:
            import kuzu  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "kuzu is not installed. Run: pip install 'agent-memory-lab[kuzu]'"
            ) from exc
        # Kùzu wants a path it can create itself (not an existing dir).
        self.path = path or os.path.join(tempfile.mkdtemp(prefix="aml_kuzu_"), "graph")
        self._db = kuzu.Database(self.path)
        self._conn = kuzu.Connection(self._db)
        self._conn.execute(_SCHEMA)

    @staticmethod
    def _row_to_item(row) -> MemoryItem:
        return MemoryItem(
            id=row[0], subject=row[1], predicate=row[2], object=row[3],
            text=row[4], t_ingested=row[5], t_valid_start=row[6],
            t_valid_end=row[7], is_retraction=bool(row[8]), source_turn=row[9],
        )

    def _rows(self, cypher: str, params: Optional[dict] = None) -> list:
        res = self._conn.execute(cypher, parameters=params or {})
        out = []
        while res.has_next():
            out.append(res.get_next())
        return out

    def add(self, item: MemoryItem) -> None:
        props = {
            "id": item.id, "subject": item.subject, "predicate": item.predicate,
            "text": item.text, "t_ingested": item.t_ingested,
            "t_valid_start": item.t_valid_start,
            "is_retraction": bool(item.is_retraction),
            "source_turn": item.source_turn,
        }
        if item.object is not None:
            props["object"] = item.object
        if item.t_valid_end is not None:
            props["t_valid_end"] = item.t_valid_end
        assignment = ", ".join(f"{k}: ${k}" for k in props)
        self._conn.execute(f"CREATE (f:Fact {{{assignment}}})", parameters=props)

    def remove(self, item_id: int) -> None:
        self._conn.execute("MATCH (f:Fact) WHERE f.id = $id DELETE f",
                          parameters={"id": item_id})

    def close(self, item_id: int, t_valid_end: int) -> None:
        self._conn.execute(
            "MATCH (f:Fact) WHERE f.id = $id SET f.t_valid_end = $te",
            parameters={"id": item_id, "te": t_valid_end})

    def items(self) -> list[MemoryItem]:
        rows = self._rows(f"MATCH (f:Fact) RETURN {_RETURN} ORDER BY f.id")
        return [self._row_to_item(r) for r in rows]

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        open_only: bool = False,
        valid_at: Optional[int] = None,
        include_retractions: bool = False,
    ) -> list[MemoryItem]:
        clauses, params = [], {}
        if subject is not None:
            clauses.append("f.subject = $subject"); params["subject"] = subject
        if predicate is not None:
            clauses.append("f.predicate = $predicate"); params["predicate"] = predicate
        if open_only:
            clauses.append("f.t_valid_end IS NULL")
        if valid_at is not None:
            clauses.append("f.t_valid_start <= $va AND "
                           "(f.t_valid_end IS NULL OR f.t_valid_end > $va)")
            params["va"] = valid_at
        if not include_retractions:
            clauses.append("f.is_retraction = false")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._rows(f"MATCH (f:Fact){where} RETURN {_RETURN} ORDER BY f.id", params)
        return [self._row_to_item(r) for r in rows]

    def size(self) -> int:
        return self._rows("MATCH (f:Fact) RETURN count(f)")[0][0]

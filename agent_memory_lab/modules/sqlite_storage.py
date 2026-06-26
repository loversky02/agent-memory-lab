"""SQLite-backed triple store (stdlib, persistent).

Same bi-temporal semantics as the in-memory `TripleStore`, but the state lives
in a real database — so "agent memory is a data system" is literal here, and a
file-backed store survives across runs. Zero install (sqlite3 ships with Python).
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from ..core.interfaces import MemoryItem, Storage

_COLS = ("id", "subject", "predicate", "object", "text", "t_ingested",
         "t_valid_start", "t_valid_end", "is_retraction", "source_turn")

_DDL = """
CREATE TABLE IF NOT EXISTS edges (
    id            INTEGER PRIMARY KEY,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT,
    text          TEXT NOT NULL,
    t_ingested    INTEGER NOT NULL,
    t_valid_start INTEGER NOT NULL,
    t_valid_end   INTEGER,
    is_retraction INTEGER NOT NULL DEFAULT 0,
    source_turn   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sp ON edges(subject, predicate);
"""


class SqliteTripleStore(Storage):
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._db = sqlite3.connect(path)
        self._db.executescript(_DDL)
        self._db.commit()

    def _row_to_item(self, row) -> MemoryItem:
        d = dict(zip(_COLS, row))
        return MemoryItem(
            id=d["id"], subject=d["subject"], predicate=d["predicate"],
            object=d["object"], text=d["text"], t_ingested=d["t_ingested"],
            t_valid_start=d["t_valid_start"], t_valid_end=d["t_valid_end"],
            is_retraction=bool(d["is_retraction"]), source_turn=d["source_turn"],
        )

    def add(self, item: MemoryItem) -> None:
        self._db.execute(
            f"INSERT OR REPLACE INTO edges ({','.join(_COLS)}) "
            f"VALUES ({','.join('?' * len(_COLS))})",
            (item.id, item.subject, item.predicate, item.object, item.text,
             item.t_ingested, item.t_valid_start, item.t_valid_end,
             int(item.is_retraction), item.source_turn),
        )
        self._db.commit()

    def remove(self, item_id: int) -> None:
        self._db.execute("DELETE FROM edges WHERE id = ?", (item_id,))
        self._db.commit()

    def close(self, item_id: int, t_valid_end: int) -> None:
        self._db.execute("UPDATE edges SET t_valid_end = ? WHERE id = ?",
                         (t_valid_end, item_id))
        self._db.commit()

    def items(self) -> list[MemoryItem]:
        cur = self._db.execute(f"SELECT {','.join(_COLS)} FROM edges ORDER BY id")
        return [self._row_to_item(r) for r in cur.fetchall()]

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        open_only: bool = False,
        valid_at: Optional[int] = None,
        include_retractions: bool = False,
    ) -> list[MemoryItem]:
        clauses, params = [], []
        if subject is not None:
            clauses.append("subject = ?"); params.append(subject)
        if predicate is not None:
            clauses.append("predicate = ?"); params.append(predicate)
        if open_only:
            clauses.append("t_valid_end IS NULL")
        if valid_at is not None:
            clauses.append("t_valid_start <= ? AND (t_valid_end IS NULL OR t_valid_end > ?)")
            params.extend([valid_at, valid_at])
        if not include_retractions:
            clauses.append("is_retraction = 0")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self._db.execute(
            f"SELECT {','.join(_COLS)} FROM edges{where} ORDER BY id", params)
        return [self._row_to_item(r) for r in cur.fetchall()]

    def size(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

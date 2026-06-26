"""In-memory triple store (a tiny bi-temporal property graph).

Triples are edges: subject -[predicate]-> object. Each edge carries valid-time
(`t_valid_start`/`t_valid_end`) and transaction-time (`t_ingested`). This is the
same shape Zep/Graphiti and GEM's MemState use; we just keep it in a list so the
lab has zero external dependencies.
"""

from __future__ import annotations

from typing import Optional

from ..core.interfaces import MemoryItem, Storage


class TripleStore(Storage):
    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self._items.append(item)

    def remove(self, item_id: int) -> None:
        self._items = [i for i in self._items if i.id != item_id]

    def close(self, item_id: int, t_valid_end: int) -> None:
        for i in self._items:
            if i.id == item_id:
                i.t_valid_end = t_valid_end
                return

    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        open_only: bool = False,
        valid_at: Optional[int] = None,
        include_retractions: bool = False,
    ) -> list[MemoryItem]:
        out = []
        for i in self._items:
            if subject is not None and i.subject != subject:
                continue
            if predicate is not None and i.predicate != predicate:
                continue
            if open_only and i.t_valid_end is not None:
                continue
            if valid_at is not None and not i.valid_at(valid_at):
                continue
            if not include_retractions and i.is_retraction:
                continue
            out.append(i)
        return out

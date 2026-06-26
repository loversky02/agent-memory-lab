"""Core data model + the 4-module composition (`MemorySystem`).

A memory system is exactly four pluggable parts (the paper's decomposition):

    extractor   turn text            -> triples
    storage     hold items, answer temporal/triple queries
    maintainer  integrate new triples into storage (the LIFECYCLE step:
                append / overwrite / revise+forget)
    retriever   query + now           -> ranked items

Swap any part to ablate. `MemorySystem` just wires them and profiles cost.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .profiler import Profiler


@dataclass
class MemoryItem:
    """A bi-temporal triple (subject -[predicate]-> object).

    A triple is a property-graph edge; valid-time (`t_valid_start`/`t_valid_end`)
    plus transaction-time (`t_ingested`) make it bi-temporal — the model used by
    Zep/Graphiti and by GEM's MemState.
    """

    id: int
    subject: str
    predicate: str
    object: Optional[str]                 # None => tombstone (a retraction)
    text: str                             # source utterance
    t_ingested: int                       # transaction time (turn it entered memory)
    t_valid_start: int                    # valid time the fact became true
    t_valid_end: Optional[int] = None     # None => still valid (open edge)
    is_retraction: bool = False
    source_turn: int = 0
    meta: dict = field(default_factory=dict)

    def valid_at(self, t: int) -> bool:
        if self.t_valid_start > t:
            return False
        if self.t_valid_end is not None and self.t_valid_end <= t:
            return False
        return True

    @property
    def is_open(self) -> bool:
        return self.t_valid_end is None


@dataclass
class Triple:
    """Proto-item produced by an extractor (no id/timestamps yet)."""

    predicate: str
    object: Optional[str]
    text: str
    is_retraction: bool = False


# --------------------------------------------------------------------------- #
# Module interfaces                                                            #
# --------------------------------------------------------------------------- #
class Extractor(ABC):
    @abstractmethod
    def extract(self, text: str, subject: str = "user") -> list[Triple]:
        ...


class Storage(ABC):
    @abstractmethod
    def add(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def remove(self, item_id: int) -> None: ...

    @abstractmethod
    def close(self, item_id: int, t_valid_end: int) -> None:
        """Set valid-time end on an edge — GEM revision / invalidation.

        A method (not in-place mutation) so DB-backed stores can persist it.
        """
        ...

    @abstractmethod
    def items(self) -> list[MemoryItem]: ...

    @abstractmethod
    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        open_only: bool = False,
        valid_at: Optional[int] = None,
        include_retractions: bool = False,
    ) -> list[MemoryItem]: ...

    def open_edges(self, subject: str, predicate: str) -> list[MemoryItem]:
        return self.query(subject=subject, predicate=predicate,
                          open_only=True, include_retractions=True)

    def closed_edges(self, subject: str, predicate: str) -> list[MemoryItem]:
        return [i for i in self.items()
                if i.subject == subject and i.predicate == predicate
                and i.t_valid_end is not None and i.object is not None]

    def size(self) -> int:
        return len(self.items())


class Maintainer(ABC):
    """The lifecycle step. Decides HOW new triples merge into storage."""

    name: str = "maintainer"

    @abstractmethod
    def integrate(self, new_items: list[MemoryItem], store: Storage,
                  now_t: int, prof: Profiler) -> None:
        ...


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int, now_t: int, store: Storage,
                 embedder, prof: Profiler) -> list[MemoryItem]:
        ...


# --------------------------------------------------------------------------- #
# The wired system                                                            #
# --------------------------------------------------------------------------- #
class MemorySystem:
    def __init__(self, name: str, extractor: Extractor, storage: Storage,
                 maintainer: Maintainer, retriever: Retriever, embedder, llm=None):
        self.name = name
        self.extractor = extractor
        self.storage = storage
        self.maintainer = maintainer
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm                  # only needed for NL datasets (answer_nl)
        self.prof = Profiler()
        self._next_id = 0

    # -- ingestion ---------------------------------------------------------- #
    def ingest_turn(self, text: str, turn_idx: int, subject: str = "user") -> int:
        t0 = time.perf_counter()
        triples = self.extractor.extract(text, subject=subject)
        items: list[MemoryItem] = []
        for tr in triples:
            items.append(MemoryItem(
                id=self._alloc_id(),
                subject=subject,
                predicate=tr.predicate,
                object=tr.object,
                text=tr.text,
                t_ingested=turn_idx,
                t_valid_start=turn_idx,
                is_retraction=tr.is_retraction,
                source_turn=turn_idx,
            ))
        self.maintainer.integrate(items, self.storage, turn_idx, self.prof)
        self.prof.ingest_seconds += time.perf_counter() - t0
        return len(items)

    def ingest(self, turns) -> None:
        """Ingest an iterable of objects exposing `.text` and `.idx`."""
        for turn in turns:
            self.ingest_turn(turn.text, turn.idx,
                             subject=getattr(turn, "subject", "user"))

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # -- querying ----------------------------------------------------------- #
    def answer(self, query: str, predicate: str, now_t: int,
               k: int = 8) -> Optional[str]:
        t0 = time.perf_counter()
        cands = self.retriever.retrieve(query, k, now_t, self.storage,
                                        self.embedder, self.prof)
        self.prof.retrieve_seconds += time.perf_counter() - t0
        matches = [c for c in cands if c.predicate == predicate and c.object]
        return matches[0].object if matches else None

    def recall_previous(self, predicate: str, now_t: int,
                        subject: str = "user") -> Optional[str]:
        """Most recently *closed* value for (subject, predicate) — history query.

        Only systems that keep a temporal trail (the bi-temporal graph) can
        serve this; append-only and key-overwrite stores return None.
        """
        closed = [i for i in self.storage.closed_edges(subject, predicate)
                  if i.t_valid_end is not None and i.t_valid_end <= now_t]
        if not closed:
            return None
        closed.sort(key=lambda i: i.t_valid_end, reverse=True)
        return closed[0].object

    def answer_nl(self, question: str, now_t: int, k: int = 8) -> Optional[str]:
        """Free-form answering for NL datasets (LoCoMo / LongMemEval).

        Retrieve the top-k valid memories, then let the LLM read and answer.
        Requires an LLM provider (mlx); returns "" with the mock provider.
        """
        if self.llm is None:
            return None
        items = self.retriever.retrieve(question, k, now_t, self.storage,
                                        self.embedder, self.prof)
        if not items:
            return None
        context = "\n".join(f"- (t={i.t_valid_start}) {i.text}" for i in items)
        prompt = (
            "Answer the question using only these memories. When they conflict, "
            "prefer the most recent; if the relevant fact was retracted, say it is "
            f"no longer true.\n\nMemories:\n{context}\n\n"
            f"Question: {question}\nShort answer:"
        )
        return self.llm.complete(prompt).strip()

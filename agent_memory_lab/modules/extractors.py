"""Extraction module: turn text -> triples.

`TemplateExtractor` is deterministic (regex over the StaleBench utterance
templates) so the offline lab needs no LLM. `LLMExtractor` prompts a real model
(mlx) for open-domain text. Both emit the SAME `Triple` shape, so backends don't
care which one is plugged in.

The templates live here as the single source of truth — the generator renders
utterances with `render_set` / `render_retract`, and the extractor parses them
back, exercising the full text->triple round-trip.
"""

from __future__ import annotations

import json
import re

from ..core.interfaces import Triple, Extractor

# predicate -> candidate values (a fact's possible objects)
PREDICATES: dict[str, list[str]] = {
    "editor":    ["Vim", "VS Code", "Neovim", "Zed", "Emacs", "Sublime Text"],
    "language":  ["Python", "Rust", "Go", "TypeScript", "Elixir", "Zig"],
    "framework": ["React", "Solid", "Svelte", "Vue", "Angular", "Qwik"],
    "city":      ["Hanoi", "Saigon", "Da Nang", "Singapore", "Tokyo", "Berlin"],
    "role":      ["junior dev", "senior engineer", "tech lead", "engineering manager"],
}

# a cue word that uniquely identifies the predicate in an utterance
_CUE: dict[str, str] = {
    "editor": "editor",
    "language": "language",
    "framework": "framework",
    "city": "city",
    "role": "role",
}

_SET_TEMPLATES: dict[str, str] = {
    "editor":    "I switched my main editor to {o}.",
    "language":  "My primary programming language now is {o}.",
    "framework": "These days my main framework is {o}.",
    "city":      "I just moved; my city is now {o}.",
    "role":      "My role at work is now {o}.",
}

_RETRACT_TEMPLATES: dict[str, str] = {
    "editor":    "I stopped using {o}; I have no main editor anymore.",
    "language":  "I no longer write {o}; that's not my language now.",
    "framework": "I stopped using the {o} framework completely.",
    "city":      "I no longer live in {o}; left that city for good.",
    "role":      "I no longer hold the {o} role.",
}

QUERIES: dict[str, str] = {
    # each query carries the predicate cue but NONE of the candidate values,
    # so old/new facts for a predicate tie on similarity and the tie-break
    # (recency vs FIFO) — i.e. the lifecycle policy — decides the answer.
    "editor":    "Which editor do I rely on these days?",
    "language":  "What programming language do I use now?",
    "framework": "Which framework do I build with now?",
    "city":      "Which city do I live in now?",
    "role":      "What role do I have at work now?",
}

_RETRACT_CUES = ("stopped using", "no longer")


def render_set(predicate: str, obj: str) -> str:
    return _SET_TEMPLATES[predicate].format(o=obj)


def render_retract(predicate: str, obj: str) -> str:
    return _RETRACT_TEMPLATES[predicate].format(o=obj)


def _find_value(low_text: str, values: list[str]) -> str | None:
    """Longest known value present in the text (word-boundary match)."""
    found = [v for v in values
             if re.search(r"\b" + re.escape(v.lower()) + r"\b", low_text)]
    return max(found, key=len) if found else None


class TemplateExtractor(Extractor):
    def extract(self, text: str, subject: str = "user") -> list[Triple]:
        low = text.lower()
        is_retract = any(cue in low for cue in _RETRACT_CUES)
        triples: list[Triple] = []
        for pred, values in PREDICATES.items():
            if _CUE[pred] not in low:
                continue
            obj = _find_value(low, values)
            if obj is None:
                continue
            triples.append(Triple(
                predicate=pred,
                object=None if is_retract else obj,
                text=text,
                is_retraction=is_retract,
            ))
        return triples


class LLMExtractor(Extractor):
    """Prompt a real model to emit triples (used with the mlx provider)."""

    _PROMPT = (
        "Extract durable facts about the user as JSON. Return a JSON array; each "
        "item is {{\"predicate\": str, \"object\": str|null, \"retraction\": bool}}. "
        "Use retraction=true (object=null) when the user says they stopped/no "
        "longer do something. If no durable fact, return [].\n\nText: {text}\nJSON:"
    )

    def __init__(self, llm) -> None:
        self.llm = llm

    def extract(self, text: str, subject: str = "user") -> list[Triple]:  # pragma: no cover - env dependent
        raw = self.llm.complete(self._PROMPT.format(text=text))
        try:
            start, end = raw.index("["), raw.rindex("]") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return []
        out: list[Triple] = []
        for d in data:
            pred = d.get("predicate")
            if not pred:
                continue
            retract = bool(d.get("retraction"))
            out.append(Triple(
                predicate=str(pred),
                object=None if retract else (d.get("object") or None),
                text=text,
                is_retraction=retract,
            ))
        return out

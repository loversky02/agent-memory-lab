"""LongMemEval adapter — multi-session memory with a `knowledge-update` type.

Schema (per item): `haystack_sessions` is a list of sessions, each a list of
`{role, content}` messages; `question` / `answer` / `question_type`. The
`knowledge-update` type is exactly the dynamic-update setting StaleBench targets.
"""

from __future__ import annotations

from .base import register, nl_probe, make_episode


@register("longmemeval")
def load_longmemeval(data, knowledge_update_only: bool = False) -> list:
    items = data if isinstance(data, list) else [data]
    episodes = []
    for it in items:
        qtype = it.get("question_type", "")
        if knowledge_update_only and "knowledge" not in qtype:
            continue
        texts = []
        for sess in it.get("haystack_sessions", []):
            for msg in sess:
                content = msg.get("content") or ""
                if content:
                    texts.append(f"{msg.get('role', '?')}: {content}")
        q, a = it.get("question"), it.get("answer")
        if not (texts and q):
            continue
        probes = [nl_probe(q, a, meta={"type": qtype, "qid": it.get("question_id")})]
        episodes.append(make_episode(
            "user", texts, probes,
            {"dataset": "longmemeval", "question_type": qtype}))
    return episodes

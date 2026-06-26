"""LoCoMo adapter — long multi-session conversations with QA (incl. temporal).

Schema (per sample): `conversation` holds `session_<n>` lists of
`{speaker, text, ...}`; `qa` holds `{question, answer, category}`. We flatten
sessions into ordered turns and turn each QA into an NL probe.
"""

from __future__ import annotations

import re

from .base import register, nl_probe, make_episode

_SESSION = re.compile(r"session_\d+$")


@register("locomo")
def load_locomo(data) -> list:
    samples = data if isinstance(data, list) else [data]
    episodes = []
    for s in samples:
        conv = s.get("conversation", {})
        sess_keys = sorted((k for k in conv if _SESSION.match(k)),
                           key=lambda k: int(k.split("_")[1]))
        texts = []
        for sk in sess_keys:
            for utt in conv[sk]:
                spk = utt.get("speaker", "?")
                txt = utt.get("text") or utt.get("clean_text") or ""
                if txt:
                    texts.append(f"{spk}: {txt}")
        probes = []
        for qa in s.get("qa", []):
            q = qa.get("question")
            a = qa.get("answer", qa.get("adversarial_answer"))
            if q:
                probes.append(nl_probe(q, a, meta={"category": qa.get("category")}))
        if texts and probes:
            episodes.append(make_episode(
                "user", texts, probes,
                {"dataset": "locomo", "sample_id": s.get("sample_id")}))
    return episodes

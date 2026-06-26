"""Real Apple-Silicon provider (Metal) via mlx-lm + mlx-embeddings.

Install extras first:  pip install "agent-memory-lab[mlx]"
Everything is lazily imported so the rest of the lab runs without MLX present.
Defaults target small models that are comfortable on an M-series Mac; override
with env vars AML_MLX_LLM / AML_MLX_EMBED.
"""

from __future__ import annotations

import math
import os

from .base import Embedder, LLM

DEFAULT_LLM = os.environ.get("AML_MLX_LLM", "mlx-community/Qwen2.5-7B-Instruct-4bit")
DEFAULT_EMBED = os.environ.get("AML_MLX_EMBED", "mlx-community/bge-small-en-v1.5-bf16")


def _l2(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec] if n > 0 else vec


class MLXEmbedder(Embedder):
    def __init__(self, model_id: str = DEFAULT_EMBED) -> None:
        super().__init__()
        try:
            from mlx_embeddings import load  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "mlx-embeddings is not installed. Run: pip install "
                "'agent-memory-lab[mlx]'"
            ) from exc
        self._model, self._tokenizer = load(model_id)

    def _embed(self, text: str) -> list[float]:  # pragma: no cover - env dependent
        out = self._model.encode([text], tokenizer=self._tokenizer)
        # mlx-embeddings returns pooled embeddings; coerce to a plain list.
        vec = out.text_embeds[0] if hasattr(out, "text_embeds") else out[0]
        try:
            vec = vec.tolist()
        except AttributeError:
            vec = list(vec)
        return _l2([float(x) for x in vec])


class MLXLLM(LLM):
    def __init__(self, model_id: str = DEFAULT_LLM) -> None:
        try:
            from mlx_lm import load  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "mlx-lm is not installed. Run: pip install "
                "'agent-memory-lab[mlx]'"
            ) from exc
        self._model, self._tokenizer = load(model_id)

    def complete(self, prompt: str) -> str:  # pragma: no cover - env dependent
        from mlx_lm import generate  # type: ignore
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):   # instruct models need the template
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        return generate(self._model, tok, prompt=prompt, max_tokens=96,
                        verbose=False).strip()

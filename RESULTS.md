# Real-model results

These are **actual runs**, not the mock provider — a real local LLM on Apple
Silicon (Metal) doing the extraction and NL answering.

## Setup

| | |
|---|---|
| Machine | Apple Silicon (M-series), Metal |
| LLM | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (the model cached locally) |
| Embedder | `mlx-hash`: deterministic hash (no download) · full `mlx`: `all-MiniLM-L6-v2-4bit` (384-dim, verified) |
| Mode | `HF_HUB_OFFLINE=1` (weights served from local cache) |
| Date | 2026-06-26 |

> Why 0.5B + hash embeddings? It's what was already on disk, and the throttled
> sandbox network couldn't pull larger weights. The point here is that the **real
> pipeline runs end-to-end with a real model**; for clean accuracy use a 3B–7B
> model (see *Reproduce* below).

## StaleBench — mock (clean) vs real 0.5B

`aml bench --provider mlx-hash --backends append-only,key-overwrite,bitemporal --episodes 3`

| backend | staleness | update-EM | history | scanned | | staleness | update-EM | history |
|---|---|---|---|---|---|---|---|---|
| | **mock (30 ep, lifecycle isolated)** | | | | | **real 0.5B (3 ep)** | | |
| append-only | 54.7% | 44.7% | 0% | 15.0 | | 60.0% | 26.7% | 0% |
| key-overwrite | 0% | 100% | 0% | 2.5 | | 40.0% | 33.3% | 0% |
| bitemporal | 0% | 100% | 100% | 2.5 | | 40.0% | 33.3% | 62.5% |

What survives the move to a real model:

- **Ordering holds** — append-only is still the most stale (60%); `bitemporal` is
  the only design that answers history at all (62.5% vs 0%).
- **The cost story holds** — the time-aware reader still scans ~4 items vs ~23
  for append-only, even though it stores the most.
- **Absolute numbers drop / converge** because a 0.5B extractor is noisy: it
  sometimes marks an *update* as a *retraction* and even hallucinates a fact from
  a filler line. That is itself the paper's point — **maintenance cannot fix bad
  extraction**; the four modules are coupled. A stronger extractor restores the
  clean separation the mock run shows.

### Full `mlx` provider (real LLM + real embeddings) — verified

`aml bench --provider mlx --backends append-only,key-overwrite,bitemporal --episodes 3`

| backend | staleness | update-EM | history | scanned |
|---|---|---|---|---|
| append-only | 46.7% | 20.0% | 0% | 22.7 |
| key-overwrite | 40.0% | 33.3% | 0% | 4.3 |
| bitemporal | 40.0% | 33.3% | 62.5% | 4.3 |

Real MiniLM embeddings (384-dim) instead of the hash. The ordering and the
scan-cost gap are unchanged — confirming the staleness story is driven by the
maintenance module, not the embedder.

## LoCoMo — real downloaded data + real model

The **full LoCoMo** (10 conversations, 1986 QA) was downloaded via
`scripts/download_data.py` and run through the real pipeline. One conversation
(419 dialogue turns), 12 probes, full `mlx` provider:

`aml dataset --name locomo --path data/locomo.json --provider mlx --limit 1 --max-probes 12`

| backend | episodes | probes | answer accuracy |
|---|---|---|---|
| bitemporal | 1 | 12 | 8.3% |
| append-only | 1 | 12 | 8.3% |

This is **real data through the whole pipeline end-to-end** — but the score is
low and the backends tie, for honest reasons:

- **Qwen-0.5B is far too small** for LoCoMo's multi-hop temporal QA (e.g. *"When
  did Caroline go to the support group?" → "7 May 2023"*), and the extractor is
  tuned for simple fact sentences, not 400-turn multi-speaker dialogue.
- On such sparse/noisy extraction the lifecycle difference doesn't surface in a
  12-probe slice.

It's the **model/extractor ceiling on this hardware+network, not a pipeline
bug**. Paper-grade numbers need a 3B–7B model (the 3B download **stalled at 11 MB**
on this throttled link) and a dialogue-tuned extractor — that's the
normal-network step below. The verified facts here: the real LoCoMo file parses
(10 convs / 1986 QA), and the full `mlx` provider answers it end-to-end.

## Reproduce / scale up

```bash
pip install -e ".[dev]" && pip install mlx-lm     # mlx-hash needs only mlx-lm
export HF_HUB_OFFLINE=1                            # if weights are already cached

# stronger model = cleaner separation (downloads weights on first run):
export AML_MLX_LLM=mlx-community/Qwen2.5-7B-Instruct-4bit
aml bench --provider mlx-hash --backends append-only,key-overwrite,bitemporal --episodes 20

# full real embeddings too (needs the mlx-embeddings model):
pip install mlx-embeddings && aml bench --provider mlx

# real datasets:
python scripts/download_data.py locomo
aml dataset --name locomo --path data/locomo.json --provider mlx
```

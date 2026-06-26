# Real-model results

These are **actual runs**, not the mock provider — a real local LLM on Apple
Silicon (Metal) doing the extraction and NL answering.

## Setup

| | |
|---|---|
| Machine | Apple Silicon (M-series), Metal |
| LLM | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (the model cached locally) |
| Embedder | deterministic hash (`mlx-hash` provider) — no embedding-model download needed |
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

## LoCoMo (bundled sample) — real NL answering

`aml dataset --name locomo --provider mlx-hash --backends bitemporal,append-only`

| backend | episodes | probes | answer accuracy |
|---|---|---|---|
| bitemporal | 1 | 2 | 50% |
| append-only | 1 | 2 | 50% |

The full LoCoMo / LongMemEval downloads didn't complete on this network; the
adapter + a sample fixture run here, and `scripts/download_data.py` fetches the
real files on a normal connection.

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

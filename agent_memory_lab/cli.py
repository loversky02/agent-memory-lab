"""Command line: `aml bench | demo | gen`  (or `python -m agent_memory_lab.cli`)."""

from __future__ import annotations

import argparse
import json
import sys

from .backends import BACKENDS, ALL_BACKENDS, build_backend
from .providers import get_provider
from .stalebench import generate_episode, controlled_episode
from .stalebench.metrics import evaluate_probe
from .stalebench.runner import run_benchmark

_PCT = ("staleness_rate", "update_em", "history_recall", "answer_accuracy")
_TABLE_COLS = ["backend", "staleness_rate", "update_em", "history_recall",
               "invalidation_latency", "avg_store_size", "avg_scanned"]


def _fmt(col: str, val) -> str:
    if col in _PCT:
        return f"{val * 100:5.1f}%"
    if col == "invalidation_latency":
        return "never" if val is None else str(val)
    return str(val)


def _print_table(rows: list[dict], cols: list[str]) -> None:
    headers = {c: c.replace("_", " ") for c in cols}
    cells = [[_fmt(c, r.get(c)) for c in cols] for r in rows]
    widths = [max(len(headers[c]), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = "  ".join(headers[c].ljust(widths[i]) for i, c in enumerate(cols))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(cols))))
    for row in cells:
        print("  ".join(row[i].rjust(widths[i]) if i else row[i].ljust(widths[i])
                        for i in range(len(cols))))


def _resolve_backends(arg: str) -> list[str]:
    if arg in ("all",):
        return ALL_BACKENDS
    if arg in ("default", "", None):
        return BACKENDS
    return [b.strip() for b in arg.split(",") if b.strip()]


def cmd_bench(args) -> int:
    backends = _resolve_backends(args.backends)
    rows = run_benchmark(
        backends, provider=args.provider, n_episodes=args.episodes,
        seed0=args.seed, scenario=args.scenario, chain_len=args.chain_len,
        filler_ratio=args.filler_ratio,
    )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"\nStaleBench  provider={args.provider}  episodes={args.episodes}  "
              f"scenario={args.scenario}\n")
        _print_table(rows, _TABLE_COLS)
        _headline(rows)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


def _headline(rows: list[dict]) -> None:
    by = {r["backend"]: r for r in rows}
    base = by.get("append-only")
    best = min(rows, key=lambda r: r["staleness_rate"])
    print()
    if base:
        print(f"append-only:  EM {base['update_em']*100:.0f}%  but "
              f"STALENESS {base['staleness_rate']*100:.0f}%  "
              f"(history recall {base['history_recall']*100:.0f}%)")
    print(f"lowest staleness: {best['backend']} at {best['staleness_rate']*100:.0f}% "
          f"(EM {best['update_em']*100:.0f}%, history {best['history_recall']*100:.0f}%)")
    print("note: end-to-end EM can look fine while staleness quietly corrupts recall.")


def cmd_demo(args) -> int:
    ep = generate_episode(args.seed, scenario=args.scenario) if args.seed is not None \
        else controlled_episode()
    print("=== Conversation stream ===")
    for t in ep.turns:
        tag = "" if t.kind == "filler" else f"  <{t.kind}:{t.predicate}={t.value}>"
        print(f"  [{t.idx:>2}] {t.text}{tag}")

    provider = get_provider(args.provider)
    for name in BACKENDS:
        system = build_backend(name, provider)
        system.ingest(ep.turns)
        print(f"\n=== {name} ===")
        for pr in ep.probes:
            rec = evaluate_probe(system, pr, ep.ask_turn)
            flag = "STALE" if rec["stale"] else ("ok" if rec["correct"] else "miss")
            print(f"  {pr.predicate:<10} {pr.kind:<8} -> {str(rec['predicted']):<14}"
                  f" (gold {str(rec['gold'])})  [{flag}]")
    return 0


def cmd_gen(args) -> int:
    ep = generate_episode(args.seed, scenario=args.scenario)
    print(f"# episode seed={args.seed} scenario={args.scenario} "
          f"retracted={ep.meta['retracted']}")
    for t in ep.turns:
        print(f"[{t.idx:>2}] {t.text}")
    print("\n# probes")
    for pr in ep.probes:
        print(f"  {pr.predicate} ({pr.kind}) gold={pr.gold} "
              f"superseded={pr.superseded} target={pr.target}")
    return 0


def cmd_dataset(args) -> int:
    from .datasets import run_dataset
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if args.provider == "mock":
        print("note: NL datasets need a real LLM — answers will be empty on the "
              "mock provider. Use --provider mlx for meaningful results.",
              file=sys.stderr)
    rows = run_dataset(args.name, path=args.path, provider=args.provider,
                       backends=backends, limit=args.limit, max_probes=args.max_probes)
    print(f"\n{args.name}  provider={args.provider}\n")
    _print_table(rows, ["backend", "dataset", "episodes", "probes", "answer_accuracy"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aml", description="Agent Memory Lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="run StaleBench across backends")
    b.add_argument("--backends", default="default",
                   help="'default', 'all', or comma list. choices: " + ",".join(ALL_BACKENDS))
    b.add_argument("--provider", default="mock", choices=["mock", "mlx", "mlx-hash"])
    b.add_argument("--episodes", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--scenario", default="all", choices=["basic", "retract", "all"])
    b.add_argument("--chain-len", type=int, default=3, dest="chain_len")
    b.add_argument("--filler-ratio", type=float, default=0.5, dest="filler_ratio")
    b.add_argument("--out", default=None, help="write results JSON here")
    b.add_argument("--json", action="store_true", help="print JSON instead of table")
    b.set_defaults(func=cmd_bench)

    d = sub.add_parser("demo", help="show one episode + per-backend answers")
    d.add_argument("--provider", default="mock", choices=["mock", "mlx", "mlx-hash"])
    d.add_argument("--seed", type=int, default=None,
                   help="omit for the curated controlled episode")
    d.add_argument("--scenario", default="all", choices=["basic", "retract", "all"])
    d.set_defaults(func=cmd_demo)

    g = sub.add_parser("gen", help="print a generated episode")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--scenario", default="all", choices=["basic", "retract", "all"])
    g.set_defaults(func=cmd_gen)

    ds = sub.add_parser("dataset", help="run a real NL benchmark (LoCoMo/LongMemEval)")
    ds.add_argument("--name", required=True, choices=["locomo", "longmemeval"])
    ds.add_argument("--path", default=None,
                    help="path to a full dataset JSON (defaults to bundled sample)")
    ds.add_argument("--provider", default="mock", choices=["mock", "mlx", "mlx-hash"])
    ds.add_argument("--backends", default="bitemporal,append-only",
                    help="comma list of backends to compare")
    ds.add_argument("--limit", type=int, default=None, help="cap number of items")
    ds.add_argument("--max-probes", type=int, default=None, dest="max_probes",
                    help="cap probes answered per item (bounds runtime)")
    ds.set_defaults(func=cmd_dataset)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

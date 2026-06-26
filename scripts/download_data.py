"""Download the real benchmark datasets the paper uses.

    python scripts/download_data.py locomo        # -> data/locomo.json  (auto, ~small)
    python scripts/download_data.py longmemeval    # prints official steps (gated mirror)

LoCoMo is a single public JSON on GitHub, so we fetch it directly. LongMemEval
is distributed via the authors' release (HF/Drive), so we point you at it and,
if `huggingface_hub` is installed, try the HF mirror.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LONGMEMEVAL_HF = "xiaowu0162/longmemeval"   # HF dataset mirror, if available


def _save(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "agent-memory-lab"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as fh:
        fh.write(r.read())
    print(f"saved {dest}  ({os.path.getsize(dest)/1024:.0f} KB)")


def get_locomo() -> str:
    dest = os.path.join(DATA, "locomo.json")
    _save(LOCOMO_URL, dest)
    return dest


def get_longmemeval() -> str | None:
    dest = os.path.join(DATA, "longmemeval.json")
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception:
        print("LongMemEval is a gated/large release. Get it from:")
        print("  https://github.com/xiaowu0162/LongMemEval  (longmemeval_s.json)")
        print("then run:  aml dataset --name longmemeval --path data/longmemeval.json --provider mlx")
        return None
    try:
        path = hf_hub_download(repo_id=LONGMEMEVAL_HF, filename="longmemeval_s.json",
                               repo_type="dataset")
        os.makedirs(DATA, exist_ok=True)
        os.replace(path, dest)
        print(f"saved {dest}")
        return dest
    except Exception as exc:  # noqa: BLE001
        print(f"HF mirror unavailable ({exc}). See the GitHub release instead:")
        print("  https://github.com/xiaowu0162/LongMemEval")
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="download real memory benchmarks")
    ap.add_argument("dataset", choices=["locomo", "longmemeval"])
    args = ap.parse_args(argv)
    path = get_locomo() if args.dataset == "locomo" else get_longmemeval()
    if path:
        print(f"\nrun:  aml dataset --name {args.dataset} --path {path} --provider mlx")
    return 0 if path else 1


if __name__ == "__main__":
    sys.exit(main())

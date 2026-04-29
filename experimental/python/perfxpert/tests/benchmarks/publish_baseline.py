"""Capture an initial baseline from a trusted manual run.

Usage (one-shot, MI300X host only):
    python3 tests/benchmarks/publish_baseline.py \
       --input  tritonbench_results.json kernelbench_results.json \
       --output benchmarks/baseline/mi300x_baseline.json

Subsequent baseline bumps are explicit PRs that replace the file —
the nightly workflow never writes to this path.
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    merged = {"suites": {}}
    for p in args.input:
        data = json.loads(p.read_text())
        merged["suites"][data["suite"]] = data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

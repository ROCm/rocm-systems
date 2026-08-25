#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Compare two RCCL test runs per-configuration (init-pipeline correctness gate).

Reads the ``tests.jsonl`` each run emits under --emit-results and reports the
per-config diff between a serial baseline and an init-pipeline (or any other)
candidate. Exits non-zero if the candidate regressed any config or dropped
coverage -- this is the plan §12 gate: identical per-config pass/fail/skip.

Usage:
  compare_results.py BASELINE.jsonl CANDIDATE.jsonl \
      [--exclude '*_CuMem1' --exclude '*PosixFd*'] [--quiet]

Exit codes: 0 = gate PASS, 1 = gate FAIL (regressions / dropped coverage),
2 = usage/IO error.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.results_diff import diff_results, format_report  # noqa: E402


def _load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-config diff of two RCCL test runs.")
    ap.add_argument("baseline", help="baseline tests.jsonl (e.g. serial run)")
    ap.add_argument("candidate", help="candidate tests.jsonl (e.g. init-pipeline run)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob of test names to ignore (known pre-existing failures); repeatable")
    ap.add_argument("--quiet", action="store_true", help="print only the summary + gate line")
    args = ap.parse_args(argv)

    try:
        base = _load_jsonl(args.baseline)
        cand = _load_jsonl(args.candidate)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    diff = diff_results(base, cand, exclude=args.exclude)
    if args.quiet:
        print(f"matched={diff['matched']} regressions={len(diff['regressions'])} "
              f"dropped={len(diff['only_baseline'])} fixes={len(diff['fixes'])} "
              f"changed={len(diff['changed'])} new={len(diff['only_candidate'])} "
              f"GATE={'PASS' if diff['gate_ok'] else 'FAIL'}")
    else:
        print(format_report(diff,
                            baseline_label=os.path.basename(args.baseline),
                            candidate_label=os.path.basename(args.candidate)))
    return 0 if diff["gate_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

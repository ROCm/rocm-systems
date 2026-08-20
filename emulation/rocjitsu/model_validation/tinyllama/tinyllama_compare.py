#!/usr/bin/env python3
"""Compare TinyLlama JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compare_logits(
    baseline: list[list[float]],
    candidate: list[list[float]],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        return {"allclose": False, "max_abs": float("inf"), "mismatch_count": -1}

    max_abs = 0.0
    mismatch_count = 0
    for base_step, candidate_step in zip(baseline, candidate):
        if len(base_step) != len(candidate_step):
            return {"allclose": False, "max_abs": float("inf"), "mismatch_count": -1}
        for expected, actual in zip(base_step, candidate_step):
            diff = abs(expected - actual)
            max_abs = max(max_abs, diff)
            if diff > atol + rtol * abs(expected):
                mismatch_count += 1
    return {
        "allclose": mismatch_count == 0,
        "max_abs": max_abs,
        "mismatch_count": mismatch_count,
    }


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline)
    for candidate_path in args.candidates:
        candidate = load(candidate_path)
        result = {
            "baseline": str(args.baseline),
            "candidate": str(candidate_path),
            "candidate_arch": candidate.get("cuda_arch", "cpu"),
            "sequence_ids_match": baseline["sequence_ids"] == candidate["sequence_ids"],
            "new_token_ids_match": baseline["new_token_ids"] == candidate["new_token_ids"],
        }
        result.update(
            compare_logits(
                baseline["logits"],
                candidate["logits"],
                rtol=args.rtol,
                atol=args.atol,
            )
        )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

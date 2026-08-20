#!/usr/bin/env python3
"""Compare saved causal-LM generation JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from common import ATOL, RTOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compare_logits(
    baseline: list[list[float]],
    candidate: list[list[float]],
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        return {
            "step_count_match": False,
            "baseline_steps": len(baseline),
            "candidate_steps": len(candidate),
            "max_abs": None,
            "allclose": False,
            "mismatch_count": -1,
        }

    max_abs = 0.0
    mismatch_count = 0
    nonfinite_count = 0
    for step_index, (baseline_step, candidate_step) in enumerate(zip(baseline, candidate)):
        if len(baseline_step) != len(candidate_step):
            return {
                "step_count_match": True,
                "vocab_size_match": False,
                "mismatch_step": step_index,
                "baseline_vocab_size": len(baseline_step),
                "candidate_vocab_size": len(candidate_step),
                "max_abs": None,
                "allclose": False,
                "mismatch_count": -1,
            }
        for expected, actual in zip(baseline_step, candidate_step):
            if not math.isfinite(expected) or not math.isfinite(actual):
                mismatch_count += 1
                nonfinite_count += 1
                continue
            diff = abs(expected - actual)
            max_abs = max(max_abs, diff)
            if diff > ATOL + RTOL * abs(expected):
                mismatch_count += 1

    return {
        "step_count_match": True,
        "vocab_size_match": True,
        "steps": len(baseline),
        "vocab_size": len(baseline[0]) if baseline else 0,
        "max_abs": max_abs,
        "allclose": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "nonfinite_count": nonfinite_count,
    }


def compare_one(
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    baseline = load(baseline_path)
    candidate = load(candidate_path)
    result = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_model_id": baseline.get("model_id"),
        "candidate_model_id": candidate.get("model_id"),
        "baseline_device": baseline.get("device"),
        "candidate_device": candidate.get("device"),
        "baseline_arch": baseline.get("cuda_arch", "cpu"),
        "candidate_arch": candidate.get("cuda_arch", "cpu"),
        "sequence_ids_match": baseline.get("sequence_ids") == candidate.get("sequence_ids"),
        "new_token_ids_match": baseline.get("new_token_ids") == candidate.get("new_token_ids"),
        "baseline_new_token_ids": baseline.get("new_token_ids"),
        "candidate_new_token_ids": candidate.get("new_token_ids"),
        "baseline_decoded_text": baseline.get("decoded_text"),
        "candidate_decoded_text": candidate.get("decoded_text"),
    }
    result.update(
        compare_logits(
            baseline["logits"],
            candidate["logits"],
        )
    )
    return result


def main() -> int:
    args = parse_args()
    results = [
        compare_one(args.baseline, candidate)
        for candidate in args.candidates
    ]
    lines = [json.dumps(result, sort_keys=True) for result in results]
    for line in lines:
        print(line)

    for result in results:
        if (
            not result["sequence_ids_match"]
            or not result["new_token_ids_match"]
            or not result["allclose"]
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

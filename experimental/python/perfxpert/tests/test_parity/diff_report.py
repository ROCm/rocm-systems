"""Structured diff of observed vs expected parity results."""

from __future__ import annotations

from typing import List

from .parity_runner import ParityResult


def field_level_diffs(result: ParityResult) -> List[str]:
    diffs: List[str] = []
    if not result.agree_bottleneck():
        diffs.append(
            f"bottleneck: expected={result.expected_bottleneck!r} "
            f"observed={result.observed.primary_bottleneck!r}"
        )
    if not result.agree_rec_type():
        diffs.append(
            f"rec_type: expected={result.expected_rec_type!r} "
            f"observed={result.observed.primary_rec_type!r}"
        )
    if not result.agree_rec_technique():
        diffs.append(
            f"rec_technique: expected={result.expected_rec_technique!r} "
            f"observed={result.observed.primary_rec_technique!r}"
        )
    return diffs


def summarize_for_failure_message(result: ParityResult) -> str:
    diffs = field_level_diffs(result)
    if not diffs:
        return f"(no diff on {result.fixture_id})"
    return (
        f"Fixture {result.fixture_id} — {len(diffs)} field disagreements:\n  "
        + "\n  ".join(diffs)
    )

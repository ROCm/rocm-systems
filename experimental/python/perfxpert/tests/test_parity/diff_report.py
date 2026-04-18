"""Structured diff of old vs new analysis results — used in test failure
messages and in the exit dashboard.
"""

from __future__ import annotations

from typing import List

from .parity_runner import DualResult


def field_level_diffs(dual: DualResult) -> List[str]:
    diffs: List[str] = []
    if not dual.agree_bottleneck():
        diffs.append(
            f"bottleneck: old={dual.old.primary_bottleneck!r} "
            f"new={dual.new.primary_bottleneck!r}"
        )
    if not dual.agree_rec_type():
        diffs.append(
            f"rec_type: old={dual.old.primary_rec_type!r} "
            f"new={dual.new.primary_rec_type!r}"
        )
    if not dual.agree_rec_technique():
        diffs.append(
            f"rec_technique: old={dual.old.primary_rec_technique!r} "
            f"new={dual.new.primary_rec_technique!r}"
        )
    return diffs


def summarize_for_failure_message(dual: DualResult) -> str:
    diffs = field_level_diffs(dual)
    if not diffs:
        return f"(no diff on {dual.fixture_id})"
    return (
        f"Fixture {dual.fixture_id} — {len(diffs)} field disagreements:\n  "
        + "\n  ".join(diffs)
    )

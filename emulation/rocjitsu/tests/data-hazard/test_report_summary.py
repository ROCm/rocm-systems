# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Pytest: mutation-score accounting in ``mutate_and_test.report_writer``.

Run from ``tests/data-hazard`` (so ``mutate_and_test`` imports)::

    cd tests/data-hazard && pytest test_report_summary.py -v

These are pure functions over report dataclasses, so no launcher, GPU or ROCm
installation is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from mutate_and_test.models import MutantResult, ShaderReport  # noqa: E402
from mutate_and_test.report_writer import compute_summary  # noqa: E402


def _mutant(index: int, *, ran: bool, correct: bool, hazards: int = 0) -> MutantResult:
    return MutantResult(
        shader="demo",
        wait_index=index,
        wait_instruction="s_wait_loadcnt 0x0",
        wait_line=10 + index,
        compiled=True,
        linked=True,
        ran=ran,
        correct=correct,
        exit_code=0 if ran else -11,
        hazard_count=hazards,
    )


def _report(*mutants: MutantResult) -> ShaderReport:
    return ShaderReport(
        shader="demo",
        asm_file="demo.s",
        compile_ok=True,
        build_ok=True,
        baseline_ok=True,
        baseline_exit_code=0,
        mutants=list(mutants),
    )


def test_crashed_mutants_are_inconclusive_not_killed() -> None:
    """A run where nothing executed must not report a perfect score."""
    report = _report(
        _mutant(0, ran=False, correct=False),
        _mutant(1, ran=False, correct=False),
    )

    summary = compute_summary([report])

    assert summary["killed"] == 0
    assert summary["survived"] == 0
    assert summary["inconclusive"] == 2
    assert summary["scored_mutants"] == 0
    assert summary["mutation_score"] == 0.0


def test_score_is_taken_over_mutants_that_ran() -> None:
    report = _report(
        _mutant(0, ran=True, correct=False, hazards=3),
        _mutant(1, ran=True, correct=True),
        _mutant(2, ran=False, correct=False),
    )

    summary = compute_summary([report])

    assert summary["killed"] == 1
    assert summary["survived"] == 1
    assert summary["inconclusive"] == 1
    assert summary["scored_mutants"] == 2
    assert summary["mutation_score"] == 50.0


def test_every_mutant_running_leaves_nothing_inconclusive() -> None:
    report = _report(
        _mutant(0, ran=True, correct=False, hazards=1),
        _mutant(1, ran=True, correct=False, hazards=2),
    )

    summary = compute_summary([report])

    assert summary["inconclusive"] == 0
    assert summary["scored_mutants"] == 2
    assert summary["mutation_score"] == 100.0

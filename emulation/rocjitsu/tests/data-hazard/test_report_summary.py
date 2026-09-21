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

import json
import sys
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from mutate_and_test.models import MutantResult, ShaderReport  # noqa: E402
from mutate_and_test.report_writer import (  # noqa: E402
    compute_summary,
    write_json_report,
    write_markdown_report,
    write_terminal_report,
)


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


def _report(
    *mutants: MutantResult,
    baseline_exit_code: int = 0,
    baseline_hazards: int = 0,
) -> ShaderReport:
    return ShaderReport(
        shader="demo",
        asm_file="demo.s",
        compile_ok=True,
        build_ok=True,
        baseline_ok=baseline_exit_code == 0,
        baseline_exit_code=baseline_exit_code,
        baseline_hazard_count=baseline_hazards,
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


def test_hazards_on_a_clean_baseline_are_a_false_positive() -> None:
    summary = compute_summary([_report(baseline_hazards=2)])

    assert summary["false_positives"] == 1
    assert summary["fp"] == 1
    assert summary["tn"] == 0
    assert summary["na"] == 0


def test_a_quiet_clean_baseline_is_a_true_negative() -> None:
    summary = compute_summary([_report()])

    assert summary["false_positives"] == 0
    assert summary["fp"] == 0
    assert summary["tn"] == 1


def test_hazards_from_a_baseline_that_crashed_convict_nothing() -> None:
    """A partial report flushed on the way out is not evidence either way."""
    summary = compute_summary([_report(baseline_exit_code=-11, baseline_hazards=3)])

    assert summary["false_positives"] == 0
    assert summary["fp"] == 0
    assert summary["tn"] == 0
    assert summary["na"] == 1


def test_a_baseline_that_timed_out_is_not_a_true_negative() -> None:
    """Reporting nothing is only creditable when the run finished."""
    summary = compute_summary([_report(baseline_exit_code=-1)])

    assert summary["fp"] == 0
    assert summary["tn"] == 0
    assert summary["na"] == 1


def test_reports_say_why_a_crashed_baselines_hazards_are_not_held_against_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    crashed = _report(baseline_exit_code=-11, baseline_hazards=3)
    crashed.baseline_hazard_report_raw = "[]"
    markdown = tmp_path / "mutation_report.md"

    write_terminal_report([crashed])
    write_markdown_report([crashed], markdown, "gfx1250")

    assert "Baseline hazards: 3 (BASELINE DID NOT FINISH)" in capsys.readouterr().out
    assert "**3** (baseline did not finish)" in markdown.read_text()


def test_json_report_keeps_a_crashed_baselines_hazards_unaccused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mutation_report.json"

    write_json_report(
        [_report(baseline_exit_code=-11, baseline_hazards=3)], path, "gfx1250"
    )

    shader = json.loads(path.read_text())["shaders"][0]
    assert shader["baseline_hazard_count"] == 3
    assert shader["baseline_false_positive"] is False

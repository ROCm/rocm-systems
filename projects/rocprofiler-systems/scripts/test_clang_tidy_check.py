#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/clang-tidy-check.py."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("clang-tidy-check.py")


def _load_module():
    """Import clang-tidy-check.py (its filename isn't a valid identifier).

    The module must be registered in sys.modules *before* exec: its dataclasses
    use `from __future__ import annotations`, so dataclass field-type resolution
    looks the module up via sys.modules[cls.__module__].
    """
    spec = importlib.util.spec_from_file_location("clang_tidy_check", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ctc = _load_module()


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "clang_tidy_binary": "clang-tidy",
        "checks": "",
        "build_path": "/build",
        "jobs": 4,
        "timeout": None,
        "base": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _diag(line: int, checks: list[str], *, severity: str = "warning") -> "ctc.Diagnostic":
    return ctc.Diagnostic(
        file="src/foo.cpp",
        line=line,
        col=1,
        severity=severity,
        message="msg",
        checks=checks,
    )


# --------------------------------------------------------------------------- #
# in_line_ranges
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "line,ranges,expected",
    [
        (15, [(10, 20)], True),
        (10, [(10, 20)], True),  # inclusive start
        (20, [(10, 20)], True),  # inclusive end
        (9, [(10, 20)], False),
        (21, [(10, 20)], False),
        (5, [], False),  # no ranges
        (30, [(10, 20), (25, 40)], True),  # second range
    ],
)
def test_in_line_ranges(line, ranges, expected):
    assert ctc.in_line_ranges(line, ranges) is expected


# --------------------------------------------------------------------------- #
# _tidy_args
# --------------------------------------------------------------------------- #
def test_tidy_args_without_checks():
    assert ctc._tidy_args(_args(checks="")) == ["-p=/build"]


def test_tidy_args_with_checks():
    assert ctc._tidy_args(_args(checks="misc-*")) == ["-checks=misc-*", "-p=/build"]


def test_build_command_shape():
    cmd = ctc.build_command("/repo/src/foo.cpp", _args(clang_tidy_binary="clang-tidy-18"))
    assert cmd == ["clang-tidy-18", "-p=/build", "/repo/src/foo.cpp"]


# --------------------------------------------------------------------------- #
# parse_diagnostics
# --------------------------------------------------------------------------- #
def test_parse_diagnostics_single():
    output = "/repo/src/foo.cpp:12:5: warning: some message [misc-const-correctness]\n"
    diags = ctc.parse_diagnostics(output, "/repo")
    assert len(diags) == 1
    d = diags[0]
    assert d.file == "src/foo.cpp"
    assert (d.line, d.col, d.severity) == (12, 5, "warning")
    assert d.message == "some message"
    assert d.checks == ["misc-const-correctness"]


def test_parse_diagnostics_multiple_checks_and_noise():
    output = (
        "Running with 4 threads...\n"  # non-matching noise, ignored
        "/repo/a.cpp:1:1: error: bad [check-a,check-b]\n"
    )
    diags = ctc.parse_diagnostics(output, "/repo")
    assert len(diags) == 1
    assert diags[0].checks == ["check-a", "check-b"]
    assert diags[0].severity == "error"


def test_parse_diagnostics_ignores_fatal_error_line():
    # clang compile failures have no [check] suffix, so they must not parse.
    output = "/repo/a.cpp:1:1: fatal error: 'x.h' file not found\n"
    assert ctc.parse_diagnostics(output, "/repo") == []


# --------------------------------------------------------------------------- #
# aggregate_diagnostics  (the pure core)
# --------------------------------------------------------------------------- #
def test_aggregate_empty():
    agg = ctc.aggregate_diagnostics([])
    assert agg.in_diff == {}
    assert agg.preexisting == {}
    assert agg.any_timed_out is False


def test_aggregate_buckets_by_line_range():
    cf = ctc.ChangedFile("src/foo.cpp", [(10, 20)])
    in_range = _diag(15, ["misc-const-correctness"])
    out_range = _diag(5, ["misc-const-correctness"])
    agg = ctc.aggregate_diagnostics([ctc.FileResult(cf, [in_range, out_range], True)])
    assert agg.in_diff["misc-const-correctness"] == [in_range]
    assert agg.preexisting["misc-const-correctness"] == [out_range]
    assert agg.any_timed_out is False


def test_aggregate_multi_check_diagnostic_appears_under_each_key():
    cf = ctc.ChangedFile("src/foo.cpp", [(1, 100)])
    d = _diag(10, ["check-a", "check-b"])
    agg = ctc.aggregate_diagnostics([ctc.FileResult(cf, [d], True)])
    assert agg.in_diff["check-a"] == [d]
    assert agg.in_diff["check-b"] == [d]
    assert agg.preexisting == {}


def test_aggregate_accumulates_same_check():
    cf = ctc.ChangedFile("src/foo.cpp", [(1, 100)])
    d1 = _diag(10, ["misc-const-correctness"])
    d2 = _diag(20, ["misc-const-correctness"])
    agg = ctc.aggregate_diagnostics([ctc.FileResult(cf, [d1, d2], True)])
    assert agg.in_diff["misc-const-correctness"] == [d1, d2]


def test_aggregate_failed_result_sets_any_timed_out():
    cf = ctc.ChangedFile("src/foo.cpp", [(1, 100)])
    agg = ctc.aggregate_diagnostics([ctc.FileResult(cf, [], False)])
    assert agg.any_timed_out is True
    assert agg.in_diff == {}


# --------------------------------------------------------------------------- #
# report_progress
# --------------------------------------------------------------------------- #
def test_report_progress_writes_to_stderr(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")  # deterministic, no ANSI codes
    ctc.report_progress(2, 5, ctc.ChangedFile("src/foo.cpp"))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "  [2/5] src/foo.cpp\n"


# --------------------------------------------------------------------------- #
# run_checks  (orchestration, with check_file injected)
# --------------------------------------------------------------------------- #
def test_run_checks_returns_one_result_per_file_and_reports_progress(monkeypatch):
    files = [ctc.ChangedFile(f"src/f{i}.cpp", [(1, 1)]) for i in range(5)]

    def fake_check_file(changed_file, args, repo_root):
        return ctc.FileResult(changed_file, [], True)

    monkeypatch.setattr(ctc, "check_file", fake_check_file)

    calls = []
    results = ctc.run_checks(
        files, _args(), "/repo", on_complete=lambda c, t, cf: calls.append((c, t, cf))
    )

    # one result per input file
    assert {r.changed_file.path for r in results} == {f.path for f in files}
    assert len(results) == len(files)
    # progress fired once per file: counters are exactly 1..N, total constant
    counters = sorted(c for c, _, _ in calls)
    assert counters == list(range(1, len(files) + 1))
    assert {t for _, t, _ in calls} == {len(files)}
    assert {cf.path for _, _, cf in calls} == {f.path for f in files}


def test_run_checks_without_callback(monkeypatch):
    files = [ctc.ChangedFile("src/a.cpp", [(1, 1)])]
    monkeypatch.setattr(ctc, "check_file", lambda cf, a, r: ctc.FileResult(cf, [], True))
    results = ctc.run_checks(files, _args(), "/repo")  # on_complete defaults to None
    assert len(results) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/clang-tidy-check.py."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("clang-tidy-check.py")


def _load_module():
    """Import clang-tidy-check.py (its filename isn't a valid identifier)."""
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


def _completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


class _CaptureRun:
    """Stand-in for subprocess.run that records the command and kwargs."""

    def __init__(self, result=None):
        self.cmd = None
        self.kwargs = None
        self._result = result

    def __call__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        return self._result if self._result is not None else _completed(cmd)


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
# classify_diagnostic
# --------------------------------------------------------------------------- #
def _diagnostic(file: str, line: int) -> "ctc.Diagnostic":
    return ctc.Diagnostic(
        file=file, line=line, col=1, severity="warning", message="msg", checks=["c"]
    )


def test_classify_diagnostic_in_diff():
    changed_file = ctc.ChangedFile("src/foo.cpp", [(10, 20)])
    diagnostic = _diagnostic("src/foo.cpp", 15)
    assert ctc.classify_diagnostic(diagnostic, changed_file) == "in_diff"


def test_classify_diagnostic_preexisting_same_file_outside_diff():
    changed_file = ctc.ChangedFile("src/foo.cpp", [(10, 20)])
    diagnostic = _diagnostic("src/foo.cpp", 999)
    assert ctc.classify_diagnostic(diagnostic, changed_file) == "preexisting"


def test_classify_diagnostic_ignores_included_header():
    # clang-tidy analyzes the whole translation unit; a diagnostic located in
    # a header the changed file includes must not be attributed to it, even
    # when the header's line number happens to fall inside the diff range.
    changed_file = ctc.ChangedFile("src/foo.cpp", [(10, 20)])
    diagnostic = _diagnostic("include/foo.hpp", 15)
    assert ctc.classify_diagnostic(diagnostic, changed_file) is None


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
# _clang_tidy  (command construction + missing binary)
# --------------------------------------------------------------------------- #
def test_clang_tidy_command_without_checks(monkeypatch):
    run = _CaptureRun()
    monkeypatch.setattr(ctc.subprocess, "run", run)
    ctc._clang_tidy(_args(checks=""), "-list-checks")
    assert run.cmd == ["clang-tidy", "-p=/build", "-list-checks"]
    # check=False is behavioral: clang-tidy exits nonzero when it finds issues,
    # so check=True would raise on any file with diagnostics and break the tool.
    assert run.kwargs["check"] is False


def test_clang_tidy_command_with_checks_and_timeout(monkeypatch):
    run = _CaptureRun()
    monkeypatch.setattr(ctc.subprocess, "run", run)
    ctc._clang_tidy(
        _args(checks="misc-*", clang_tidy_binary="clang-tidy-18"), "/f.cpp", timeout=5
    )
    assert run.cmd == ["clang-tidy-18", "-checks=misc-*", "-p=/build", "/f.cpp"]
    assert run.kwargs["timeout"] == 5


def test_clang_tidy_missing_binary_exits(monkeypatch):
    def boom(_cmd, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(ctc.subprocess, "run", boom)
    with pytest.raises(SystemExit) as excinfo:
        ctc._clang_tidy(_args(clang_tidy_binary="nope"), "/f.cpp")
    assert "nope" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# run_clang_tidy
# --------------------------------------------------------------------------- #
def test_run_clang_tidy_merges_stdout_stderr(monkeypatch):
    monkeypatch.setattr(
        ctc,
        "_clang_tidy",
        lambda a, *e, timeout=None: _completed(list(e), 0, "out\n", "err\n"),
    )
    output, ok = ctc.run_clang_tidy(_args(), "/f.cpp", None)
    assert ok is True
    assert output == "out\nerr\n"


def test_run_clang_tidy_timeout_returns_false(monkeypatch):
    def boom(_a, *e, timeout=None):
        raise subprocess.TimeoutExpired(cmd=list(e), timeout=timeout)

    monkeypatch.setattr(ctc, "_clang_tidy", boom)
    output, ok = ctc.run_clang_tidy(_args(), "/f.cpp", 3)
    assert ok is False
    assert "Timed out after 3s on /f.cpp" in output


# --------------------------------------------------------------------------- #
# has_enabled_checks
# --------------------------------------------------------------------------- #
def test_has_enabled_checks_true_when_list_populated(monkeypatch):
    out = (
        "Enabled checks:\n"
        "    misc-const-correctness\n"
        "    modernize-use-auto\n"
        "\n"
        "trailing noise\n"
    )
    monkeypatch.setattr(
        ctc, "_clang_tidy", lambda a, *e, timeout=None: _completed(list(e), 0, out, "")
    )
    assert ctc.has_enabled_checks(_args()) is True


def test_has_enabled_checks_failure_warns_and_returns_false(monkeypatch, capsys):
    monkeypatch.setattr(
        ctc, "_clang_tidy", lambda a, *e, timeout=None: _completed(list(e), 1, "", "boom")
    )
    assert ctc.has_enabled_checks(_args()) is False
    assert "could not resolve check list" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# print_timed_out_files
# --------------------------------------------------------------------------- #
def test_print_timed_out_files_lists_each_file(capsys):
    ctc.print_timed_out_files(["a.cpp", "b.cpp"], 600)
    out = capsys.readouterr().out
    # A timeout is otherwise a silent failure: it makes the run exit nonzero
    # while the diff section still reports "No issues found."
    assert "Timed out after 600s (2)" in out
    assert "a.cpp" in out
    assert "b.cpp" in out


def test_print_timed_out_files_prints_nothing_when_empty(capsys):
    ctc.print_timed_out_files([], 600)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# _git  (error handling)
# --------------------------------------------------------------------------- #
def test_git_missing_binary_exits(monkeypatch):
    def boom(_cmd, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(ctc.subprocess, "run", boom)
    with pytest.raises(SystemExit) as excinfo:
        ctc._git(["status"])
    assert "git" in str(excinfo.value).lower()


def test_git_failure_surfaces_stderr(monkeypatch):
    def boom(cmd, **_kwargs):
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=cmd,
            stderr="fatal: detected dubious ownership in repository at '/repo'",
        )

    monkeypatch.setattr(ctc.subprocess, "run", boom)
    with pytest.raises(SystemExit) as excinfo:
        ctc._git(["rev-parse", "--show-toplevel"])
    # git's own stderr (the actionable safe.directory hint) must reach the user.
    assert "dubious ownership" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# get_diff_changed_files
# --------------------------------------------------------------------------- #
_DIFF = """\
diff --git a/src/foo.cpp b/src/foo.cpp
--- a/src/foo.cpp
+++ b/src/foo.cpp
@@ -10,0 +11,3 @@
+a
+b
+c
@@ -30,1 +34,0 @@
-gone
diff --git a/notes.txt b/notes.txt
--- a/notes.txt
+++ b/notes.txt
@@ -1,0 +2,5 @@
+x
"""


def test_get_diff_changed_files_parses_ranges_and_filters(monkeypatch):
    monkeypatch.setattr(ctc, "_git", lambda git_args, cwd=None: _DIFF)
    monkeypatch.setattr(ctc.os.path, "isfile", lambda p: True)
    changed = ctc.get_diff_changed_files("/repo", "origin/develop")
    # notes.txt is not a source extension -> excluded
    assert set(changed) == {"src/foo.cpp"}
    # +11,3 -> (11, 13); the pure-deletion hunk (+34,0) is skipped
    assert changed["src/foo.cpp"].line_ranges == [(11, 13)]


def test_get_diff_changed_files_skips_missing_files(monkeypatch):
    monkeypatch.setattr(ctc, "_git", lambda git_args, cwd=None: _DIFF)
    monkeypatch.setattr(ctc.os.path, "isfile", lambda p: False)
    assert ctc.get_diff_changed_files("/repo", None) == {}


def test_get_diff_changed_files_single_line_hunk(monkeypatch):
    # A one-line hunk has no count (`+7` not `+7,3`), exercising the
    # `int(group(3) or 1)` default -> range (7, 7).
    diff = (
        "diff --git a/x.cpp b/x.cpp\n"
        "--- a/x.cpp\n"
        "+++ b/x.cpp\n"
        "@@ -5 +7 @@\n"
        "+one line\n"
    )
    monkeypatch.setattr(ctc, "_git", lambda git_args, cwd=None: diff)
    monkeypatch.setattr(ctc.os.path, "isfile", lambda p: True)
    changed = ctc.get_diff_changed_files("/repo", None)
    assert changed["x.cpp"].line_ranges == [(7, 7)]


# --------------------------------------------------------------------------- #
# get_untracked_changed_files
# --------------------------------------------------------------------------- #
def test_get_untracked_changed_files(tmp_path, monkeypatch):
    (tmp_path / "a.cpp").write_text("l1\nl2\nl3\n")
    (tmp_path / "empty.cpp").write_text("")
    monkeypatch.setattr(
        ctc, "_git", lambda git_args, cwd=None: "a.cpp\nempty.cpp\nnotes.txt\n"
    )
    untracked = ctc.get_untracked_changed_files(str(tmp_path))
    # empty.cpp has no lines; notes.txt is filtered by extension
    assert set(untracked) == {"a.cpp"}
    assert untracked["a.cpp"].line_ranges == [(1, 3)]


# --------------------------------------------------------------------------- #
# get_changed_files  (merge of diff + untracked)
# --------------------------------------------------------------------------- #
def test_get_changed_files_untracked_overrides_diff_on_same_path(monkeypatch):
    diff_version = ctc.ChangedFile("a.cpp", [(1, 2)])
    untracked_version = ctc.ChangedFile("a.cpp", [(1, 99)])
    other = ctc.ChangedFile("b.cpp", [(3, 4)])
    monkeypatch.setattr(
        ctc,
        "get_diff_changed_files",
        lambda repo, base: {"a.cpp": diff_version, "b.cpp": other},
    )
    monkeypatch.setattr(
        ctc, "get_untracked_changed_files", lambda repo: {"a.cpp": untracked_version}
    )
    result = {cf.path: cf for cf in ctc.get_changed_files("/repo", None)}
    assert set(result) == {"a.cpp", "b.cpp"}
    # untracked entry wins for the shared path (dict.update override semantic)
    assert result["a.cpp"] is untracked_version
    assert result["b.cpp"] is other


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

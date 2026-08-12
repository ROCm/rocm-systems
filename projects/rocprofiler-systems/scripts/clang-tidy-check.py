#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run clang-tidy on changed lines and summarize the results.

Covers committed changes since --base (default: HEAD), staged/unstaged
changes on top, and new untracked files. Runs clang-tidy scoped to only the
changed lines of each affected file, then prints which files/lines were
checked and which clang-tidy checks fired (with locations and messages).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

SOURCE_EXTENSION_RE = re.compile(
    r".*\.(cpp|cc|c\+\+|cxx|c|cl|h|hpp|m|mm|inc)$", re.IGNORECASE
)
DIFF_FILE_RE = re.compile(r'^\+\+\+\ "?b/([^ \t\n"]*)')
DIFF_HUNK_RE = re.compile(r"^@@.*\+(\d+)(,(\d+))?")
DIAGNOSTIC_RE = re.compile(r"^(.*):(\d+):(\d+): (warning|error): (.*) \[([\w,.\-]+)\]$")

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "cyan": "\033[36m",
}


def _color(text: str, *codes: str) -> str:
    """Wrap `text` in ANSI codes, unless NO_COLOR is set or the terminal is dumb."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return text
    prefix = "".join(_ANSI[c] for c in codes if c in _ANSI)
    return f"{prefix}{text}{_ANSI['reset']}"


@dataclass
class ChangedFile:
    """A source file with the line ranges touched by the local diff."""

    path: str
    line_ranges: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Diagnostic:
    """A single clang-tidy diagnostic, with its location and message."""

    file: str
    line: int
    col: int
    severity: str
    message: str
    checks: list[str]


@dataclass
class FileResult:
    """clang-tidy outcome for a single file."""

    changed_file: ChangedFile
    diagnostics: list[Diagnostic]
    succeeded: bool


@dataclass
class AggregatedDiagnostics:
    """Diagnostics bucketed by whether they fall inside the changed lines."""

    in_diff: dict[str, list[Diagnostic]]
    preexisting: dict[str, list[Diagnostic]]
    any_timed_out: bool


def in_line_ranges(line: int, ranges: list[tuple[int, int]]) -> bool:
    """Return True if `line` falls within any of the given (start, end) ranges."""
    return any(start <= line <= end for start, end in ranges)


def _git(git_args: list[str], cwd: str | None = None) -> str:
    """Run a git command, returning stdout; exit with git's own error on failure.

    Surfaces git's stderr (e.g. the "detected dubious ownership ... call
    `git config --global --add safe.directory <path>`" hint) instead of an
    opaque CalledProcessError traceback.
    """
    try:
        result = subprocess.run(
            ["git", *git_args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        sys.exit("error: 'git' not found on PATH")
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or f"git {' '.join(git_args)} failed"
        sys.exit(f"error: {message}")
    return result.stdout


def get_repo_root() -> str:
    """Return the top-level directory of the current git repository."""
    return _git(["rev-parse", "--show-toplevel"]).strip()


def get_diff_changed_files(repo_root: str, base: str | None) -> dict[str, ChangedFile]:
    """Parse a git diff into per-file changed line ranges.

    Diffs `base` (or HEAD, by default) against the working tree, so the
    result covers committed changes since `base` plus any staged/unstaged
    changes on top, in one pass.
    """
    ref = base or "HEAD"
    diff = _git(["diff", "--no-color", "-U0", ref], cwd=repo_root)

    changed: dict[str, ChangedFile] = {}
    current_path: str | None = None
    for line in diff.splitlines():
        file_match = DIFF_FILE_RE.match(line)
        if file_match:
            path = file_match.group(1)
            current_path = path if SOURCE_EXTENSION_RE.match(path) else None
            continue

        if current_path is None:
            continue

        hunk_match = DIFF_HUNK_RE.match(line)
        if not hunk_match:
            continue

        start_line = int(hunk_match.group(1))
        line_count = int(hunk_match.group(3) or 1)
        if line_count == 0:
            continue

        full_path = os.path.join(repo_root, current_path)
        if not os.path.isfile(full_path):
            continue

        changed.setdefault(current_path, ChangedFile(current_path))
        end_line = start_line + line_count - 1
        changed[current_path].line_ranges.append((start_line, end_line))

    return changed


def get_untracked_changed_files(repo_root: str) -> dict[str, ChangedFile]:
    """Return new (untracked) source files, each spanning its full line range."""
    paths = _git(
        ["ls-files", "--others", "--exclude-standard"], cwd=repo_root
    ).splitlines()

    untracked: dict[str, ChangedFile] = {}
    for path in paths:
        if not SOURCE_EXTENSION_RE.match(path):
            continue
        full_path = os.path.join(repo_root, path)
        if not os.path.isfile(full_path):
            continue
        with open(full_path, encoding="utf-8", errors="ignore") as source_file:
            line_count = sum(1 for _ in source_file)
        if line_count == 0:
            continue
        untracked[path] = ChangedFile(path, [(1, line_count)])
    return untracked


def get_changed_files(repo_root: str, base: str | None) -> list[ChangedFile]:
    """Return changed files: committed/staged/unstaged diff plus new files."""
    changed = get_diff_changed_files(repo_root, base)
    changed.update(get_untracked_changed_files(repo_root))
    return list(changed.values())


def _tidy_args(args: argparse.Namespace) -> list[str]:
    """Return the clang-tidy flags common to every invocation."""
    flags = [f"-checks={args.checks}"] if args.checks else []
    return [*flags, f"-p={args.build_path}"]


def get_enabled_checks(args: argparse.Namespace, sample_file: str) -> list[str]:
    """Resolve the effective check list clang-tidy will apply to `sample_file`."""
    command = [args.clang_tidy_binary, "-list-checks", *_tidy_args(args), sample_file]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError:
        sys.exit(f"error: clang-tidy binary '{args.clang_tidy_binary}' not found")
    if result.returncode != 0:
        detail = result.stderr.strip() or "clang-tidy -list-checks failed"
        print(
            _color(f"warning: could not resolve check list: {detail}", "yellow"),
            file=sys.stderr,
        )
        return []
    checks = []
    in_list = False
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped == "Enabled checks:":
            in_list = True
            continue
        if in_list:
            if not stripped:
                break
            checks.append(stripped)
    return checks


def print_enabled_checks(checks: list[str]) -> None:
    print(_color(f"Rules to apply ({len(checks)}):", "bold"))
    for check in checks:
        print(f"  {_color(check, 'cyan')}")
    print()


def print_changed_files(changed_files: list[ChangedFile]) -> None:
    print(_color(f"Changed files ({len(changed_files)}):", "bold"))
    for changed_file in changed_files:
        ranges = ", ".join(f"{start}-{end}" for start, end in changed_file.line_ranges)
        print(f"  {changed_file.path} (lines {ranges})")
    print()


def build_command(file_path: str, args: argparse.Namespace) -> list[str]:
    """Build the clang-tidy command that checks `file_path`."""
    command = [args.clang_tidy_binary, *_tidy_args(args), file_path]
    return command


def run_clang_tidy(command: list[str], timeout: float | None) -> tuple[str, bool]:
    """Run a clang-tidy command, returning its output and whether it completed.

    clang-tidy's own exit code reflects diagnostics anywhere in the file
    (including pre-existing issues outside the diff), so it is not a
    reliable success signal here; only a timeout counts as a run failure.
    """
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout}s: {' '.join(command)}\n", False
    return proc.stdout + proc.stderr, True


def parse_diagnostics(output: str, repo_root: str) -> list[Diagnostic]:
    """Parse every clang-tidy diagnostic out of `output` for one file."""
    diagnostics = []
    for line in output.splitlines():
        match = DIAGNOSTIC_RE.match(line)
        if not match:
            continue
        file_path, line_no, col_no, severity, message, checks = match.groups()
        diagnostics.append(
            Diagnostic(
                file=os.path.relpath(file_path, repo_root),
                line=int(line_no),
                col=int(col_no),
                severity=severity,
                message=message,
                checks=checks.split(","),
            )
        )
    return diagnostics


def print_rule_diagnostics(
    title: str, rule_diagnostics: dict[str, list[Diagnostic]]
) -> None:
    print(_color(title, "bold"))
    if not rule_diagnostics:
        print(f"  {_color('No issues found.', 'green')}")
        return
    for check_name, diagnostics in sorted(
        rule_diagnostics.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        print(f"  {_color(f'{len(diagnostics):>4}', 'yellow')}  {check_name}")
        for diagnostic in diagnostics:
            location = f"{diagnostic.file}:{diagnostic.line}:{diagnostic.col}"
            color = "red" if diagnostic.severity == "error" else "yellow"
            print(f"        {_color(location, color)}: {diagnostic.message}")
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-path",
        required=True,
        help="Directory containing compile_commands.json",
    )
    parser.add_argument(
        "--clang-tidy-binary",
        default="clang-tidy",
        help="Path to the clang-tidy binary (default: clang-tidy)",
    )
    parser.add_argument(
        "--checks",
        default="",
        help="Checks filter override (default: use .clang-tidy)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel clang-tidy instances",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-file timeout in seconds (default: no timeout)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help=(
            "Diff against this ref instead of HEAD, e.g. origin/develop. "
            "Covers committed changes since the ref plus any "
            "staged/unstaged changes on top."
        ),
    )
    return parser.parse_args()


def check_file(
    changed_file: ChangedFile, args: argparse.Namespace, repo_root: str
) -> FileResult:
    """Run clang-tidy on one file and parse its diagnostics."""
    command = build_command(os.path.join(repo_root, changed_file.path), args)
    output, succeeded = run_clang_tidy(command, args.timeout)
    return FileResult(changed_file, parse_diagnostics(output, repo_root), succeeded)


def report_progress(completed: int, total: int, changed_file: ChangedFile) -> None:
    """Print a per-file completion line to stderr."""
    print(
        _color(f"  [{completed}/{total}] {changed_file.path}", "cyan"),
        file=sys.stderr,
    )


def run_checks(
    changed_files: list[ChangedFile],
    args: argparse.Namespace,
    repo_root: str,
    on_complete: Callable[[int, int, ChangedFile], None] | None = None,
) -> list[FileResult]:
    """Run clang-tidy over all files in parallel, calling on_complete as each finishes."""
    total = len(changed_files)
    results: list[FileResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(check_file, changed_file, args, repo_root): changed_file
            for changed_file in changed_files
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            result = future.result()
            if on_complete is not None:
                on_complete(completed, total, result.changed_file)
            results.append(result)
    return results


def aggregate_diagnostics(results: list[FileResult]) -> AggregatedDiagnostics:
    """Bucket diagnostics into in-diff vs pre-existing, keyed by check name."""
    in_diff: dict[str, list[Diagnostic]] = {}
    preexisting: dict[str, list[Diagnostic]] = {}
    any_timed_out = False
    for result in results:
        any_timed_out = any_timed_out or not result.succeeded
        for diagnostic in result.diagnostics:
            target = (
                in_diff
                if in_line_ranges(diagnostic.line, result.changed_file.line_ranges)
                else preexisting
            )
            for check_name in diagnostic.checks:
                target.setdefault(check_name, []).append(diagnostic)
    return AggregatedDiagnostics(in_diff, preexisting, any_timed_out)


def main() -> int:
    """Run clang-tidy over changed files and print the results; return an exit code."""
    args = parse_args()
    repo_root = get_repo_root()

    changed_files = get_changed_files(repo_root, args.base)
    if not changed_files:
        print("No relevant changes found.")
        return 0

    sample_file = os.path.join(repo_root, changed_files[0].path)
    print_enabled_checks(get_enabled_checks(args, sample_file))
    print_changed_files(changed_files)

    print("Running clang-tidy ...")
    results = run_checks(changed_files, args, repo_root, on_complete=report_progress)
    aggregated = aggregate_diagnostics(results)

    print_rule_diagnostics(
        "Detected clang-tidy rules (in this diff):", aggregated.in_diff
    )
    print_rule_diagnostics(
        "Pre-existing issues (outside this diff):", aggregated.preexisting
    )

    return 1 if aggregated.in_diff or aggregated.any_timed_out else 0


if __name__ == "__main__":
    sys.exit(main())

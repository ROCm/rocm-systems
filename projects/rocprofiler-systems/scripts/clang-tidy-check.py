#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run clang-tidy on changed lines and summarize the results.

Covers committed changes since --base (default: HEAD), staged/unstaged
changes on top, and new untracked files. Runs clang-tidy scoped to only the
changed lines of each affected file, then prints which files/lines were
checked and which clang-tidy checks fired (with locations and messages).
"""

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
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
}


def _color(text: str, *codes: str) -> str:
    """Color `text` with ANSI codes.

    Skips coloring when NO_COLOR is set or the terminal can't render them
    (TERM=dumb).
    """
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


def in_line_ranges(line: int, ranges: list[tuple[int, int]]) -> bool:
    """Return True if `line` falls within any of the given (start, end) ranges."""
    return any(start <= line <= end for start, end in ranges)


def classify_diagnostic(diagnostic: Diagnostic, changed_file: ChangedFile) -> str | None:
    """Classify `diagnostic` as belonging to `changed_file`, or not at all.

    clang-tidy analyzes the whole translation unit, so scanning one file can
    also emit diagnostics located in headers it includes (per
    HeaderFilterRegex in .clang-tidy). Those are a different file's concern
    and are picked up when that file is scanned instead, so only diagnostics
    actually located in `changed_file.path` are classified here.

    Returns "in_diff", "preexisting", or None if the diagnostic is in a
    different file.
    """
    if os.path.normpath(diagnostic.file) != os.path.normpath(changed_file.path):
        return None
    return (
        "in_diff"
        if in_line_ranges(diagnostic.line, changed_file.line_ranges)
        else "preexisting"
    )


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
            timeout=60,
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


def _clang_tidy(
    args: argparse.Namespace, *extra: str, timeout: float | None = None
) -> subprocess.CompletedProcess:
    """Run clang-tidy with the shared `-checks`/`-p` flags plus `extra`.

    clang-tidy's exit code reflects diagnostics anywhere in the file (not just
    the diff), so it is not a reliable success signal; callers inspect the
    captured output instead. Propagates TimeoutExpired when `timeout` elapses,
    and exits if the clang-tidy binary is missing.
    """
    command = [args.clang_tidy_binary]
    if args.checks:
        command.append(f"-checks={args.checks}")
    command.append(f"-p={args.build_path}")
    command.extend(extra)
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        sys.exit(f"error: clang-tidy binary '{args.clang_tidy_binary}' not found")


def run_clang_tidy(
    args: argparse.Namespace, file_path: str, timeout: float | None
) -> tuple[str, bool]:
    """Run clang-tidy on `file_path` for the diff scan; return (output, ok).

    Adapts `_clang_tidy` for the parallel loop: merges stdout+stderr (clang-tidy
    writes diagnostics to both) and turns a timeout into `ok=False` instead of a
    raised exception. clang-tidy's exit code reflects diagnostics anywhere in the
    file, not just the diff, so it is not a reliable success signal; only a
    timeout counts as a run failure.
    """
    try:
        proc = _clang_tidy(args, file_path, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout}s on {file_path}\n", False
    return proc.stdout + proc.stderr, True


def has_enabled_checks(args: argparse.Namespace) -> bool:
    """Report whether clang-tidy will apply at least one check."""
    result = _clang_tidy(args, "-list-checks")

    if result.returncode != 0:
        detail = result.stderr.strip() or "clang-tidy -list-checks failed"
        print(
            _color(f"warning: could not resolve check list: {detail}", "yellow"),
            file=sys.stderr,
        )
        return False

    in_list = False
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped == "Enabled checks:":
            in_list = True
            continue
        if in_list and stripped:
            return True
    return False


def print_changed_files(changed_files: list[ChangedFile]) -> None:
    print(_color(f"Changed files ({len(changed_files)}):", "bold"))
    for changed_file in changed_files:
        ranges = ", ".join(f"{start}-{end}" for start, end in changed_file.line_ranges)
        print(f"  {changed_file.path} (lines {ranges})")
    print()


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
        print()
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


def print_timed_out_files(paths: list[str], timeout: float | None) -> None:
    """Report files clang-tidy could not finish, which make the run fail."""
    if not paths:
        return
    print(_color(f"Timed out after {timeout}s ({len(paths)}):", "bold"))
    for path in paths:
        print(f"  {_color(path, 'red')}")
    print(
        "  These files were not fully checked, so the run is reported as "
        "failed; raise --timeout to give them more time."
    )
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


def main() -> int:
    """Run clang-tidy over changed files and print the results; return an exit code."""
    args = parse_args()
    repo_root = get_repo_root()

    changed_files = get_changed_files(repo_root, args.base)
    if not changed_files:
        print("No relevant changes found.")
        return 0

    if not has_enabled_checks(args):
        print("No checks enabled.")
        return 0

    print_changed_files(changed_files)

    print(f"Running clang-tidy on {len(changed_files)} files ...")

    rule_diagnostics: dict[str, list[Diagnostic]] = {}
    preexisting_rule_diagnostics: dict[str, list[Diagnostic]] = {}
    timed_out_files: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_clang_tidy,
                args,
                os.path.join(repo_root, changed_file.path),
                args.timeout,
            ): changed_file
            for changed_file in changed_files
        }
        for future in concurrent.futures.as_completed(futures):
            changed_file = futures[future]
            output, succeeded = future.result()
            if not succeeded:
                timed_out_files.append(changed_file.path)

            for diagnostic in parse_diagnostics(output, repo_root):
                classification = classify_diagnostic(diagnostic, changed_file)
                if classification is None:
                    continue
                target = (
                    rule_diagnostics
                    if classification == "in_diff"
                    else preexisting_rule_diagnostics
                )
                for check_name in diagnostic.checks:
                    target.setdefault(check_name, []).append(diagnostic)

    print_rule_diagnostics("Detected clang-tidy rules (in this diff):", rule_diagnostics)
    print_rule_diagnostics(
        "Pre-existing issues (outside this diff):", preexisting_rule_diagnostics
    )
    print_timed_out_files(timed_out_files, args.timeout)

    return 1 if rule_diagnostics or timed_out_files else 0


if __name__ == "__main__":
    sys.exit(main())

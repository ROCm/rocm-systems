#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply clang-tidy fixes within --dir or --file, one check at a time.

Requires Python 3.10+, PyYAML (python -m pip install PyYAML), and clang-tidy.
-p/--build-path must contain compile_commands.json. Run from any directory;
relative CLI paths are relative to the current working directory.

Only explicit check names are accepted, in comma-separated execution order.
Existing .clang-tidy check options are honored; its enabled-check selection and
warnings-as-errors are overridden. Fixes are exported, validated, deduplicated,
and applied after parallel analysis; clang-tidy never receives -fix.

--diff-only uses git merge-base HEAD BASE, including committed PR changes,
staged/unstaged changes, and untracked files. BASE is --base, or the upstream of
the current branch (which must be the PR target, not the remote feature branch).
The initial changed-line scope is tracked through edits across check passes.
Pure deletions introduce no eligible lines. This is line scoping, not proof
that a warning was introduced by the PR.

A .cpp/.c/etc. --file is analyzed directly using its compilation database entry.
For --dir, only database translation units whose resolved source paths are
inside that directory are scheduled. In-scope headers are checked through those
translation units; headers with no selected includer are not checked. A header
--file still uses all database translation units to discover its includers.
Header diagnostics are filtered to the selected directory or exact file.
Included external headers must still be parsed; compiler errors in them can
still surface. Edits remain restricted to the selected scope. Ignored/generated/
vendor trees located INSIDE the selected directory are not implicitly excluded.
Symlink files and symlink escapes outside a selected directory are not edited.

Multi-edit diagnostics are all-or-nothing for scope/conflict validation.
Alternative fixes attached only to notes are not selected automatically.
Byte-identical replacements from different translation units are deduplicated.
A failing analysis aborts the current pass without applying its fixes; successful
earlier passes remain on disk. No commits or automatic rollback are performed.

Exit codes: 0 = completed (possibly with unfixable/skipped diagnostics),
1 = invocation/analysis/application failure, 2 = invalid CLI arguments.
--dry-run analyzes only the unchanged input, so later-pass previews may differ
from an actual sequential run. Rebuild and test after applying fixes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:
    import yaml
except ImportError:
    yaml = None

SOURCE_SUFFIXES = {
    ".cpp",
    ".cc",
    ".c++",
    ".cxx",
    ".c",
    ".cl",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".h++",
    ".m",
    ".mm",
    ".inc",
    ".inl",
    ".ipp",
    ".tpp",
}
TU_SUFFIXES = {".cpp", ".cc", ".c++", ".cxx", ".c", ".cl", ".m", ".mm"}
CHECK_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


class FixError(Exception):
    """An actionable error suitable for printing without a traceback."""


@dataclass(frozen=True, order=True)
class Replacement:
    path: Path
    offset: int
    length: int
    text: bytes


@dataclass
class FileState:
    data: bytes
    # Half-open byte spans originally covered by changed lines, mapped after edits.
    allowed: list[tuple[int, int]]


@dataclass
class RunResult:
    source: Path
    directory: Path
    diagnostics: list[dict]
    output: str
    error: str = ""


def run(
    command: list[str], *, cwd: Path | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess:
    """Run without a shell and retain diagnostics, including compiler failures."""
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FixError(f"timed out after {timeout}s: {command[0]}") from exc
    except OSError as exc:
        raise FixError(f"could not run {command[0]}: {exc}") from exc


def git(directory: Path, *args: str) -> bytes:
    """Use NUL-delimited Git paths where requested; do not parse patch filenames."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FixError(f"git failed: {exc}") from exc
    if proc.returncode:
        raise FixError(proc.stderr.decode("utf-8", "replace").strip() or "git failed")
    return proc.stdout


def canonical(raw: str, directory: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else directory / path).resolve()


def is_excluded(path: Path, excluded_dirs: list[Path]) -> bool:
    return any(path.is_relative_to(excluded) for excluded in excluded_dirs)


def collect_files(
    target: Path, is_directory: bool, excluded_dirs: list[Path]
) -> dict[Path, FileState]:
    """Snapshot only regular source files belonging to the requested scope."""
    files = target.rglob("*") if is_directory else [target]
    states = {}
    for file in files:
        if (
            file.is_symlink()
            or not file.is_file()
            or file.suffix.lower() not in SOURCE_SUFFIXES
        ):
            continue
        path = file.resolve()
        if is_directory and not path.is_relative_to(target):
            continue
        if is_excluded(path, excluded_dirs):
            continue
        data = path.read_bytes()
        states[path] = FileState(data, [(0, len(data))])
    return states


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def changed_spans(data: bytes, patch: str) -> list[tuple[int, int]]:
    """Convert Git's new-file line ranges to byte spans (LF, including CRLF)."""
    starts = [0] + [i + 1 for i, byte in enumerate(data) if byte == 10]
    spans = []
    for match in HUNK_RE.finditer(patch):
        first, count = int(match[1]), int(match[2] if match[2] is not None else 1)
        if not count or first < 1 or first > len(starts):
            continue
        start = starts[first - 1]
        end_index = first - 1 + count
        end = starts[end_index] if end_index < len(starts) else len(data)
        if start < end:
            spans.append((start, end))
    return merge_spans(spans)


def restrict_to_diff(
    states: dict[Path, FileState], target: Path, base: str | None
) -> dict[Path, FileState]:
    directory = target if target.is_dir() else target.parent
    root = Path(
        os.fsdecode(git(directory, "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    if base is None:
        try:
            base = os.fsdecode(
                git(root, "rev-parse", "--abbrev-ref", "@{upstream}")
            ).strip()
        except FixError as exc:
            raise FixError(
                "--diff-only needs --base <PR-target-branch> when no upstream is configured"
            ) from exc
        print(
            f"Using upstream {base!r}; ensure this is the PR target, otherwise pass --base."
        )
    # Resolve user input as a revision before passing it to merge-base.
    revision = os.fsdecode(
        git(root, "rev-parse", "--verify", "--end-of-options", base + "^{commit}")
    ).strip()
    merge_base = os.fsdecode(git(root, "merge-base", "HEAD", revision)).strip()
    print(
        f"Diff scope: merge-base(HEAD, {base}) = {merge_base[:12]} through working tree"
    )
    dirty = {
        canonical(os.fsdecode(p), root)
        for p in git(
            root,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-only",
            "-z",
            "--no-renames",
            merge_base,
            "--",
        ).split(b"\0")
        if p
    }
    untracked = {
        canonical(os.fsdecode(p), root)
        for p in git(root, "ls-files", "--others", "--exclude-standard", "-z").split(
            b"\0"
        )
        if p
    }
    selected = {}
    for path, state in states.items():
        if not path.is_relative_to(root):
            continue
        if path in untracked:
            state.allowed = [(0, len(state.data))] if state.data else []
        elif path in dirty:
            patch = git(
                root,
                "--literal-pathspecs",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
                "--no-renames",
                "--unified=0",
                merge_base,
                "--",
                path.relative_to(root).as_posix(),
            )
            state.allowed = changed_spans(state.data, patch.decode("utf-8", "replace"))
        else:
            continue
        if state.allowed:
            selected[path] = state
    return selected


def load_units(
    build_path: Path, target: Path, is_directory: bool, excluded_dirs: list[Path]
) -> list[tuple[Path, Path]]:
    database = build_path / "compile_commands.json"
    try:
        entries = json.loads(database.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FixError(f"cannot read {database}: {exc}") from exc
    if not isinstance(entries, list):
        raise FixError("compile_commands.json must contain an array")
    units = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("file"), str)
            or not isinstance(entry.get("directory"), str)
        ):
            raise FixError(
                "invalid compilation database entry: file and directory must be strings"
            )
        directory = canonical(entry["directory"], build_path)
        path = canonical(entry["file"], directory)
        # Filter by the canonical SOURCE path, not the build working directory.
        # Path containment avoids matching siblings such as source-extra, and
        # resolving first excludes source/link.cpp -> ../externals/link.cpp.
        if is_directory and not path.is_relative_to(target):
            continue
        if not is_directory and target.suffix.lower() in TU_SUFFIXES and path != target:
            continue
        if is_excluded(path, excluded_dirs):
            continue
        units.setdefault(path, directory)
    if not units:
        raise FixError(
            "no matching translation units in compile_commands.json; regenerate the database if needed"
        )
    for path, directory in units.items():
        if not path.is_file() or not directory.is_dir():
            raise FixError(
                f"stale compilation database entry: {path} (directory {directory})"
            )
    return sorted(units.items())


def scope_header_filter(target: Path, is_directory: bool) -> str:
    """Build an anchored LLVM-compatible regex, with directory boundaries."""
    # Escape ERE metacharacters rather than using Python-only regex constructs.
    escaped = re.sub(r"([.\[\]{}()*+?^$|\\])", r"\\\1", target.as_posix())
    escaped = escaped.replace("/", r"[/\\]")
    if is_directory:
        return "^" + escaped.rstrip("/") + ("" if target == target.parent else r"[/\\]")
    return "^" + escaped + "$"


def analyze(
    args: argparse.Namespace, check: str, source: Path, directory: Path, export: Path
) -> RunResult:
    target = (args.dir if args.dir is not None else args.file).resolve()
    header_filter = scope_header_filter(target, args.dir is not None)
    command = [
        args.clang_tidy_binary,
        str(source),
        f"-p={args.build_path}",
        f"--checks=-*,{check}",
        "--warnings-as-errors=",
        f"--header-filter={header_filter}",
        "--use-color=false",
        f"--export-fixes={export}",
    ]
    try:
        proc = run(command, cwd=directory, timeout=args.timeout)
        if proc.returncode:
            return RunResult(
                source,
                directory,
                [],
                proc.stdout,
                f"clang-tidy exited with status {proc.returncode}",
            )
        document = (
            yaml.safe_load(export.read_text(encoding="utf-8")) if export.exists() else {}
        )
        if document is None:
            document = {}
        if not isinstance(document, dict) or not isinstance(
            document.get("Diagnostics", []), list
        ):
            raise FixError("invalid exported diagnostic document")
        diagnostics = document.get("Diagnostics", [])
        if any(not isinstance(item, dict) for item in diagnostics):
            raise FixError("invalid exported diagnostic entry")
        if any(
            item.get("Level") in {"Error", "Fatal"}
            or item.get("DiagnosticName") == "clang-diagnostic-error"
            for item in diagnostics
        ):
            return RunResult(
                source, directory, [], proc.stdout, "compiler errors in translation unit"
            )
        return RunResult(source, directory, diagnostics, proc.stdout)
    except (FixError, OSError, ValueError, yaml.YAMLError) as exc:
        return RunResult(source, directory, [], "", str(exc))


def decode_replacement(item: dict, directory: Path) -> Replacement:
    if not isinstance(item, dict):
        raise FixError("malformed exported replacement")
    path, offset, length, text = (
        item.get(k) for k in ("FilePath", "Offset", "Length", "ReplacementText")
    )
    if (
        not isinstance(path, str)
        or not path
        or type(offset) is not int
        or type(length) is not int
        or not isinstance(text, str)
    ):
        raise FixError("malformed exported replacement fields")
    if offset < 0 or length < 0:
        raise FixError("negative replacement offset or length")
    return Replacement(canonical(path, directory), offset, length, text.encode("utf-8"))


def in_scope(edit: Replacement, states: dict[Path, FileState], diff_only: bool) -> bool:
    state = states.get(edit.path)
    if state is None or edit.offset + edit.length > len(state.data):
        return False
    if not diff_only:
        return True
    for start, end in state.allowed:
        if edit.length and start <= edit.offset and edit.offset + edit.length <= end:
            return True
        if not edit.length and start <= edit.offset < end:
            return True
        # At EOF without LF, insertion is still on the final changed line.
        if (
            not edit.length
            and edit.offset == end == len(state.data)
            and start < end
            and not state.data.endswith(b"\n")
        ):
            return True
    return False


def overlap(left: Replacement, right: Replacement) -> bool:
    """Conservatively conflict on insertion/deletion boundaries too."""
    if left == right or left.path != right.path:
        return False
    if not left.length or not right.length:
        return max(left.offset, right.offset) <= min(
            left.offset + left.length, right.offset + right.length
        )
    return max(left.offset, right.offset) < min(
        left.offset + left.length, right.offset + right.length
    )


def select_edits(
    results: list[RunResult], check: str, states: dict[Path, FileState], diff_only: bool
) -> tuple[list[Replacement], dict[str, int]]:
    """Reject entire diagnostics with any out-of-scope or conflicting edits."""
    counts = {
        "diagnostics": 0,
        "no_fix": 0,
        "outside_scope": 0,
        "duplicate_groups": 0,
        "conflicting_groups": 0,
        "accepted_groups": 0,
    }
    groups: list[tuple[Replacement, ...]] = []
    seen = set()
    for result in results:
        for diagnostic in result.diagnostics:
            if check not in diagnostic.get("DiagnosticName", "").split(","):
                continue
            counts["diagnostics"] += 1
            message = diagnostic.get("DiagnosticMessage", {})
            if not isinstance(message, dict):
                raise FixError("malformed DiagnosticMessage")
            raw = message.get("Replacements", [])
            if not isinstance(raw, list):
                raise FixError("malformed Replacements list")
            # Never combine competing suggestions from diagnostic notes.
            if not raw:
                counts["no_fix"] += 1
                continue
            build_directory = diagnostic.get("BuildDirectory")
            directory = (
                canonical(build_directory, result.directory)
                if build_directory
                else result.directory
            )
            edits = tuple(
                sorted(set(decode_replacement(item, directory) for item in raw))
            )
            if not all(in_scope(edit, states, diff_only) for edit in edits):
                counts["outside_scope"] += 1
                continue
            if edits in seen:
                counts["duplicate_groups"] += 1
                continue
            seen.add(edits)
            groups.append(edits)
    # Sweep per file; mark BOTH diagnostic groups on a real overlap. Identical
    # replacements shared by otherwise different groups are safe to deduplicate.
    by_file: dict[Path, list[tuple[Replacement, int]]] = {}
    for group_id, edits in enumerate(groups):
        for edit in edits:
            by_file.setdefault(edit.path, []).append((edit, group_id))
    rejected = set()
    for entries in by_file.values():
        active: list[tuple[Replacement, int]] = []
        for edit, group_id in sorted(entries):
            active = [
                (old, owner)
                for old, owner in active
                if old.offset + old.length >= edit.offset
            ]
            for old, owner in active:
                if overlap(old, edit):
                    rejected.update((owner, group_id))
            active.append((edit, group_id))
    counts["conflicting_groups"] = len(rejected)
    counts["accepted_groups"] = len(groups) - len(rejected)
    edits = sorted(
        {
            edit
            for index, group in enumerate(groups)
            if index not in rejected
            for edit in group
        }
    )
    return edits, counts


def transformed(state: FileState, edits: list[Replacement]) -> FileState:
    """Apply nonoverlapping byte replacements and carry diff spans forward."""
    ordered = sorted(edits, key=lambda edit: edit.offset)
    parts, cursor = [], 0
    for edit in ordered:
        parts.extend((state.data[cursor : edit.offset], edit.text))
        cursor = edit.offset + edit.length
    parts.append(state.data[cursor:])
    spans = []
    for start, end in state.allowed:
        before = sum(
            len(edit.text) - edit.length for edit in ordered if edit.offset < start
        )
        inside = sum(
            len(edit.text) - edit.length
            for edit in ordered
            if start <= edit.offset < end
            or (edit.offset == end == len(state.data) and not edit.length)
        )
        if start + before < end + before + inside:
            spans.append((start + before, end + before + inside))
    return FileState(b"".join(parts), spans)


def ensure_unchanged(path: Path, expected: bytes) -> None:
    """Avoid overwriting edits made during analysis or following a moved link."""
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise FixError(f"file disappeared or became a symlink: {path}")
    if path.stat().st_nlink > 1:
        raise FixError(f"refusing to replace a hard-linked file: {path}")
    if path.read_bytes() != expected:
        raise FixError(f"file changed during the run; no overwrite allowed: {path}")


def apply_edits(
    edits: list[Replacement], states: dict[Path, FileState], dry_run: bool
) -> int:
    by_file: dict[Path, list[Replacement]] = {}
    for edit in edits:
        by_file.setdefault(edit.path, []).append(edit)
    updated = {
        path: transformed(states[path], changes) for path, changes in by_file.items()
    }
    updated = {
        path: state for path, state in updated.items() if state.data != states[path].data
    }
    for path in updated:
        ensure_unchanged(path, states[path].data)
    for path in sorted(updated):
        print(
            f"  {'Would update' if dry_run else 'Updating'} {path} ({len(by_file[path])} replacements)"
        )
    if dry_run:
        return len(updated)
    pending: dict[Path, Path] = {}
    try:
        # Stage all outputs before replacing any originals. Each replace is
        # atomic, but the whole multi-file pass is NOT a filesystem transaction.
        for path, state in updated.items():
            descriptor, name = tempfile.mkstemp(
                prefix=".clang-tidy-fix-", dir=path.parent
            )
            pending[path] = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(state.data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(name, stat.S_IMODE(path.stat().st_mode))
        for path in pending:
            ensure_unchanged(path, states[path].data)
        for path, temporary in pending.items():
            ensure_unchanged(path, states[path].data)
            os.replace(temporary, path)
            states[path] = updated[path]
    finally:
        for temporary in pending.values():
            temporary.unlink(missing_ok=True)
    return len(updated)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def positive_float(value: str) -> float:
    import math

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--dir",
        type=Path,
        help="Analyze translation units and allow edits only inside this directory",
    )
    scope.add_argument(
        "--file", type=Path, help="Allow edits only in this exact source/header file"
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        type=Path,
        default=[],
        help="Exclude translation units and files under this directory " "(repeatable)",
    )
    parser.add_argument(
        "-p",
        "--build-path",
        required=True,
        type=Path,
        help="Build directory containing compile_commands.json",
    )
    parser.add_argument(
        "--checks",
        required=True,
        help="Comma-separated explicit check names, in execution order",
    )
    parser.add_argument(
        "--diff-only",
        action="store_true",
        help="Limit every replacement to PR-changed lines",
    )
    parser.add_argument(
        "--base",
        help="PR target ref, e.g. origin/main; otherwise use configured upstream",
    )
    parser.add_argument(
        "--clang-tidy-binary",
        default="clang-tidy",
        help="clang-tidy executable (default: clang-tidy)",
    )
    parser.add_argument("-j", "--jobs", type=positive_int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=None,
        help="Timeout in seconds per clang-tidy process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report proposed fixes without modifying sources",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print full clang-tidy output"
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip translation units whose analysis fails (e.g. compiler errors) "
        "instead of aborting the whole check pass",
    )
    args = parser.parse_args(argv)
    if args.base and not args.diff_only:
        parser.error("--base requires --diff-only")
    names = [name.strip() for name in args.checks.split(",") if name.strip()]
    if not names or any(
        not CHECK_RE.fullmatch(name) or name.startswith("clang-diagnostic-")
        for name in names
    ):
        parser.error(
            "--checks must contain explicit check names; wildcards, exclusions, and compiler diagnostics are not supported"
        )
    args.checks = list(dict.fromkeys(names))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if yaml is None:
            raise FixError(
                "PyYAML is required; install it with: python -m pip install PyYAML"
            )
        executable = shutil.which(args.clang_tidy_binary)
        if executable is None:
            raise FixError(f"clang-tidy binary not found: {args.clang_tidy_binary}")
        args.clang_tidy_binary = str(Path(executable).resolve())
        is_directory = args.dir is not None
        original_target = args.dir if is_directory else args.file
        if original_target.is_symlink():
            raise FixError("the selected target must not be a symlink")
        target = original_target.resolve()
        if not (target.is_dir() if is_directory else target.is_file()):
            raise FixError(
                f"selected {'directory' if is_directory else 'file'} does not exist: {target}"
            )
        if not is_directory and target.suffix.lower() not in SOURCE_SUFFIXES:
            raise FixError(f"unsupported source extension: {target.suffix}")
        args.build_path = args.build_path.resolve()
        excluded_dirs = []
        for excluded in args.exclude_dir:
            if excluded.is_symlink():
                raise FixError("--exclude-dir must not be a symlink")
            resolved = excluded.resolve()
            if not resolved.is_dir():
                raise FixError(f"--exclude-dir does not exist: {resolved}")
            excluded_dirs.append(resolved)
        units = load_units(args.build_path, target, is_directory, excluded_dirs)
        states = collect_files(target, is_directory, excluded_dirs)
        if args.diff_only:
            states = restrict_to_diff(states, target, args.base)
        if not states:
            print("No eligible source files or changed lines found.")
            return 0
        # Fail before any edits if the binary does not recognize a requested rule.
        listed = run(
            [args.clang_tidy_binary, "--list-checks", "--checks=*"],
            cwd=units[0][1],
            timeout=args.timeout,
        )
        if listed.returncode:
            raise FixError(f"could not list checks:\n{listed.stdout}")
        available = {
            line.strip()
            for line in listed.stdout.splitlines()
            if CHECK_RE.fullmatch(line.strip())
        }
        unknown = [name for name in args.checks if name not in available]
        if unknown:
            raise FixError(
                "checks unavailable in this clang-tidy binary: " + ", ".join(unknown)
            )
        print(f"Edit scope: {target}; {len(states)} eligible files")
        print(f"Analysis: {len(units)} translation units per check, {args.jobs} workers")
        if args.dry_run:
            print(
                "Dry run: no files will change; later passes see original, not transformed code."
            )
        total_edits = total_files = 0
        for index, check in enumerate(args.checks, 1):
            print(f"\n[{index}/{len(args.checks)}] {check}", flush=True)
            # Revalidate all editable snapshots before starting the next analysis.
            for path, state in states.items():
                ensure_unchanged(path, state.data)
            with tempfile.TemporaryDirectory(prefix="clang-tidy-fixes-") as temporary:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                    futures = [
                        pool.submit(
                            analyze,
                            args,
                            check,
                            source,
                            directory,
                            Path(temporary) / f"{number}.yaml",
                        )
                        for number, (source, directory) in enumerate(units)
                    ]
                    results = [
                        future.result()
                        for future in concurrent.futures.as_completed(futures)
                    ]
            results.sort(key=lambda item: item.source)
            failed = False
            skipped_units = 0
            for result in results:
                if result.error or args.verbose:
                    print(f"--- {result.source} ---\n{result.output.rstrip()}")
                if result.error:
                    print(f"error: {result.error}", file=sys.stderr)
                    if args.skip_errors:
                        skipped_units += 1
                    else:
                        failed = True
            if failed:
                raise FixError(
                    f"analysis failed for {check}; no fixes from this pass were applied "
                    "(use --skip-errors to skip failing translation units instead)"
                )
            if skipped_units:
                print(
                    f"  Skipping {skipped_units} translation unit(s) with analysis errors"
                )
                results = [result for result in results if not result.error]
            edits, counts = select_edits(results, check, states, args.diff_only)
            files = apply_edits(edits, states, args.dry_run)
            total_edits += len(edits)
            total_files += files
            print(
                f"  {len(edits)} accepted replacements in {files} files; "
                f"{counts['accepted_groups']} accepted diagnostic groups"
            )
            print(
                f"  Skipped: {counts['no_fix']} without primary fixes, "
                f"{counts['outside_scope']} outside scope, "
                f"{counts['conflicting_groups']} conflicting groups; "
                f"{counts['duplicate_groups']} duplicate groups deduplicated"
            )
        print(
            f"\n{'Previewed' if args.dry_run else 'Applied'} {total_edits} replacements; "
            f"{total_files} file updates across {len(args.checks)} passes."
        )
        print(
            "Completion does not imply warning-free code. Review the diff, rebuild, and run tests."
        )
        return 0
    except (FixError, OSError, ValueError) as exc:
        print(
            f"error: {exc}\nEarlier successful passes, if any, remain applied.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Earlier applied changes remain; review your working tree.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate code coverage reports using gcovr.

Runs gcovr on a build directory instrumented with --coverage flags,
producing XML (Cobertura), HTML, and Markdown reports.

Usage:
    python3 scripts/generate-coverage.py --build-dir build/coverage --source-dir .

    # Compare against a baseline:
    python3 scripts/generate-coverage.py --build-dir build/coverage --source-dir . \
        --baseline .codecov/baseline.json

    # Custom output directory:
    python3 scripts/generate-coverage.py --build-dir build/coverage --source-dir . \
        --output-dir .codecov --label all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

GCOVR_EXCLUDE_PATTERNS = [
    r"/usr/.*",
    r"/opt/.*",
    r".*external/.*",
    r".*examples/.*",
    r".*tests/.*",
    r".*/googletest/.*",
]


def find_tool(name: str, required: bool = True) -> Optional[str]:
    path = shutil.which(name)
    if required and path is None:
        print(f"ERROR: {name} not found in PATH", file=sys.stderr)
        sys.exit(1)
    return path


def scope_filter(dir_scope: Path) -> str:
    """Build a regex restricting gcovr to files under dir_scope.

    Must start with a literal "/" (no leading "^") so gcovr's FilterOption
    classifies it as an AbsoluteFilter; a leading "^" makes it a RelativeFilter
    matched against cwd-relative paths, silently filtering out every file.
    """
    escaped = re.sub(r"([.\[\]{}()*+?^$|\\])", r"\\\1", dir_scope.as_posix())
    return escaped.rstrip("/") + "/"


def in_scope(filename: str, source_dir: Path, dir_scope: Path) -> bool:
    """Re-check containment; never trust gcovr's --filter alone."""
    path = canonical(filename, source_dir)
    return path.is_relative_to(dir_scope)


def canonical(raw: str, directory: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else directory / path).resolve()


class GitError(Exception):
    """Raised when a git invocation fails."""


def git(directory: Path, *args: str) -> bytes:
    """Use NUL-delimited Git paths; do not parse patch/diff-name output."""
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
        raise GitError(f"git failed: {exc}") from exc
    if proc.returncode:
        raise GitError(proc.stderr.decode("utf-8", "replace").strip() or "git failed")
    return proc.stdout


def changed_files(source_dir: Path, base: Optional[str]) -> set[Path]:
    """Files changed vs merge-base(HEAD, base): committed, staged/unstaged, untracked."""
    root = Path(
        os.fsdecode(git(source_dir, "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    if base is None:
        try:
            base = os.fsdecode(
                git(root, "rev-parse", "--abbrev-ref", "@{upstream}")
            ).strip()
        except GitError as exc:
            raise GitError(
                "--diff-only needs --base <target-branch> "
                "when no upstream is configured"
            ) from exc
        print(f"Using upstream {base!r} as diff base; pass --base to override.")
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
    return dirty | untracked


def run_gcovr(
    *,
    gcov_cmd: str,
    source_dir: Path,
    build_dir: Path,
    output_dir: Path,
    label: str,
    dir_scope: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{label}.json"
    xml_path = output_dir / f"{label}.xml"
    html_path = output_dir / f"{label}.html"

    gcovr_cmd = [sys.executable, "-m", "gcovr"]
    cmd = [
        *gcovr_cmd,
        "--root",
        str(source_dir),
        "--gcov-executable",
        gcov_cmd,
        "--exclude-unreachable-branches",
        "--exclude-throw-branches",
        "--gcov-ignore-parse-errors",
        "--merge-lines",
        "--merge-mode-functions=merge-use-line-max",
        "-s",
        "-p",
        "--filter",
        scope_filter(dir_scope),
        "--json",
        str(json_path),
        "--xml",
        str(xml_path),
        "--html-details",
        str(html_path),
    ]

    for pattern in GCOVR_EXCLUDE_PATTERNS:
        cmd.extend(["--exclude", pattern])

    cmd.append(str(build_dir))

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"gcovr stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return json_path


def load_coverage_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_file_coverage(
    data: dict,
    *,
    source_dir: Path,
    dir_scope: Path,
    changed: Optional[set[Path]] = None,
) -> list[dict]:
    files = []
    for file_data in data.get("files", []):
        filename = file_data.get("filename", "") or file_data.get("file", "")
        if not filename:
            continue
        if not in_scope(filename, source_dir, dir_scope):
            continue
        if changed is not None and canonical(filename, source_dir) not in changed:
            continue
        lines = file_data.get("lines", [])
        if not lines:
            continue

        line_counts: dict[int, int] = {}
        for line in lines:
            ln = line.get("line_number", 0)
            count = line.get("count", 0)
            if ln not in line_counts or count > line_counts[ln]:
                line_counts[ln] = count

        covered = sum(1 for c in line_counts.values() if c > 0)
        total = len(line_counts)
        pct = (covered / total * 100) if total > 0 else 0.0

        func_by_line: dict[int, bool] = {}
        for func in file_data.get("functions", []):
            lineno = func.get("lineno", 0)
            ran = func.get("execution_count", 0) > 0
            func_by_line[lineno] = func_by_line.get(lineno, False) or ran

        func_covered = sum(1 for v in func_by_line.values() if v)
        func_total = len(func_by_line)
        func_pct = (func_covered / func_total * 100) if func_total > 0 else 0.0

        files.append(
            {
                "filename": filename,
                "covered_lines": covered,
                "total_lines": total,
                "coverage_pct": pct,
                "covered_functions": func_covered,
                "total_functions": func_total,
                "function_pct": func_pct,
            }
        )

    return sorted(files, key=lambda f: f["coverage_pct"])


def compute_totals(file_coverages: list[dict]) -> dict:
    total_covered = sum(f["covered_lines"] for f in file_coverages)
    total_lines = sum(f["total_lines"] for f in file_coverages)
    pct = (total_covered / total_lines * 100) if total_lines > 0 else 0.0

    total_func_covered = sum(f.get("covered_functions", 0) for f in file_coverages)
    total_funcs = sum(f.get("total_functions", 0) for f in file_coverages)
    func_pct = (total_func_covered / total_funcs * 100) if total_funcs > 0 else 0.0

    return {
        "covered_lines": total_covered,
        "total_lines": total_lines,
        "coverage_pct": pct,
        "file_count": len(file_coverages),
        "covered_functions": total_func_covered,
        "total_functions": total_funcs,
        "function_pct": func_pct,
    }


def coverage_bar(pct: float) -> str:
    if pct >= 80:
        icon = "🟢"
    elif pct >= 50:
        icon = "🟡"
    else:
        icon = "🔴"
    return f"{icon} **{pct:.1f}%**"


def file_delta_str(f: dict, baseline_by_filename: dict[str, float]) -> str:
    baseline_pct = baseline_by_filename.get(f["filename"])
    if baseline_pct is None:
        return ""
    delta = f["coverage_pct"] - baseline_pct
    if round(delta, 2) == 0:
        return ""
    sign = "+" if delta >= 0 else ""
    emoji = "📈" if delta >= 0 else "📉"
    return f" ({emoji} {sign}{delta:.2f}%)"


def generate_markdown(
    *,
    file_coverages: list[dict],
    totals: dict,
    baseline_totals: Optional[dict],
    baseline_files: Optional[list[dict]] = None,
    source_dir: Path,
) -> str:
    baseline_by_filename = (
        {f["filename"]: f["coverage_pct"] for f in baseline_files}
        if baseline_files
        else {}
    )
    lines = []

    delta_str = ""
    if baseline_totals:
        delta = totals["coverage_pct"] - baseline_totals["coverage_pct"]
        if round(delta, 2) != 0:
            sign = "+" if delta >= 0 else ""
            emoji = "📈" if delta >= 0 else "📉"
            delta_str = f" ({emoji} {sign}{delta:.2f}% vs base)"

    lines.append(
        f"**Lines**: {coverage_bar(totals['coverage_pct'])}{delta_str} "
        f"— {totals['covered_lines']:,}/{totals['total_lines']:,} "
        f"across {totals['file_count']} files"
    )
    lines.append(
        f"**Functions**: {coverage_bar(totals['function_pct'])} "
        f"— {totals['covered_functions']:,}/{totals['total_functions']:,}"
    )
    lines.append("")

    if file_coverages:
        groups = [
            ("🔴 0-20%", [f for f in file_coverages if f["coverage_pct"] < 20]),
            ("🟠 20-50%", [f for f in file_coverages if 20 <= f["coverage_pct"] < 50]),
            ("🟡 50-80%", [f for f in file_coverages if 50 <= f["coverage_pct"] < 80]),
            ("🟢 80-100%", [f for f in file_coverages if f["coverage_pct"] >= 80]),
        ]
        for group_label, group_files in groups:
            if not group_files:
                continue
            lines.append(f"<details>")
            lines.append(f"<summary>{group_label} ({len(group_files)} files)</summary>")
            lines.append("")
            lines.append("| Lines | Functions | File |")
            lines.append("|-------|-----------|------|")
            for f in group_files:
                rel = os.path.relpath(f["filename"], source_dir)
                func_str = (
                    f"{f['covered_functions']}/{f['total_functions']}"
                    if f.get("total_functions", 0) > 0
                    else "—"
                )
                delta_str = file_delta_str(f, baseline_by_filename)
                lines.append(
                    f"| {f['coverage_pct']:5.1f}% {f['covered_lines']}/{f['total_lines']}"
                    f"{delta_str} | "
                    f"{func_str} | "
                    f"`{rel}` |"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")

    buckets = {"0-20%": 0, "20-50%": 0, "50-80%": 0, "80-100%": 0}
    for f in file_coverages:
        p = f["coverage_pct"]
        if p < 20:
            buckets["0-20%"] += 1
        elif p < 50:
            buckets["20-50%"] += 1
        elif p < 80:
            buckets["50-80%"] += 1
        else:
            buckets["80-100%"] += 1

    lines.append("### Distribution")
    lines.append("")
    lines.append("| Range | Files |")
    lines.append("|-------|-------|")
    for bucket, count in buckets.items():
        lines.append(f"| {bucket} | {count} |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate code coverage reports")
    parser.add_argument(
        "--build-dir",
        required=True,
        type=Path,
        help="Build directory with .gcda/.gcno files",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Source directory root",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Restrict the report to files within this directory "
        "(default: --source-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports (default: <source-dir>/.codecov)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="all",
        help="Coverage report label",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline coverage JSON for delta comparison",
    )
    parser.add_argument(
        "--diff-only",
        action="store_true",
        help="Restrict the report to files changed vs --base (file-level scope)",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="Diff base ref for --diff-only, e.g. origin/develop; "
        "otherwise uses the configured upstream",
    )
    parser.add_argument(
        "--gcov",
        type=str,
        default=None,
        help="Path to gcov executable",
    )

    args = parser.parse_args()
    if args.base and not args.diff_only:
        parser.error("--base requires --diff-only")

    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()
    output_dir = (args.output_dir or source_dir / ".codecov").resolve()
    gcov_cmd = args.gcov or find_tool("gcov")

    dir_scope = (args.dir or source_dir).resolve()
    if not dir_scope.is_dir():
        print(f"ERROR: --dir does not exist: {dir_scope}", file=sys.stderr)
        sys.exit(1)
    if not dir_scope.is_relative_to(source_dir):
        print(
            f"ERROR: --dir {dir_scope} is not inside --source-dir {source_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    gitignore_path = output_dir / ".gitignore"
    if not gitignore_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        gitignore_path.write_text("/*\n")

    changed = None
    if args.diff_only:
        try:
            changed = changed_files(source_dir, args.base)
        except GitError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Diff-only: {len(changed)} changed file(s)")

    print(f"Source dir:  {source_dir}")
    print(f"Scope dir:   {dir_scope}")
    print(f"Build dir:   {build_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Label:       {args.label}")
    print(f"gcov:        {gcov_cmd}")
    print()

    json_path = run_gcovr(
        gcov_cmd=gcov_cmd,
        source_dir=source_dir,
        build_dir=build_dir,
        output_dir=output_dir,
        label=args.label,
        dir_scope=dir_scope,
    )

    data = load_coverage_json(json_path)
    file_coverages = compute_file_coverage(
        data, source_dir=source_dir, dir_scope=dir_scope, changed=changed
    )
    totals = compute_totals(file_coverages)

    baseline_totals = None
    baseline_files = None
    if args.baseline and args.baseline.exists():
        baseline_data = load_coverage_json(args.baseline)
        baseline_files = compute_file_coverage(
            baseline_data, source_dir=source_dir, dir_scope=dir_scope, changed=changed
        )
        baseline_totals = compute_totals(baseline_files)

    md = generate_markdown(
        file_coverages=file_coverages,
        totals=totals,
        baseline_totals=baseline_totals,
        baseline_files=baseline_files if args.diff_only else None,
        source_dir=source_dir,
    )

    md_path = output_dir / f"{args.label}.md"
    md_path.write_text(md)
    print(f"Wrote markdown report: {md_path}")

    summary = {
        "label": args.label,
        "coverage_pct": round(totals["coverage_pct"], 2),
        "covered_lines": totals["covered_lines"],
        "total_lines": totals["total_lines"],
        "function_pct": round(totals["function_pct"], 2),
        "covered_functions": totals["covered_functions"],
        "total_functions": totals["total_functions"],
        "file_count": totals["file_count"],
    }
    if baseline_totals:
        summary["baseline_pct"] = round(baseline_totals["coverage_pct"], 2)
        summary["delta_pct"] = round(
            totals["coverage_pct"] - baseline_totals["coverage_pct"], 2
        )

    summary_path = output_dir / f"{args.label}-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary JSON:    {summary_path}")

    print(f"\nCoverage: {totals['coverage_pct']:.2f}%")
    if baseline_totals:
        delta = totals["coverage_pct"] - baseline_totals["coverage_pct"]
        print(f"Delta:    {'+' if delta >= 0 else ''}{delta:.2f}%")


if __name__ == "__main__":
    main()

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


def run_gcovr(
    *,
    gcov_cmd: str,
    source_dir: Path,
    build_dir: Path,
    output_dir: Path,
    label: str,
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


def compute_file_coverage(data: dict) -> list[dict]:
    files = []
    for file_data in data.get("files", []):
        filename = file_data.get("filename", "") or file_data.get("file", "")
        if not filename:
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


def generate_markdown(
    *,
    label: str,
    file_coverages: list[dict],
    totals: dict,
    baseline_totals: Optional[dict],
    source_dir: Path,
) -> str:
    lines = []
    lines.append(f"## Code Coverage: {label}")
    lines.append("")

    delta_str = ""
    if baseline_totals:
        delta = totals["coverage_pct"] - baseline_totals["coverage_pct"]
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
                lines.append(
                    f"| {f['coverage_pct']:5.1f}% {f['covered_lines']}/{f['total_lines']} | "
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
        "--gcov",
        type=str,
        default=None,
        help="Path to gcov executable",
    )

    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()
    output_dir = (args.output_dir or source_dir / ".codecov").resolve()
    gcov_cmd = args.gcov or find_tool("gcov")

    gitignore_path = output_dir / ".gitignore"
    if not gitignore_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        gitignore_path.write_text("/*\n")

    print(f"Source dir:  {source_dir}")
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
    )

    data = load_coverage_json(json_path)
    file_coverages = compute_file_coverage(data)
    totals = compute_totals(file_coverages)

    baseline_totals = None
    if args.baseline and args.baseline.exists():
        baseline_data = load_coverage_json(args.baseline)
        baseline_files = compute_file_coverage(baseline_data)
        baseline_totals = compute_totals(baseline_files)

    md = generate_markdown(
        label=args.label,
        file_coverages=file_coverages,
        totals=totals,
        baseline_totals=baseline_totals,
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

#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate code coverage reports.

Supports two modes:
  gcov  -- uses gcovr (GCC --coverage builds)
  llvm  -- uses llvm-profdata + llvm-cov (Clang -fprofile-instr-generate builds)

Usage (gcov mode):
    python3 scripts/generate-coverage.py --mode gcov \
        --build-dir build/coverage --source-dir .

Usage (llvm mode):
    python3 scripts/generate-coverage.py --mode llvm \
        --profraw-dir build/coverage/profraw/unit \
        --binary-dir  build/coverage \
        --source-dir  . \
        --label unit-tests
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

EXCLUDE_PATTERNS = [
    r"/usr/.*",
    r"/opt/.*",
    r".*external/.*",
    r".*examples/.*",
    r".*tests/.*",
    r".*/googletest/.*",
]


# ---------------------------------------------------------------------------
# shared utilities
# ---------------------------------------------------------------------------


def find_tool(name: str, required: bool = True) -> Optional[str]:
    path = shutil.which(name)
    if required and path is None:
        print(f"ERROR: {name} not found in PATH", file=sys.stderr)
        sys.exit(1)
    return path


def compute_file_coverage(files_raw: list[dict]) -> list[dict]:
    files = []
    for f in files_raw:
        filename = f.get("filename", "")
        covered = f.get("covered_lines", 0)
        total = f.get("total_lines", 0)
        if not filename or total == 0:
            continue
        files.append(
            {
                "filename": filename,
                "covered_lines": covered,
                "total_lines": total,
                "coverage_pct": (covered / total * 100) if total > 0 else 0.0,
            }
        )
    return sorted(files, key=lambda f: f["coverage_pct"])


def compute_totals(file_coverages: list[dict]) -> dict:
    total_covered = sum(f["covered_lines"] for f in file_coverages)
    total_lines = sum(f["total_lines"] for f in file_coverages)
    return {
        "covered_lines": total_covered,
        "total_lines": total_lines,
        "coverage_pct": (total_covered / total_lines * 100) if total_lines > 0 else 0.0,
        "file_count": len(file_coverages),
    }


def coverage_bar(pct: float) -> str:
    icon = "🟢" if pct >= 80 else ("🟡" if pct >= 50 else "🔴")
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

    lines.append(f"**Overall**: {coverage_bar(totals['coverage_pct'])}{delta_str}")
    lines.append(
        f"**Lines**: {totals['covered_lines']:,} / {totals['total_lines']:,} "
        f"across {totals['file_count']} files"
    )
    lines.append("")

    groups = [
        ("🔴 0-20%", [f for f in file_coverages if f["coverage_pct"] < 20]),
        ("🟠 20-50%", [f for f in file_coverages if 20 <= f["coverage_pct"] < 50]),
        ("🟡 50-80%", [f for f in file_coverages if 50 <= f["coverage_pct"] < 80]),
        ("🟢 80-100%", [f for f in file_coverages if f["coverage_pct"] >= 80]),
    ]
    for group_label, group_files in groups:
        if not group_files:
            continue
        lines.append("<details>")
        lines.append(f"<summary>{group_label} ({len(group_files)} files)</summary>")
        lines.append("")
        lines.append("| Coverage | Lines | File |")
        lines.append("|----------|-------|------|")
        for f in group_files:
            rel = os.path.relpath(f["filename"], source_dir)
            pct = f["coverage_pct"]
            lines.append(
                f"| {pct:5.1f}% | "
                f"{f['covered_lines']}/{f['total_lines']} | "
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


def write_reports(
    *,
    label: str,
    file_coverages: list[dict],
    totals: dict,
    baseline_totals: Optional[dict],
    source_dir: Path,
    output_dir: Path,
) -> None:
    md = generate_markdown(
        label=label,
        file_coverages=file_coverages,
        totals=totals,
        baseline_totals=baseline_totals,
        source_dir=source_dir,
    )
    md_path = output_dir / f"{label}.md"
    md_path.write_text(md)
    print(f"Wrote markdown report: {md_path}")

    summary = {
        "label": label,
        "coverage_pct": round(totals["coverage_pct"], 2),
        "covered_lines": totals["covered_lines"],
        "total_lines": totals["total_lines"],
        "file_count": totals["file_count"],
    }
    if baseline_totals:
        summary["baseline_pct"] = round(baseline_totals["coverage_pct"], 2)
        summary["delta_pct"] = round(
            totals["coverage_pct"] - baseline_totals["coverage_pct"], 2
        )

    summary_path = output_dir / f"{label}-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary JSON:    {summary_path}")

    print(f"\nCoverage: {totals['coverage_pct']:.2f}%")
    if baseline_totals:
        delta = totals["coverage_pct"] - baseline_totals["coverage_pct"]
        print(f"Delta:    {'+' if delta >= 0 else ''}{delta:.2f}%")


# ---------------------------------------------------------------------------
# gcov mode
# ---------------------------------------------------------------------------

GCOVR_EXCLUDE_PATTERNS = EXCLUDE_PATTERNS


def run_gcov_mode(args: argparse.Namespace) -> None:
    gcovr_cmd = args.gcovr or find_tool("gcovr")
    gcov_cmd = args.gcov or find_tool("gcov")

    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()
    output_dir = (args.output_dir or source_dir / ".codecov").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{args.label}.json"
    xml_path = output_dir / f"{args.label}.xml"
    html_path = output_dir / f"{args.label}.html"

    cmd = [
        gcovr_cmd,
        "--root",
        str(source_dir),
        "--gcov-executable",
        gcov_cmd,
        "--exclude-unreachable-branches",
        "--exclude-throw-branches",
        "--gcov-ignore-parse-errors",
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

    with open(json_path) as f:
        data = json.load(f)

    raw = []
    for file_data in data.get("files", []):
        filename = file_data.get("filename", "") or file_data.get("file", "")
        lines = file_data.get("lines", [])
        if not filename or not lines:
            continue
        covered = sum(1 for ln in lines if ln.get("count", 0) > 0)
        total = len(lines)
        raw.append({"filename": filename, "covered_lines": covered, "total_lines": total})

    file_coverages = compute_file_coverage(raw)
    totals = compute_totals(file_coverages)

    baseline_totals = None
    if args.baseline and args.baseline.exists():
        with open(args.baseline) as f:
            bd = json.load(f)
        br = []
        for fd in bd.get("files", []):
            fname = fd.get("filename", "") or fd.get("file", "")
            lns = fd.get("lines", [])
            if fname and lns:
                cov = sum(1 for ln in lns if ln.get("count", 0) > 0)
                br.append(
                    {"filename": fname, "covered_lines": cov, "total_lines": len(lns)}
                )
        baseline_totals = compute_totals(compute_file_coverage(br))

    write_reports(
        label=args.label,
        file_coverages=file_coverages,
        totals=totals,
        baseline_totals=baseline_totals,
        source_dir=source_dir,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# llvm mode
# ---------------------------------------------------------------------------


def _find_elf_binaries(binary_dir: Path) -> list[Path]:
    binaries = []
    for path in binary_dir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            with open(path, "rb") as f:
                if f.read(4) == b"\x7fELF":
                    binaries.append(path)
        except (IOError, PermissionError):
            pass
    return binaries


def run_llvm_mode(args: argparse.Namespace) -> None:
    llvm_profdata = args.llvm_profdata or find_tool("llvm-profdata")
    llvm_cov = args.llvm_cov or find_tool("llvm-cov")

    source_dir = args.source_dir.resolve()
    profraw_dir = args.profraw_dir.resolve()
    binary_dir = args.binary_dir.resolve()
    output_dir = (args.output_dir or source_dir / ".codecov").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profraw_files = list(profraw_dir.rglob("*.profraw"))
    if not profraw_files:
        print(f"ERROR: no .profraw files found in {profraw_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(profraw_files)} .profraw file(s)")

    profdata_path = output_dir / f"{args.label}.profdata"
    merge_cmd = [
        llvm_profdata,
        "merge",
        "--sparse",
        *[str(p) for p in profraw_files],
        "-o",
        str(profdata_path),
    ]
    print(f"Running: {' '.join(merge_cmd)}")
    result = subprocess.run(merge_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"llvm-profdata stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    binaries = _find_elf_binaries(binary_dir)
    if not binaries:
        print(f"ERROR: no ELF binaries found in {binary_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(binaries)} ELF binary/libraries for coverage mapping")

    object_flags: list[str] = []
    for b in binaries:
        object_flags.extend(["-object", str(b)])

    export_cmd = [
        llvm_cov,
        "export",
        f"-instr-profile={profdata_path}",
        "--format=text",
        *object_flags,
    ]
    for pattern in EXCLUDE_PATTERNS:
        export_cmd.extend([f"--ignore-filename-regex={pattern}"])

    print(
        f"Running: llvm-cov export -instr-profile=... --format=text [{len(binaries)} objects] ..."
    )
    result = subprocess.run(export_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"llvm-cov stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result.stdout)

    raw = []
    for record in data.get("data", []):
        for file_data in record.get("files", []):
            filename = file_data.get("filename", "")
            summary = file_data.get("summary", {})
            lines = summary.get("lines", {})
            total = lines.get("count", 0)
            covered = lines.get("covered", 0)
            if filename and total > 0:
                raw.append(
                    {
                        "filename": filename,
                        "covered_lines": covered,
                        "total_lines": total,
                    }
                )

    file_coverages = compute_file_coverage(raw)
    totals = compute_totals(file_coverages)

    baseline_totals = None
    if args.baseline and args.baseline.exists():
        with open(args.baseline) as f:
            bd = json.load(f)
        br = []
        for record in bd.get("data", []):
            for fd in record.get("files", []):
                fname = fd.get("filename", "")
                lns = fd.get("summary", {}).get("lines", {})
                total = lns.get("count", 0)
                if fname and total > 0:
                    br.append(
                        {
                            "filename": fname,
                            "covered_lines": lns.get("covered", 0),
                            "total_lines": total,
                        }
                    )
        baseline_totals = compute_totals(compute_file_coverage(br))

    write_reports(
        label=args.label,
        file_coverages=file_coverages,
        totals=totals,
        baseline_totals=baseline_totals,
        source_dir=source_dir,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate code coverage reports")
    parser.add_argument(
        "--mode",
        choices=["gcov", "llvm"],
        default="gcov",
        help="Coverage toolchain (default: gcov)",
    )
    parser.add_argument(
        "--source-dir", required=True, type=Path, help="Source directory root"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for reports"
    )
    parser.add_argument("--label", type=str, default="all", help="Coverage report label")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline coverage JSON for delta comparison",
    )

    # gcov-mode args
    parser.add_argument(
        "--build-dir", type=Path, help="[gcov] Build directory with .gcda/.gcno files"
    )
    parser.add_argument("--gcovr", type=str, default=None, help="[gcov] Path to gcovr")
    parser.add_argument("--gcov", type=str, default=None, help="[gcov] Path to gcov")

    # llvm-mode args
    parser.add_argument(
        "--profraw-dir", type=Path, help="[llvm] Directory containing .profraw files"
    )
    parser.add_argument(
        "--binary-dir",
        type=Path,
        help="[llvm] Directory to search for instrumented ELF binaries",
    )
    parser.add_argument(
        "--llvm-profdata", type=str, default=None, help="[llvm] Path to llvm-profdata"
    )
    parser.add_argument(
        "--llvm-cov", type=str, default=None, help="[llvm] Path to llvm-cov"
    )

    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = (args.output_dir or source_dir / ".codecov").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gitignore = output_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("/*\n")

    print(f"Source dir:  {source_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Label:       {args.label}")
    print(f"Mode:        {args.mode}")
    print()

    args.output_dir = output_dir

    if args.mode == "gcov":
        if not args.build_dir:
            parser.error("--build-dir is required for gcov mode")
        run_gcov_mode(args)
    else:
        if not args.profraw_dir:
            parser.error("--profraw-dir is required for llvm mode")
        if not args.binary_dir:
            parser.error("--binary-dir is required for llvm mode")
        run_llvm_mode(args)


if __name__ == "__main__":
    main()

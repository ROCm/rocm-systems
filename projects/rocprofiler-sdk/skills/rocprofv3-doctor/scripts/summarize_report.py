#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


"""Summarize a rocprofv3-doctor JSON report into a markdown triage.

Reads the file written by ``rocprofv3-doctor --format json --output FILE``
(schema_version 1) and reports, in order: the root causes that need fixing,
each with the checks it blocked and its diagnoses; doctor errors, which are
bugs in a check rather than findings about the system; informational
findings; and what the run did not verify. Uses only the Python standard
library.

Exit status: 0 when a summary was written, 2 when the input cannot be read
as a rocprofv3-doctor report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUPPORTED_SCHEMA = 1

# A warn on a check registered with this severity is context, not a problem.
INFO_SEVERITY = "info"

# Checks whose absence from a run means rocprofv3 itself was never started.
SMOKE_LAUNCHER = "smoke.rocprofv3-launcher"


class ReportError(Exception):
    """The input is not a usable rocprofv3-doctor report."""


def load_report(path):
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise ReportError("cannot read {}: {}".format(path, exc.strerror or exc))
    try:
        report = json.loads(text)
    except ValueError as exc:
        raise ReportError("{} is not JSON ({})".format(path, exc))
    if not isinstance(report, dict) or "checks" not in report:
        raise ReportError(
            "{} is not a rocprofv3-doctor report; create one with "
            "`rocprofv3-doctor --format json --output FILE`".format(path)
        )
    version = report.get("schema_version")
    if version != SUPPORTED_SCHEMA:
        raise ReportError(
            "{} has schema_version {!r}; this script reads version {}".format(
                path, version, SUPPORTED_SCHEMA
            )
        )
    return report


def root_of(check_id, by_id):
    """Follow ``skipped_because_of`` links from a skipped check to its root."""
    seen = set()
    while check_id in by_id and check_id not in seen:
        seen.add(check_id)
        entry = by_id[check_id]
        if entry["status"] != "skip":
            return check_id
        upstream = entry.get("data", {}).get("skipped_because_of")
        if not upstream:
            return check_id
        check_id = upstream
    # the chain ends at a prerequisite that was excluded from the run
    return check_id


def classify(report):
    """Split checks into root causes, doctor errors, notes, and gaps."""
    checks = report["checks"]
    by_id = {entry["id"]: entry for entry in checks}

    blocked = {}
    excluded = {}
    for entry in checks:
        if entry["status"] != "skip":
            continue
        root = root_of(entry["id"], by_id)
        if root in by_id and by_id[root]["status"] != "skip":
            blocked.setdefault(root, []).append(entry["id"])
        elif root not in by_id:
            excluded.setdefault(root, []).append(entry["id"])

    causes = []
    errors = []
    notes = []
    for entry in checks:
        status = entry["status"]
        if status == "fail":
            causes.append(entry)
        elif status == "error":
            errors.append(entry)
        elif status == "warn":
            if entry.get("severity") == INFO_SEVERITY:
                notes.append(entry)
            else:
                causes.append(entry)
    # failures before warnings, keeping report order within each
    causes.sort(key=lambda entry: entry["status"] != "fail")
    return causes, errors, notes, blocked, excluded


def _indent(text, prefix="    "):
    return "\n".join(prefix + line if line else "" for line in text.splitlines())


def _diagnoses(entry):
    lines = []
    for item in entry.get("data", {}).get("diagnoses", []) or []:
        label = "likely" if item.get("confidence") == "high" else "possible"
        lines.append(
            "   - Diagnosis ({}): {}".format(label, item.get("summary", "").strip())
        )
        if item.get("evidence"):
            lines.append("     Evidence: `{}`".format(item["evidence"]))
    return lines


def render(report):
    causes, errors, notes, blocked, excluded = classify(report)
    summary = report.get("summary", {})
    ids = {entry["id"] for entry in report["checks"]}
    out = ["# rocprofv3-doctor summary", ""]

    out.append(
        "- ROCm root: `{}` (found via {}; {})".format(
            report.get("rocm_root") or "unknown",
            report.get("rocm_root_source") or "unknown",
            report.get("install_kind") or "unknown install kind",
        )
    )
    if report.get("tool_version"):
        out.append("- rocprofv3-doctor version: {}".format(report["tool_version"]))
    out.append(
        "- Checks: {pass} passed, {warn} warnings, {fail} failed, {skip} skipped, "
        "{error} errors ({total} total)".format(
            **{
                key: summary.get(key, 0)
                for key in ("pass", "warn", "fail", "skip", "error", "total")
            }
        )
    )
    out.append("")

    out.append("## Root causes to fix")
    out.append("")
    if not causes:
        out.append("None: no check failed and no actionable warning was raised.")
        out.append("")
    for number, entry in enumerate(causes, 1):
        out.append(
            "{}. **[{}] {}**: {}".format(
                number, entry["status"].upper(), entry["id"], entry.get("title", "")
            )
        )
        if entry.get("detail"):
            out.append("   - Finding: {}".format(entry["detail"].strip()))
        out.extend(_diagnoses(entry))
        if entry["id"] in blocked:
            out.append("   - Blocks: {}".format(", ".join(sorted(blocked[entry["id"]]))))
        if entry.get("remediation"):
            out.append("   - Fix:")
            out.append("")
            out.append("     ```")
            out.append(_indent(entry["remediation"].strip(), "     "))
            out.append("     ```")
        out.append("")

    if errors:
        out.append("## Doctor errors (not findings about this system)")
        out.append("")
        for entry in errors:
            out.append(
                "- {}: {} -- report this as a rocprofv3-doctor bug with the "
                "traceback from `--verbose`".format(
                    entry["id"], (entry.get("detail") or "").strip()
                )
            )
        out.append("")

    if notes:
        out.append("## Notes (informational)")
        out.append("")
        for entry in notes:
            out.append(
                "- {}: {}".format(entry["id"], (entry.get("detail") or "").strip())
            )
        out.append("")

    gaps = []
    for prerequisite, dependents in sorted(excluded.items()):
        gaps.append(
            "{} not checked: their prerequisite {} was excluded from the run".format(
                ", ".join(sorted(dependents)), prerequisite
            )
        )
    if SMOKE_LAUNCHER not in ids:
        gaps.append(
            "rocprofv3 itself was not started (smoke checks not run; add "
            "--run-smoke-test to include them)"
        )
    else:
        gaps.append(
            "Kernel tracing itself is not verified: the launcher smoke check runs "
            "a program that launches no GPU work"
        )
    out.append("## Not verified by this run")
    out.append("")
    for gap in gaps:
        out.append("- " + gap)
    return "\n".join(out).rstrip() + "\n"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Summarize a rocprofv3-doctor JSON report as markdown."
    )
    parser.add_argument("report", help="file written by rocprofv3-doctor --format json")
    parser.add_argument("-o", "--output", help="also write the summary to this file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        text = render(load_report(args.report))
    except ReportError as exc:
        sys.stderr.write("summarize_report.py: {}\n".format(exc))
        return 2
    sys.stdout.write(text)
    if args.output:
        Path(args.output).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

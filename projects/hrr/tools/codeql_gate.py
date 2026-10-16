#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Filter a CodeQL SARIF file against the reviewed alerts and fail on new ones.

The alerts file lists every alert someone has looked at, each with the reason:

- "false-positive": removed from the SARIF, so it never reaches code scanning.
- "open": a real defect not fixed yet. Kept in the SARIF, so the alert stays
  open on code scanning, but it does not fail the check.

Any other result is new: it is kept, reported as an error, and the exit status
is 1. An entry names the rule, the file, the text of the flagged line and the
sources the message links to (for a taint query, every input that reaches the
sink), not a line number. So it survives code moving around it, and stops
matching when the flagged line changes or a new input reaches it.
An entry that matches nothing is reported as stale, without failing.

Usage:
    codeql_gate.py --alerts <alerts.json> --source-root <dir> <in.sarif> <out.sarif>
"""

import argparse
import json
import re
import sys
from pathlib import Path

STATUSES = ("false-positive", "open")
FIELDS = ("rule", "file", "code", "status", "reason")
# "[user input (a command-line argument)](1)" in a result message.
SOURCE_RE = re.compile(r"\[([^\]]*)\]\(\d+\)")


class AlertsError(ValueError):
    """The alerts file is malformed."""


def load_alerts(path: Path) -> list[dict]:
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["alerts"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AlertsError(f"cannot read {path}: {exc}") from exc
    for i, entry in enumerate(entries):
        missing = [f for f in FIELDS if not isinstance(entry.get(f), str) or not entry[f].strip()]
        if missing:
            raise AlertsError(f"{path}: entry {i} lacks {', '.join(missing)}")
        if not isinstance(entry.get("sources"), list) or not all(
                isinstance(s, str) for s in entry["sources"]):
            raise AlertsError(f"{path}: entry {i} needs sources, a list of strings")
        if entry["status"] not in STATUSES:
            raise AlertsError(f"{path}: entry {i} has status {entry['status']!r}, "
                              f"expected one of {', '.join(STATUSES)}")
    return entries


def location(result: dict) -> tuple[str, int]:
    try:
        loc = result["locations"][0]["physicalLocation"]
        return loc["artifactLocation"]["uri"], int(loc["region"]["startLine"])
    except (KeyError, IndexError, TypeError, ValueError):
        return "", 0


def source_line(root: Path, uri: str, line: int) -> str:
    try:
        lines = (root / uri).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def sources(result: dict) -> set[str]:
    return set(SOURCE_RE.findall(result.get("message", {}).get("text", "")))


def matches(entry: dict, result: dict, uri: str, code: str) -> bool:
    return (result.get("ruleId") == entry["rule"] and uri == entry["file"]
            and code == entry["code"].strip() and sources(result) == set(entry["sources"]))


def gate(sarif: dict, entries: list[dict], root: Path) -> tuple[list[tuple], list[dict]]:
    """Drop false positives from `sarif` in place.

    Returns the new alerts as (file, line, rule, message) and the entries that
    matched nothing.
    """
    new: list[tuple] = []
    used: set[int] = set()
    for run in sarif.get("runs", []):
        kept = []
        for result in run.get("results", []):
            uri, line = location(result)
            code = source_line(root, uri, line)
            hit = next((i for i, e in enumerate(entries) if matches(e, result, uri, code)), None)
            if hit is not None:
                used.add(hit)
                if entries[hit]["status"] == "false-positive":
                    continue
            else:
                text = result.get("message", {}).get("text", "").splitlines()
                new.append((uri, line, result.get("ruleId"), text[0] if text else ""))
            kept.append(result)
        run["results"] = kept
    return new, [e for i, e in enumerate(entries) if i not in used]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alerts", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    try:
        entries = load_alerts(args.alerts)
        sarif = json.loads(args.input.read_text(encoding="utf-8"))
    except (AlertsError, OSError, ValueError) as exc:
        print(f"::error::{exc}")
        return 2
    new, stale = gate(sarif, entries, args.source_root)
    args.output.write_text(json.dumps(sarif), encoding="utf-8")

    for entry in stale:
        print(f"::warning file={entry['file']}::stale entry in {args.alerts}: "
              f"{entry['rule']} on `{entry['code']}` matched no result")
    for uri, line, rule, text in new:
        print(f"::error file={uri},line={line}::new CodeQL alert {rule}: {text}")
    print(f"{len(new)} new alert(s); {len(stale)} stale entr{'y' if len(stale) == 1 else 'ies'}.")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())

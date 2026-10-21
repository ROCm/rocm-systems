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
sink), not a line number. A source is named by its caption, its file and the
text of its line, because a caption alone does not tell two inputs apart:
every fread is "user input (string read by fread)". So an entry survives code
moving around it, and stops matching when the flagged line changes or a new
input reaches it, even one with a caption already listed.
An entry that matches nothing is reported as stale, without failing, on a full
scan only. On a pull request (GITHUB_EVENT_NAME pull_request or
pull_request_target) CodeQL reports only the lines the PR changes, so a
reviewed alert on any other line is missing by design; the stale check is
skipped there and the run says so. New alerts fail on every event.

Usage:
    codeql_gate.py --alerts <alerts.json> --source-root <dir> <in.sarif> <out.sarif>
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

STATUSES = ("false-positive", "open")
# Events on which codeql-action limits the results to the PR's diff.
DIFF_ONLY_EVENTS = ("pull_request", "pull_request_target")
FIELDS = ("rule", "file", "code", "status", "reason")
SOURCE_FIELDS = ("caption", "file", "code")
# "[user input (a command-line argument)](1)" in a result message: the caption,
# and the id of the related location the input is at.
SOURCE_RE = re.compile(r"\[([^\]]*)\]\((\d+)\)")


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
                isinstance(s, dict) and all(isinstance(s.get(f), str) and s[f].strip()
                                            for f in SOURCE_FIELDS)
                for s in entry["sources"]):
            raise AlertsError(f"{path}: entry {i} needs sources, a list of "
                              f"{{{', '.join(SOURCE_FIELDS)}}} objects")
        if entry["status"] not in STATUSES:
            raise AlertsError(f"{path}: entry {i} has status {entry['status']!r}, "
                              f"expected one of {', '.join(STATUSES)}")
    return entries


def place(loc) -> tuple[str, int]:
    try:
        phys = loc["physicalLocation"]
        return phys["artifactLocation"]["uri"], int(phys["region"]["startLine"])
    except (KeyError, TypeError, ValueError):
        return "", 0


def location(result: dict) -> tuple[str, int]:
    try:
        return place(result["locations"][0])
    except (KeyError, IndexError, TypeError):
        return "", 0


def source_line(root: Path, uri: str, line: int) -> str:
    try:
        lines = (root / uri).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def sources(result: dict, root: Path) -> set[tuple[str, str, str]]:
    """The inputs the message links to, as (caption, file, text of the line).

    A link with no related location gets an empty file and line, which no
    entry names, so the result is new.
    """
    related = {loc.get("id"): loc for loc in result.get("relatedLocations", [])
               if isinstance(loc, dict)}
    found = set()
    for caption, ref in SOURCE_RE.findall(result.get("message", {}).get("text", "")):
        uri, line = place(related.get(int(ref)))
        found.add((caption, uri, source_line(root, uri, line) if uri else ""))
    return found


def entry_sources(entry: dict) -> set[tuple[str, str, str]]:
    return {(s["caption"], s["file"], s["code"].strip()) for s in entry["sources"]}


def matches(entry: dict, result: dict, uri: str, code: str,
            srcs: set[tuple[str, str, str]]) -> bool:
    return (result.get("ruleId") == entry["rule"] and uri == entry["file"]
            and code == entry["code"].strip() and srcs == entry_sources(entry))


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
            srcs = sources(result, root)
            hit = next((i for i, e in enumerate(entries)
                        if matches(e, result, uri, code, srcs)), None)
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

    diff_only = os.environ.get("GITHUB_EVENT_NAME") in DIFF_ONLY_EVENTS
    if diff_only:
        stale = []
        print("Stale entries are checked on full scans only (develop pushes, "
              "workflow_dispatch): on a pull request CodeQL reports only the changed lines.")
    for entry in stale:
        print(f"::warning file={entry['file']}::stale entry in {args.alerts}: "
              f"{entry['rule']} on `{entry['code']}` matched no result")
    for uri, line, rule, text in new:
        print(f"::error file={uri},line={line}::new CodeQL alert {rule}: {text}")
    print(f"{len(new)} new alert(s); " + ("stale entries not checked." if diff_only else
          f"{len(stale)} stale entr{'y' if len(stale) == 1 else 'ies'}."))
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())

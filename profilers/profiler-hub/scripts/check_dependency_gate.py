#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fail if profiler-hub CMake can reach the network outside the
PROFILER_HUB_FETCH_DEPENDENCIES gate.

Usage: check_dependency_gate.py [PROFILER_HUB_DIR]
"""

import re
import sys
from pathlib import Path

GATE = "PROFILER_HUB_FETCH_DEPENDENCIES"

# Commands that acquire something over the network when they execute.
# FetchContent_Declare is absent on purpose: declaring is not downloading.
NETWORK_COMMANDS = {
    "fetchcontent_makeavailable": "FetchContent_MakeAvailable downloads the declared content",
    "fetchcontent_populate": "FetchContent_Populate downloads the declared content",
    "externalproject_add": "ExternalProject_Add downloads an external project",
}

# Deliberate, temporary exceptions. Each entry is a file that reaches the
# network outside the gate and is known to. Remove the entry when the file
# is brought under the gate; the lint fails if an entry stops being needed.
ALLOWED_UNGATED = {
    "cmake/sqlite3.cmake": (
        "clones and builds the SQLite amalgamation unconditionally; "
        "pending a separate owner decision on how SQLite is acquired"
    ),
}

BRACKET_OPEN = re.compile(r"\[(=*)\[")
COMMAND_START = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def strip_comments(text):
    """Blank out '#' and '#[[ ]]' comments, preserving offsets and line count.

    Quoted strings and bracket arguments keep their contents, so a network
    command spelled inside one is still seen.
    """
    out = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                out.append(text[i : i + 2])
                i += 2
                continue
            if c == '"':
                in_string = False
            out.append(c)
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "[":
            m = BRACKET_OPEN.match(text, i)
            if m:
                end = _close_bracket(text, m)
                out.append(text[i:end])
                i = end
                continue
        if c == "#":
            m = BRACKET_OPEN.match(text, i + 1)
            end = _close_bracket(text, m) if m else _end_of_line(text, i)
            out.append(re.sub(r"[^\n]", " ", text[i:end]))
            i = end
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _close_bracket(text, match):
    closer = "]" + match.group(1) + "]"
    found = text.find(closer, match.end())
    return len(text) if found < 0 else found + len(closer)


def _end_of_line(text, start):
    found = text.find("\n", start)
    return len(text) if found < 0 else found


def commands(text):
    """Yield (lowercased name, argument text, 1-based line) for each invocation.

    Parentheses are matched with a depth counter, so nested calls and
    parenthesised conditions do not truncate an argument list.
    """
    for m in COMMAND_START.finditer(text):
        i, depth, in_string = m.end(), 1, False
        while i < len(text) and depth:
            c = text[i]
            if in_string:
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    in_string = False
            elif c == '"':
                in_string = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        line = text.count("\n", 0, m.start()) + 1
        yield m.group(1).lower(), text[m.end() : i - 1], line


def _mentions_gate(condition):
    return re.fullmatch(r"\$?\{?" + GATE + r"\}?", condition.strip()) is not None


def _negates_gate(condition):
    stripped = condition.strip()
    return bool(re.fullmatch(r"NOT\s+\$?\{?" + GATE + r"\}?", stripped, re.I))


def network_reason(name, args):
    """Why this invocation reaches the network, or None.

    Matching is textual within the argument list, so a git command assembled
    from a variable is not recognised.
    """
    if name in NETWORK_COMMANDS:
        return NETWORK_COMMANDS[name]
    if name == "file" and re.match(r"\s*DOWNLOAD\b", args):
        return "file(DOWNLOAD) fetches a URL"
    if (
        name == "execute_process"
        and re.search(r"\bclone\b", args)
        and re.search(r"git", args, re.I)
    ):
        return "execute_process runs a git clone"
    return None


def scan(text):
    """Return [(line, reason)] for every network call not under the gate.

    Only `if(GATE)` and `if(NOT GATE)` gate; any other spelling, such as a
    compound condition, reads as ungated and so errs towards a false alarm.
    """
    findings = []
    stack = []
    for name, args, line in commands(strip_comments(text)):
        if name == "if":
            stack.append(
                {"prior_not_gate": _negates_gate(args), "gated": _mentions_gate(args)}
            )
        elif name == "elseif" and stack:
            top = stack[-1]
            top["gated"] = top["prior_not_gate"] or _mentions_gate(args)
            top["prior_not_gate"] = top["prior_not_gate"] or _negates_gate(args)
        elif name == "else" and stack:
            stack[-1]["gated"] = stack[-1]["prior_not_gate"]
        elif name == "endif" and stack:
            stack.pop()
        else:
            reason = network_reason(name, args)
            if reason and not any(frame["gated"] for frame in stack):
                findings.append((line, reason))
    return findings


def main(argv):
    root = (
        Path(argv[1]).resolve()
        if len(argv) > 1
        else Path(__file__).resolve().parents[1]
    )
    sources = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and (p.suffix == ".cmake" or p.name == "CMakeLists.txt")
        and not any(part.startswith("build") for part in p.relative_to(root).parts)
    )
    if not sources:
        print(f"::error::no CMake sources found under {root}")
        return 1

    failures = 0
    for path in sources:
        rel = path.relative_to(root).as_posix()
        findings = scan(path.read_text(encoding="utf-8", errors="replace"))
        if rel in ALLOWED_UNGATED:
            if not findings:
                print(
                    f"::error file={rel}::allowance is stale, remove it from ALLOWED_UNGATED"
                )
                failures += 1
            else:
                print(f"allowed: {rel} - {ALLOWED_UNGATED[rel]}")
            continue
        for line, reason in findings:
            print(f"::error file={rel},line={line}::{reason} outside {GATE}")
            failures += 1

    for rel in ALLOWED_UNGATED:
        if not (root / rel).is_file():
            print(
                f"::error::allowed file {rel} no longer exists, remove it from ALLOWED_UNGATED"
            )
            failures += 1

    if failures:
        print(
            f"\n{failures} ungated network call(s) across {len(sources)} CMake file(s)."
        )
        print(
            f"Move the call inside {GATE}, or add a reasoned allowance to this script."
        )
        return 1
    print(f"OK: {len(sources)} CMake file(s) reach the network only under {GATE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

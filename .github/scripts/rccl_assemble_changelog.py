#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fold the RCCL changelog fragments into the newest release section.

A fragment is a file named ``<name>.<section>.md`` in ``projects/rccl/changelog.d/``.
``<section>`` is one of ``added``, ``changed``, ``removed``, ``resolved`` or ``known``.
The script appends each fragment to that section of the first ``## `` release
section in ``projects/rccl/CHANGELOG.md``, then deletes the fragment files.
Fragments are applied in merge order: the order of the first-parent commits that
added them. Run it in a clone with full history, because a shallow clone loses
that order.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

CHANGELOG = Path("projects/rccl/CHANGELOG.md")
FRAGMENTS = Path("projects/rccl/changelog.d")
SECTIONS = {
    "added": "Added",
    "changed": "Changed",
    "removed": "Removed",
    "resolved": "Resolved issues",
    "known": "Known issues",
}
_ORDER = list(SECTIONS.values())
_SECTION_RE = re.compile(r"^### (.+?)\s*$")
_FRAGMENT_RE = re.compile(rf"^[^/]+\.({'|'.join(SECTIONS)})\.md$")
FRAGMENT_NAME_RULE = (
    f"<name>.<section>.md, directly in {FRAGMENTS.as_posix()}/, with <section> one of {', '.join(SECTIONS)}"
)


def fragment_section(name: str) -> str | None:
    """The ``CHANGELOG.md`` section that a fragment file name selects, or None if the name is not valid."""
    match = _FRAGMENT_RE.match(name)
    return SECTIONS[match.group(1)] if match else None


def _git(root: Path, args: list[str]) -> str:
    # surrogateescape decodes a non-UTF-8 file name the way Path does, so the names still match.
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, encoding="utf-8", errors="surrogateescape"
    ).stdout


def _merge_order(root: Path) -> dict[str, int]:
    """Position of each fragment path in the order that the commits added it, oldest first."""
    if _git(root, ["rev-parse", "--is-shallow-repository"]).strip() == "true":
        raise ValueError(f"{root} is a shallow clone, so the merge order of the fragments is lost. Fetch full history.")
    log = _git(
        root,
        ["-c", "core.quotePath=false", "log", "--first-parent", "--relative", "--no-renames", "--diff-filter=A",
         "--reverse", "--name-only", "--format=", "--", FRAGMENTS.as_posix()],
    )
    return {name: index for index, name in enumerate(line for line in log.splitlines() if line)}


def read_fragments(root: Path) -> tuple[dict[str, list[str]], list[Path]]:
    """Fragment text per section title, and the fragment files that were read, in merge order.

    A fragment that no commit added yet comes last, in file-name order.
    """
    order = _merge_order(root)
    found: dict[str, list[str]] = {title: [] for title in _ORDER}
    paths: list[Path] = []
    for path in sorted(
        (root / FRAGMENTS).glob("*.md"),
        key=lambda path: (order.get((FRAGMENTS / path.name).as_posix(), len(order)), path.name),
    ):
        if path.name == "README.md":
            continue
        if path.is_symlink():
            raise ValueError(f"{path.name}: a fragment must be a regular file, not a symlink")
        section = fragment_section(path.name)
        if section is None:
            raise ValueError(f"{path.name}: name a fragment {FRAGMENT_NAME_RULE}")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            found[section].append(text)
        paths.append(path)
    return found, paths


def _rank(title: str) -> int:
    return _ORDER.index(title) if title in _ORDER else len(_ORDER)


def merge(changelog: str, fragments: dict[str, list[str]]) -> str:
    """Return ``changelog`` with the fragments appended to the newest release section."""
    lines = changelog.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith("## ")), None)
    if start is None or "(Unreleased)" not in lines[start]:
        raise ValueError(f"{CHANGELOG}: the newest release section must be marked (Unreleased).")
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")), len(lines))

    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    for line in lines[start + 1 : end]:
        match = _SECTION_RE.match(line)
        if match:
            sections.append((match.group(1), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble.append(line)

    for title in _ORDER:
        texts = fragments.get(title, [])
        if not texts:
            continue
        index = next((i for i, (name, _body) in enumerate(sections) if name == title), None)
        if index is None:
            index = next((i for i, (name, _body) in enumerate(sections) if _rank(name) > _rank(title)), len(sections))
            sections.insert(index, (title, []))
        body = sections[index][1]
        while body and not body[-1].strip():
            body.pop()
        for text in texts:
            body.extend(text.splitlines())
        body.append("")

    rebuilt = [lines[start], *preamble]
    for title, body in sections:
        rebuilt.extend([f"### {title}", *body])
    return "\n".join([*lines[:start], *rebuilt, *lines[end:]]) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root (default: .).")
    args = parser.parse_args(argv)
    fragments, paths = read_fragments(args.root)
    if not paths:
        print("No changelog fragments to assemble.")
        return 0
    changelog = args.root / CHANGELOG
    changelog.write_text(merge(changelog.read_text(encoding="utf-8"), fragments), encoding="utf-8")
    for path in paths:
        path.unlink()
    print(f"Assembled {len(paths)} fragment(s) into {CHANGELOG}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

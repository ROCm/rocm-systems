#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Require an HRR archive version bump when existing API IDs move.

The HRR generator places runtime dispatch IDs before compiler dispatch IDs.
Appending a runtime API therefore moves the compiler tail. Those numeric IDs
are stored in events.bin, so retaining the old HRR_VERSION would make an old
archive decode as the wrong APIs. Reusing a removed API's ID for a new name
is the same mistake: the old event now names the new API.

Usage:
    check_hrr_api_compat.py --base-header <old> --current-header <new>
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


VERSION_RE = re.compile(
    r"^\s*#define\s+HRR_VERSION\s+\(\(uint16_t\)(\d+)u\)",
    re.MULTILINE,
)
ENUM_RE = re.compile(
    r"typedef\s+enum\s+hrr_api_id\s*\{(?P<body>.*?)\}\s*hrr_api_id_t\s*;",
    re.DOTALL,
)
ENTRY_RE = re.compile(r"^\s*(HRR_API_[A-Z0-9_]+)\s*=\s*(\d+)\s*,?\s*$")


class HeaderError(ValueError):
    """The generated HRR header is missing or has an ambiguous schema."""


@dataclass(frozen=True)
class ApiSchema:
    version: int
    ids: dict[str, int]


def parse_header(path: Path) -> ApiSchema:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HeaderError(f"cannot read {path}: {exc}") from exc

    version_match = VERSION_RE.search(text)
    if not version_match:
        raise HeaderError(f"{path}: HRR_VERSION definition not found")

    enum_match = ENUM_RE.search(text)
    if not enum_match:
        raise HeaderError(f"{path}: hrr_api_id_t definition not found")

    ids: dict[str, int] = {}
    values: dict[int, str] = {}
    for line in enum_match.group("body").splitlines():
        stripped = line.split("//", 1)[0].strip()
        if not stripped:
            continue
        entry = ENTRY_RE.fullmatch(stripped)
        if not entry:
            raise HeaderError(f"{path}: malformed hrr_api_id_t entry: {stripped}")
        name, raw_value = entry.groups()
        if name == "HRR_API_COUNT":
            continue
        value = int(raw_value)
        if name in ids:
            raise HeaderError(f"{path}: duplicate API name {name}")
        if value in values:
            raise HeaderError(
                f"{path}: duplicate API ID {value} for {values[value]} and {name}"
            )
        ids[name] = value
        values[value] = name

    if not ids:
        raise HeaderError(f"{path}: hrr_api_id_t contains no API entries")
    return ApiSchema(version=int(version_match.group(1)), ids=ids)


def check_compatibility(base: ApiSchema, current: ApiSchema) -> list[str]:
    if current.version < base.version:
        return [
            f"HRR_VERSION decreased from {base.version} to {current.version}"
        ]

    if current.version > base.version:
        return []

    moved = sorted(
        (name, old_id, current.ids[name])
        for name, old_id in base.ids.items()
        if name in current.ids and current.ids[name] != old_id
    )
    current_by_id = {api_id: name for name, api_id in current.ids.items()}
    # A name present on both sides is already in `moved` when its ID changes.
    # This is the other direction: the old name is gone and a new name owns
    # its ID, so an old archive decodes as the new API.
    reused = sorted(
        (name, old_id, current_by_id[old_id])
        for name, old_id in base.ids.items()
        if name not in current.ids and old_id in current_by_id
    )
    if not moved and not reused:
        return []

    details = [
        f"{name}: {old_id} -> {new_id}" for name, old_id, new_id in moved[:10]
    ]
    details.extend(
        f"{old_name} removed; {new_name} now uses {old_id}"
        for old_name, old_id, new_name in reused[:10]
    )
    shown = len(moved) + len(reused)
    if shown > len(details):
        details.append(f"... and {shown - len(details)} more")
    return [
        f"{shown} existing HRR API ID(s) moved or were reused while "
        f"HRR_VERSION remained {current.version}",
        *details,
        "increment HRR_VERSION in projects/hrr/tools/gen_hrr_api_args.py and "
        "regenerate the checked-in HRR sources",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-header", required=True, type=Path)
    parser.add_argument("--current-header", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        base = parse_header(args.base_header)
        current = parse_header(args.current_header)
    except HeaderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    errors = check_compatibility(base, current)
    if errors:
        print("error: incompatible HRR API ID change:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print(
        f"OK: existing HRR API IDs are compatible "
        f"(version {base.version} -> {current.version})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit GitHub Actions matrix JSON for one CI build group."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_MATRIX_FILE = (
    Path(__file__).resolve().parents[1] / ".github" / "ci-build-matrix.json"
)
GROUP_LABELS = {
    "ubuntu-22.04": "Ubuntu 22.04",
    "ubuntu-24.04": "Ubuntu 24.04",
    "debian": "Debian",
    "rhel": "RHEL",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=sorted(GROUP_LABELS))
    parser.add_argument(
        "--matrix-file",
        type=Path,
        default=DEFAULT_MATRIX_FILE,
        help="Path to ci-build-matrix.json.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the selected matrices instead of writing GITHUB_OUTPUT.",
    )
    return parser.parse_args(argv)


def load_matrix(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {
        "primary": list(data.get("primary", [])),
        "system_deps": list(data.get("system_deps", [])),
    }


def filter_entries(entries: Sequence[Dict[str, Any]], group: str) -> List[Dict[str, Any]]:
    return [entry for entry in entries if entry.get("group") == group]


def matrix_object(entries: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {"include": list(entries)}


def write_outputs(payload: Dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{key}={value}\n" for key, value in payload.items()]
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as file:
            file.writelines(lines)
    else:
        for line in lines:
            print(line, end="")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data = load_matrix(args.matrix_file)
    primary = filter_entries(data["primary"], args.group)
    system_deps = filter_entries(data["system_deps"], args.group)
    payload = {
        "primary_matrix": json.dumps(matrix_object(primary), separators=(",", ":")),
        "system_deps_matrix": json.dumps(
            matrix_object(system_deps), separators=(",", ":")
        ),
        "has_primary": str(bool(primary)).lower(),
        "has_system_deps": str(bool(system_deps)).lower(),
    }

    if args.print_json:
        print(json.dumps(payload, indent=2))
        return 0

    write_outputs(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

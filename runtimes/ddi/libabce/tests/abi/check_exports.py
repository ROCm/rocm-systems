#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check the retained ABCE_1 dynamic ABI in a linked Linux runtime image."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[2] / "abi/exports.txt"
    expected = {}
    for line in manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        kind, symbol = line.split()
        expected[symbol] = "OBJECT" if kind == "data" else "FUNC"
    output = subprocess.run(["readelf", "--dyn-syms", "--wide", str(args.library)],
                            check=True, text=True, stdout=subprocess.PIPE).stdout
    actual = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[7].startswith("abce_") or fields[6] == "UND":
            continue
        symbol, separator, version = fields[7].partition("@@")
        if not separator or version != "ABCE_1":
            raise ValueError(f"ABCE export has no default ABCE_1 version: {fields[7]}")
        actual[symbol] = fields[3]
    if actual != expected:
        raise ValueError(f"ABCE export drift: {set(actual.items()) ^ set(expected.items())}")
    print(f"{len(actual)} ABCE_1 function/data exports retained in {args.library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

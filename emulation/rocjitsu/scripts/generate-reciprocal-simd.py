#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regenerate SIMD correction hash plans from the scalar reciprocal tables.

Run with --check to verify the plans without writing. The C++ constexpr payload
builder separately checks that every residual occupies a unique slot.
"""

import argparse
from pathlib import Path
import re


def array_values(text, name):
    match = re.search(r"\b" + name + r"\[\]\s*=\s*\{(.*?)\};", text, re.S)
    if match is None:
        raise ValueError(f"missing array: {name}")
    return [int(value, 0) for value in re.findall(r"0x[0-9a-fA-F]+|\b\d+", match[1])]


def hash_plan(offsets, corrections):
    result = []
    for begin, end in zip(offsets, offsets[1:]):
        keys = [entry & 0x7FFF for entry in corrections[begin:end]]
        # Start at <=50% occupancy to bound the seed search and gather footprint.
        initial_bits = (max(1, 2 * len(keys)) - 1).bit_length()
        for bits in range(initial_bits, 16):
            for seed in range(1, 65536, 2):
                slots = {((key * seed) & 0x7FFF) >> (15 - bits) for key in keys}
                if len(slots) == len(keys):
                    result.append(seed | (bits << 16))
                    break
            else:
                continue
            break
        else:
            raise ValueError(f"no collision-free layout for bucket at {begin}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = (
        root / "lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/reciprocal_simd.cpp"
    )
    text = target.read_text()
    for operation in ("rcp", "rsq"):
        source = (root / f"lib/util/include/util/amdgpu_{operation}.h").read_text()
        prefix = "kAmdgpu" + operation.title() + "Correction"
        plan = hash_plan(
            array_values(source, prefix + "Offsets"), array_values(source, prefix + "s")
        )
        name = operation + "_hash_plan"
        if args.check:
            if plan != array_values(text, name):
                raise SystemExit(f"stale {name}; run {Path(__file__).name}")
            continue
        rows = [
            "    "
            + ", ".join(f"0x{value:x}u" for value in plan[index : index + 9])
            + ","
            for index in range(0, len(plan), 9)
        ]
        text = re.sub(
            r"(" + name + r"\[\]\s*=\s*\{).*?\};",
            lambda match: match[1] + "\n" + "\n".join(rows) + "\n};",
            text,
            flags=re.S,
        )
    if not args.check:
        target.write_text(text)


if __name__ == "__main__":
    main()

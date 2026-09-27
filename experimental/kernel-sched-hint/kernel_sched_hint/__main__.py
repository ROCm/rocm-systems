#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################
"""Estimate kernel duration and bound from a lowered ISA or COMGR metadata."""

from __future__ import annotations

import argparse
import json
import sys

from kernel_sched_hint.hint import predict_text
from kernel_sched_hint.devices import known_devices


def _format_seconds(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.3f} us"
    return f"{seconds * 1e9:.3f} ns"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", help="LLVM AMDGPU assembly, LLVM IR, or a metadata block"
    )
    parser.add_argument(
        "--gfx",
        required=True,
        help=f"one of {', '.join(known_devices())}, or mi300x/mi250x/mi355x",
    )
    parser.add_argument(
        "--grid", type=int, nargs=3, default=(1, 1, 1), metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--block", type=int, nargs=3, default=(256, 1, 1), metavar=("X", "Y", "Z")
    )
    parser.add_argument("--trip-count", type=int, default=1)
    parser.add_argument("--trip-count-known", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    text = open(args.path, encoding="utf-8").read()
    predictions = predict_text(
        text,
        args.gfx,
        tuple(args.grid),
        tuple(args.block),
        trip_count=args.trip_count,
        trip_count_known=args.trip_count_known,
    )
    if args.format == "json":
        json.dump([item.to_json_dict() for item in predictions], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    for pred in predictions:
        ai = (
            "inf"
            if pred.arithmetic_intensity == float("inf")
            else f"{pred.arithmetic_intensity:.3f}"
        )
        print(f"kernel: {pred.name}")
        print(
            f"bound: {pred.bound} ({pred.resource}), overlap with: {pred.overlap_with}"
        )
        print(
            f"duration: {_format_seconds(pred.duration_s)} "
            f"(roofline {_format_seconds(pred.roofline_s)}, "
            f"range {_format_seconds(pred.duration_lo_s)}–{_format_seconds(pred.duration_hi_s)})"
        )
        print(f"arithmetic intensity: {ai} flop/byte")
        print(f"waves: {pred.waves}  trip: {pred.trip_count}")
        print(f"confidence: class {pred.class_confidence}, time {pred.time_confidence}")
        for note in pred.notes:
            print(f"  - {note}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from rocprof_trace_decoder import Decoder
from rocprof_trace_decoder.code_index import CodeIndex


def _expand(paths: list[str]) -> tuple[list[Path], list[Path]]:
    att_files: list[Path] = []
    csv_files: list[Path] = []

    for arg in paths:
        expanded = sorted(glob.glob(arg)) if any(ch in arg for ch in "*?") else [arg]
        if not expanded:
            raise SystemExit(f"No matches for pattern: {arg}")
        for item in expanded:
            path = Path(item).resolve()
            suffix = path.suffix.lower()
            if suffix == ".att":
                att_files.append(path)
            elif suffix == ".csv":
                csv_files.append(path)

    return att_files, csv_files


def main() -> int:
    parser = argparse.ArgumentParser(description="CSV integration test for rocprof-trace-decoder")
    parser.add_argument("--lib", required=True, help="Path to the decoder shared library")
    parser.add_argument(
        "--suppress-warnings",
        action="store_true",
        help="Suppress info/warning messages from the decoder",
    )
    parser.add_argument("files", nargs="*", help=".att and .csv files or glob patterns")
    args = parser.parse_args()

    if not args.files:
        return 0

    att_files, csv_files = _expand(args.files)
    code_index = CodeIndex.from_stats_csv(csv_files)

    with Decoder(args.lib) as decoder:
        for att_file in att_files:
            records = decoder.parse_file(att_file, isa=code_index)
            if (
                records.info
                and not args.suppress_warnings
                and os.getenv("ATT_SUPPRESS_WARNING") is None
            ):
                for info in records.info:
                    print(f"Warning: {decoder.info_string(info)}", file=sys.stderr)
            for wave in records.waves:
                code_index.accumulate_wave(wave)

    errors = code_index.validate_expected()
    if errors:
        print(errors[0])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

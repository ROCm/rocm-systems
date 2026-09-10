#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import csv
import glob
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from rocprof_trace_decoder import CodeIndex, Decoder, HiddenLatency, Pc, analyze_hidden_latency

HEADER = [
    "CodeObj",
    "Vaddr",
    "Instruction",
    "Latency",
    "HiddenIdle",
    "HiddenStall",
    "HiddenIssue",
]

# Ways the control table can be corrupted, each covering one way the comparison could be
# wrong: missing a row the output has, missing a row the output lacks, and ignoring any of the
# three value columns. See --self-test.
SELF_TESTS = ("drop-row", "add-row", "edit-hidden", "edit-instruction", "edit-latency")
SELF_TEST_PC = Pc(address=0xDEADBEEF, code_object_id=0xFFFF)


@dataclass(frozen=True)
class Row:
    """One control row: what the instruction is, what it cost, and what was hidden."""

    instruction: str
    latency: int
    idle: int
    stall: int
    issue: int


def _expand(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        text = str(path)
        matches = sorted(glob.glob(text)) if any(char in text for char in "*?[]") else [text]
        if not matches:
            raise ValueError(f"No files matched: {path}")
        expanded.extend(Path(match).resolve() for match in matches)
    return expanded


def _read_expected(path: Path) -> dict[Pc, Row]:
    expected: dict[Pc, Row] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADER:
            raise ValueError(f"Unexpected hidden-latency header in {path}: {reader.fieldnames}")
        for row in reader:
            pc = Pc(
                address=int(row["Vaddr"], 0),
                code_object_id=int(row["CodeObj"], 0),
            )
            expected[pc] = Row(
                instruction=row["Instruction"],
                latency=int(row["Latency"], 0),
                idle=int(row["HiddenIdle"], 0),
                stall=int(row["HiddenStall"], 0),
                issue=int(row["HiddenIssue"], 0),
            )
    return expected


def _write_expected(path: Path, rows: dict[Pc, Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        for pc in _sorted_pcs(rows):
            row = rows[pc]
            writer.writerow(
                [
                    pc.code_object_id,
                    pc.address,
                    row.instruction,
                    row.latency,
                    row.idle,
                    row.stall,
                    row.issue,
                ]
            )


def _sorted_pcs(rows: dict[Pc, Row]) -> list[Pc]:
    return sorted(rows, key=lambda pc: (pc.code_object_id, pc.address))


def _add(target: dict[Pc, HiddenLatency], source: dict[Pc, HiddenLatency]) -> None:
    for pc, hidden in source.items():
        current = target.setdefault(pc, HiddenLatency())
        current += hidden


def _mutate(expected: dict[Pc, Row], mode: str) -> dict[Pc, Row]:
    """Corrupt the control table in one specific way so the comparison must reject it."""
    mutated = dict(expected)
    if mode == "add-row":
        if SELF_TEST_PC in mutated:
            raise ValueError(f"Self-test PC {SELF_TEST_PC} is present in the control CSV")
        mutated[SELF_TEST_PC] = Row(instruction="s_nop 0", latency=1, idle=1, stall=1, issue=1)
        return mutated

    if not mutated:
        raise ValueError(f"Self-test {mode} needs at least one control row")
    pc = _sorted_pcs(mutated)[0]
    if mode == "drop-row":
        del mutated[pc]
    elif mode == "edit-hidden":
        mutated[pc] = replace(mutated[pc], idle=mutated[pc].idle + 1)
    elif mode == "edit-instruction":
        mutated[pc] = replace(mutated[pc], instruction=mutated[pc].instruction + " ; mutated")
    elif mode == "edit-latency":
        mutated[pc] = replace(mutated[pc], latency=mutated[pc].latency + 1)
    else:
        raise ValueError(f"Unknown self-test mode: {mode}")
    return mutated


def _report(actual: dict[Pc, Row], expected: dict[Pc, Row], limit: int = 5) -> None:
    differing = [pc for pc in actual.keys() | expected.keys() if actual.get(pc) != expected.get(pc)]
    print(f"{len(differing)} program counter(s) differ from the control CSV:")
    for pc in sorted(differing, key=lambda pc: (pc.code_object_id, pc.address))[:limit]:
        print(f"  PC {pc.code_object_id},{pc.address}")
        print(f"    actual:   {actual.get(pc)}")
        print(f"    expected: {expected.get(pc)}")


def _compute(args: argparse.Namespace) -> tuple[dict[Pc, Row], int]:
    """Decode the captures and reduce them to the rows a control CSV holds."""
    att_paths = _expand(args.att)
    code_index = CodeIndex.from_stats_csv(_expand(args.stats))

    hidden: dict[Pc, HiddenLatency] = {}
    decoded_wave_count = 0
    with Decoder(args.lib) as decoder:
        for index, att_path in enumerate(att_paths):
            records = decoder.parse_file(att_path, isa=code_index)
            decoded_wave_count += len(records.waves)
            # Scope keys only have to be distinct per capture.
            result = analyze_hidden_latency(
                {index: records},
                code_index=code_index,
            )
            _add(hidden, result.by_pc)
            for wave in records.waves:
                code_index.accumulate_wave(wave)

    rows: dict[Pc, Row] = {}
    for pc, value in hidden.items():
        if not value.total():
            continue
        entry = code_index.entries.get(pc)
        rows[pc] = Row(
            instruction=entry.inst if entry is not None else "",
            latency=entry.latency if entry is not None else 0,
            idle=value.idle,
            stall=value.stall,
            issue=value.issue,
        )
    return rows, decoded_wave_count


def _run_self_test(args: argparse.Namespace) -> int:
    """Require the comparison to reject every corruption in SELF_TESTS.

    A comparison that ignored a column, or that only checked the control is a subset of the
    output, would otherwise keep passing every control test forever.
    """
    actual, decoded_wave_count = _compute(args)
    if not decoded_wave_count:
        print("No wave records were decoded.")
        return 1

    control = _read_expected(args.expected)
    if actual != control:
        print("The unmutated control does not match; fix that before trusting these checks.")
        _report(actual, control)
        return 1

    undetected = [mode for mode in SELF_TESTS if actual == _mutate(control, mode)]
    if undetected:
        print(f"The comparison did not reject: {', '.join(undetected)}")
        return 1

    print(f"comparison rejected all {len(SELF_TESTS)} corruptions of the control table")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate hidden-latency totals against a control CSV."
    )
    parser.add_argument("--lib", required=True, help="Path to the decoder shared library")
    parser.add_argument("--expected", type=Path, help="Hidden-latency CSV")
    parser.add_argument("att", nargs="+", type=Path, help="ATT trace files")
    parser.add_argument(
        "--stats",
        nargs="+",
        required=True,
        type=Path,
        help="Existing instruction-statistics control CSV files",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate the CSV named by --expected instead of comparing against it",
    )
    parser.add_argument(
        "--expect-empty",
        action="store_true",
        help="Require waves to decode with no nonzero hidden latency, without a control CSV",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check that the comparison rejects every corruption of the control table",
    )
    args = parser.parse_args()

    if args.expected is None and not args.expect_empty:
        parser.error("--expected is required unless --expect-empty is given")

    if args.self_test:
        return _run_self_test(args)

    actual, decoded_wave_count = _compute(args)

    if not decoded_wave_count:
        print("No wave records were decoded.")
        return 1

    if args.expect_empty:
        if actual:
            print(f"Expected no hidden latency, found {len(actual)} program counter(s).")
            return 1
        return 0

    if args.write:
        _write_expected(args.expected, actual)
        print(f"Wrote {len(actual)} row(s) to {args.expected}")
        return 0

    expected = _read_expected(args.expected)
    if actual == expected:
        return 0

    _report(actual, expected)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import filecmp
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REFERENCE_OUTPUT_DIRS = (
    ("external/dbt-rdna4", Path("external") / "dbt-rdna4"),
    ("external/external-runtime", Path("external") / "external-runtime"),
    ("external/kmd", Path("external") / "kmd"),
    ("runtime", Path("runtime")),
    ("rocjitsu", Path("rocjitsu")),
)

EXPECTED_XFAILS = {
    "ds_special_ops": "EXPAND not yet implemented for ds_atomic_async_barrier_arrive_b64",
    "monitor_load_ops": "EXPAND not yet implemented for global_load_monitor_b32",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run gfx1250 semantics corpus cases through gfx1250 -> RDNA4 DBT."
    )
    parser.add_argument(
        "--runner", required=True, type=Path, help="gfx1250_semantics_dbt path"
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
        type=Path,
        help="gfx1250-semantics-corpus directory with cases/, artifacts/, and rocjitsu/",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Build-tree output directory for DBT-produced binaries",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only the named case; may be passed multiple times",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-case runner timeout in seconds",
    )
    return parser.parse_args()


def load_cases(corpus_dir, selected):
    case_paths = sorted((corpus_dir / "cases").glob("*.json"))
    cases = []
    selected_set = set(selected or [])
    for case_path in case_paths:
        with case_path.open("r", encoding="utf-8") as f:
            case = json.load(f)
        if selected_set and case["name"] not in selected_set:
            continue
        cases.append(case)

    found = {case["name"] for case in cases}
    missing = selected_set - found
    if missing:
        raise RuntimeError("unknown corpus case(s): " + ", ".join(sorted(missing)))
    return cases


def build_runner_args(args, case, out_dir):
    name = case["name"]
    artifact_dir = args.corpus_dir / "artifacts" / name
    addresses = case["addresses"]
    runner_args = [
        str(args.runner),
        "--case",
        name,
        "--program",
        str(artifact_dir / "bin" / name),
        "--kernel",
        case["kernel"],
        "--output-dir",
        str(out_dir),
        "--grid",
        str(case["grid_size_x"]),
        "--block",
        str(case["workgroup_size_x"]),
        "--kernarg",
        str(artifact_dir / "inputs" / "kernarg.bin"),
    ]

    for input_buffer in case.get("inputs", []):
        addr_name = input_buffer.get("addr_name", input_buffer["name"])
        runner_args.extend(
            [
                "--input",
                f"{input_buffer['name']}:{addresses[addr_name]}:"
                f"{artifact_dir / input_buffer['file']}",
            ]
        )

    for output_buffer in case.get("outputs", []):
        runner_args.extend(
            [
                "--output",
                f"{output_buffer['name']}:{addresses[output_buffer['name']]}:"
                f"{output_buffer['bytes']}",
            ]
        )

    return runner_args


def summarize_failure(proc):
    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    return combined, lines[-1] if lines else f"exit {proc.returncode}"


def find_reference_outputs(corpus_dir, case):
    outputs = case.get("outputs", [])
    for label, relative_dir in REFERENCE_OUTPUT_DIRS:
        gold_dir = corpus_dir / relative_dir / case["name"]
        if all((gold_dir / output["file"]).exists() for output in outputs):
            return label, gold_dir
    return None, None


def compare_outputs(corpus_dir, case, out_dir):
    reference_label, gold_dir = find_reference_outputs(corpus_dir, case)
    if gold_dir is None:
        return None, [output["name"] for output in case.get("outputs", [])], []

    missing = []
    mismatches = []
    for output_buffer in case.get("outputs", []):
        produced = out_dir / "outputs" / f"{output_buffer['name']}.bin"
        expected = gold_dir / output_buffer["file"]
        if not produced.exists() or not expected.exists():
            missing.append(output_buffer["name"])
        elif not outputs_match(case, output_buffer, produced, expected):
            mismatches.append(output_buffer["name"])
    return reference_label, missing, mismatches


def outputs_match(case, output_buffer, produced, expected):
    if case["name"] != "scalar_control_ops" or output_buffer["name"] != "out_u64":
        return filecmp.cmp(produced, expected, shallow=False)

    produced_bytes = produced.read_bytes()
    expected_bytes = expected.read_bytes()
    if len(produced_bytes) != len(expected_bytes) or len(produced_bytes) != 12 * 8:
        return False

    # s_swap_pc returns the runtime-relocated address of its fallthrough. HSA
    # loads the translated code object at a different address than the fixed
    # KMD simulator mapping, while both kernel entries are 256-byte aligned.
    # Compare the exact fallthrough residue and keep every other result exact.
    pc_offset = 9 * 8
    produced_pc = struct.unpack_from("<Q", produced_bytes, pc_offset)[0]
    expected_pc = struct.unpack_from("<Q", expected_bytes, pc_offset)[0]
    return (
        produced_bytes[:pc_offset] == expected_bytes[:pc_offset]
        and produced_pc != 0
        and expected_pc != 0
        and (produced_pc & 0xFF) == (expected_pc & 0xFF)
        and produced_bytes[pc_offset + 8 :] == expected_bytes[pc_offset + 8 :]
    )


def run_case(args, case):
    name = case["name"]
    out_dir = args.output_dir / name
    shutil.rmtree(out_dir, ignore_errors=True)
    runner_args = build_runner_args(args, case, out_dir)

    try:
        proc = subprocess.run(
            runner_args,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "FAIL", f"timed out after {args.timeout:g}s"

    expected_failure = EXPECTED_XFAILS.get(name)
    if proc.returncode != 0:
        combined, short = summarize_failure(proc)
        if expected_failure and expected_failure in combined:
            return "XFAIL", expected_failure
        return "FAIL", short

    reference_label, missing, mismatches = compare_outputs(
        args.corpus_dir, case, out_dir
    )
    if missing:
        detail = "missing outputs"
        if reference_label:
            detail += f" against {reference_label}"
        detail += ": " + ", ".join(missing)
        if expected_failure and expected_failure in detail:
            return "XFAIL", expected_failure
        return "FAIL", detail
    if mismatches:
        detail = "mismatched outputs"
        if reference_label:
            detail += f" against {reference_label}"
        detail += ": " + ", ".join(mismatches)
        if expected_failure and expected_failure in detail:
            return "XFAIL", expected_failure
        return "FAIL", detail
    if expected_failure:
        return "XPASS", "expected failure now passes"
    return "PASS", f"{len(case.get('outputs', []))} outputs"


def main():
    args = parse_args()
    if not args.runner.exists():
        raise RuntimeError(f"runner not found: {args.runner}")
    if not (args.corpus_dir / "cases").is_dir():
        raise RuntimeError(
            f"corpus cases directory not found: {args.corpus_dir / 'cases'}"
        )

    cases = load_cases(args.corpus_dir, args.cases)
    if not cases:
        raise RuntimeError("no corpus cases selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {"PASS": 0, "XFAIL": 0, "XPASS": 0, "FAIL": 0}
    for case in cases:
        status, detail = run_case(args, case)
        counts[status] += 1
        print(f"{case['name']}: {status} {detail}", flush=True)

    total = sum(counts.values())
    print(
        "SUMMARY "
        f"pass={counts['PASS']} xfail={counts['XFAIL']} "
        f"xpass={counts['XPASS']} fail={counts['FAIL']} total={total}"
    )
    return 0 if counts["FAIL"] == 0 and counts["XPASS"] == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

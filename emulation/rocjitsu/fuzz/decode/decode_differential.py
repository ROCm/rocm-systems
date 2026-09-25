#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare rocjitsu decoder results with llvm-mc."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable

WINDOW_SIZE = 16
LLVM_CPUS = {
    "cdna1": "gfx908",
    "cdna2": "gfx90a",
    "gfx90a": "gfx90a",
    "cdna3": "gfx942",
    "gfx942": "gfx942",
    "cdna4": "gfx950",
    "gfx950": "gfx950",
    "rdna1": "gfx1010",
    "rdna2": "gfx1030",
    "rdna3": "gfx1100",
    "rdna3_5": "gfx1150",
    "rdna4": "gfx1201",
    "gfx1200": "gfx1200",
    "gfx1201": "gfx1201",
    "gfx1250": "gfx1250",
    "gfx1251": "gfx1251",
}
SUPPORTED_TARGETS = tuple(LLVM_CPUS)
ENCODING_RE = re.compile(r";\s*encoding:\s*\[([^]]+)\]")
INTEGER_RE = re.compile(r"(?<![a-z0-9_])[+-]?(?:0x[0-9a-f]+|[0-9]+)(?![a-z0-9_])")


def _canonical_integer(match: re.Match[str]) -> str:
    token = match.group(0)
    sign = 1
    if token.startswith(("-", "+")):
        if token[0] == "-":
            sign = -1
        token = token[1:]
    base = 16 if token.startswith("0x") else 10
    return str(sign * int(token, base))


def canonicalize_text(text: str) -> str:
    """Normalize presentation only; preserve spelling, token order, and modifiers."""
    result = " ".join(text.lower().split())
    result = INTEGER_RE.sub(_canonical_integer, result)
    result = re.sub(r"\s*([,\[\]:])\s*", r"\1", result)
    return result


def numeric_tokens(text: str) -> list[str]:
    lowered = text.lower()
    return [_canonical_integer(match) for match in INTEGER_RE.finditer(lowered)]


def parse_llvm_output(stdout: str, stderr: str) -> dict[str, Any]:
    """Parse only the first instruction in one llvm-mc byte-stream line."""
    invalid_at_start = re.search(
        r"^[^\n]*:1:1: (?:warning|error): invalid instruction encoding\s*$",
        stderr,
        flags=re.MULTILINE,
    )
    if invalid_at_start:
        return {"status": "invalid", "diagnostic": invalid_at_start.group(0)}

    for line in stdout.splitlines():
        encoding = ENCODING_RE.search(line)
        if not encoding:
            continue
        disassembly = line[: encoding.start()].strip()
        encoded_bytes = re.findall(r"0x[0-9a-fA-F]{2}", encoding.group(1))
        if not disassembly or not encoded_bytes:
            continue
        if re.search(r"/\*\s*invalid", disassembly, flags=re.IGNORECASE):
            comparable_disassembly = re.sub(
                r"/\*\s*invalid.*?\*/", "", disassembly, flags=re.IGNORECASE
            ).strip()
            return {
                "status": "operand_rejected",
                "size": len(encoded_bytes),
                "mnemonic": comparable_disassembly.split(maxsplit=1)[0],
                "disassembly": comparable_disassembly,
                "annotated_disassembly": disassembly,
                "diagnostic": "LLVM rejected an operand while decoding the instruction",
            }
        return {
            "status": "valid",
            "size": len(encoded_bytes),
            "mnemonic": disassembly.split(maxsplit=1)[0],
            "disassembly": disassembly,
        }

    return {
        "status": "invalid",
        "diagnostic": stderr.strip() or "no decoded instruction",
    }


def _llvm_input(data: bytes) -> str:
    return " ".join(f"0x{byte:02x}" for byte in data) + "\n"


def llvm_args(target: str) -> tuple[str, ...]:
    return (
        "--triple=amdgcn-amd-amdhsa",
        f"--mcpu={LLVM_CPUS[target]}",
        "--disassemble",
        "--show-encoding",
    )


def _run_llvm(
    llvm_mc: Path, data: bytes, timeout: float, target: str
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            [str(llvm_mc), *llvm_args(target)],
            input=_llvm_input(data),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None


def llvm_consumed_size(
    data: bytes,
    llvm_mc: Path,
    timeout: float,
    expected_disassembly: str,
    probe_failures: list[dict[str, Any]] | None = None,
    target: str = "gfx1250",
) -> int | None:
    """Find the shortest DWORD prefix LLVM accepts as the same instruction.

    llvm-mc's --show-encoding output is a canonical re-encoding. Its byte count
    can differ from the input bytes consumed, particularly for literal forms.
    A shorter prefix can also decode as a different instruction, so acceptance
    alone is insufficient for identifying the full decode's input size.

    If a tool failure prevents proving which prefix is the shortest accepted
    one, return ``None`` while preserving its details in ``probe_failures``.
    """
    expected = canonicalize_text(expected_disassembly)
    saw_tool_failure = False
    for size in range(4, len(data), 4):
        process = _run_llvm(llvm_mc, data[:size], timeout, target)
        if process is None:
            saw_tool_failure = True
            if probe_failures is not None:
                probe_failures.append(
                    {
                        "size": size,
                        "returncode": None,
                        "stderr": f"timed out after {timeout:g} seconds",
                    }
                )
            continue
        if process.returncode != 0:
            saw_tool_failure = True
            if probe_failures is not None:
                probe_failures.append(
                    {
                        "size": size,
                        "returncode": process.returncode,
                        "stderr": process.stderr.strip(),
                    }
                )
            continue
        record = parse_llvm_output(process.stdout, process.stderr)
        if record["status"] in ("valid", "operand_rejected") and (
            canonicalize_text(record["disassembly"]) == expected
        ):
            return None if saw_tool_failure else size
    return None if saw_tool_failure else len(data)


def compare_records(rocjitsu: dict[str, Any], llvm: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    if rocjitsu["status"] == "tool_failure":
        categories.append("rocjitsu_tool_failure")
    if llvm["status"] == "tool_failure":
        categories.append("llvm_tool_failure")
    if categories:
        return categories
    llvm_status = llvm["status"]
    if llvm_status == "operand_rejected":
        categories.append("llvm_operand_rejected")
        llvm_status = "valid"
    if llvm.get("prefix_tool_failures"):
        categories.append("llvm_prefix_tool_failure")
    if rocjitsu["status"] != llvm_status:
        categories.append("acceptance_mismatch")
        return categories
    if rocjitsu["status"] == "invalid":
        return categories
    if llvm.get("size") is not None and rocjitsu["size"] != llvm["size"]:
        categories.append("size_mismatch")
    if rocjitsu["mnemonic"].lower() != llvm["mnemonic"].lower():
        categories.append("mnemonic_mismatch")
    if numeric_tokens(rocjitsu["disassembly"]) != numeric_tokens(llvm["disassembly"]):
        categories.append("numeric_token_mismatch")
    if canonicalize_text(rocjitsu["disassembly"]) != canonicalize_text(
        llvm["disassembly"]
    ):
        categories.append("text_mismatch")
    return categories


def _run_checked(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command, text=True, capture_output=True, check=False, **kwargs
    )
    if process.returncode != 0:
        rendered = shlex.join(command)
        raise RuntimeError(
            f"command failed ({process.returncode}): {rendered}\n{process.stderr}"
        )
    return process


def run_one(
    path: Path,
    decoder: Path,
    llvm_mc: Path,
    target: str,
    input_qualification: str,
    timeout: float = 10.0,
) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) != WINDOW_SIZE:
        raise RuntimeError(f"input is not exactly {WINDOW_SIZE} bytes: {path}")

    try:
        roc_process = subprocess.run(
            [
                str(decoder),
                "--input",
                str(path),
                "--json",
                "--target",
                target,
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        roc_process = None
    if roc_process is None:
        roc_record = {
            "status": "tool_failure",
            "returncode": None,
            "stderr": f"timed out after {timeout:g} seconds",
        }
    elif roc_process.returncode == 0:
        try:
            roc_record = json.loads(roc_process.stdout)
        except json.JSONDecodeError as error:
            roc_record = {
                "status": "tool_failure",
                "returncode": 0,
                "stderr": f"invalid replay JSON: {error}",
                "stdout": roc_process.stdout,
            }
    else:
        roc_record = {
            "status": "tool_failure",
            "returncode": roc_process.returncode,
            "stderr": roc_process.stderr.strip(),
        }
    llvm_input = _llvm_input(data)
    llvm_process = _run_llvm(llvm_mc, data, timeout, target)
    if llvm_process is None:
        llvm_record = {
            "status": "tool_failure",
            "returncode": None,
            "stderr": f"timed out after {timeout:g} seconds",
        }
    elif llvm_process.returncode == 0:
        llvm_record = parse_llvm_output(llvm_process.stdout, llvm_process.stderr)
        if llvm_record["status"] in ("valid", "operand_rejected"):
            llvm_record["shown_encoding_size"] = llvm_record["size"]
            prefix_tool_failures: list[dict[str, Any]] = []
            llvm_record["size"] = llvm_consumed_size(
                data,
                llvm_mc,
                timeout,
                llvm_record["disassembly"],
                prefix_tool_failures,
                target,
            )
            if prefix_tool_failures:
                llvm_record["prefix_tool_failures"] = prefix_tool_failures
    else:
        llvm_record = {
            "status": "tool_failure",
            "returncode": llvm_process.returncode,
            "stderr": llvm_process.stderr.strip(),
        }
    categories = compare_records(roc_record, llvm_record)
    return {
        "input": str(path),
        "bytes": data.hex(),
        "target": target,
        "input_qualification": input_qualification,
        "categories": categories,
        "rocjitsu": roc_record,
        "llvm": llvm_record,
        "normalized": {
            "rocjitsu": canonicalize_text(roc_record.get("disassembly", "")),
            "llvm": canonicalize_text(llvm_record.get("disassembly", "")),
        },
        "reproduce": {
            "rocjitsu": shlex.join(
                [
                    str(decoder),
                    "--input",
                    str(path),
                    "--json",
                    "--target",
                    target,
                ]
            ),
            "llvm": (
                f"printf '%s\\n' {shlex.quote(llvm_input.rstrip())} | "
                f"{shlex.join([str(llvm_mc), *llvm_args(target)])}"
            ),
        },
    }


def _revision(command: list[str], fallback: str) -> str:
    try:
        lines = [
            line.strip()
            for line in _run_checked(command).stdout.splitlines()
            if line.strip()
        ]
        version_line = next((line for line in lines if "version" in line.lower()), None)
        return version_line or lines[0]
    except (OSError, RuntimeError, IndexError):
        return fallback


def _git_revision(repository: Path) -> str:
    revision = _revision(["git", "-C", str(repository), "rev-parse", "HEAD"], "unknown")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return revision + ("+dirty" if status.returncode == 0 and status.stdout else "")


def _inputs(single_input: Path | None, corpus: Path | None) -> list[Path]:
    if single_input:
        paths = [single_input]
    else:
        assert corpus is not None
        paths = sorted(
            path
            for path in corpus.rglob("*")
            if path.is_file()
            and path.name != "README.txt"
            and not any(part.startswith(".") for part in path.relative_to(corpus).parts)
        )
    wrong_size = [path for path in paths if path.stat().st_size != WINDOW_SIZE]
    if wrong_size:
        examples = ", ".join(str(path) for path in wrong_size[:3])
        raise RuntimeError(f"corpus contains non-{WINDOW_SIZE}-byte inputs: {examples}")
    if not paths:
        raise RuntimeError("no inputs found")
    return paths


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="compare one exact 16-byte input")
    source.add_argument(
        "--corpus", type=Path, help="recursively compare a window corpus"
    )
    parser.add_argument(
        "--decoder", type=Path, required=True, help="rj_decode_fuzz executable"
    )
    parser.add_argument(
        "--llvm-mc", type=Path, required=True, help="pinned llvm-mc executable"
    )
    parser.add_argument(
        "--target",
        choices=SUPPORTED_TARGETS,
        default="gfx1250",
        help="rocjitsu target profile (default: gfx1250)",
    )
    parser.add_argument(
        "--input-qualification",
        choices=("unqualified", "generated-unqualified", "manual-qualified"),
        default="unqualified",
        help=(
            "record how instruction provenance and validity were established; "
            "generated opcode seeds and raw AFL mutations are not manual qualification"
        ),
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="mismatch JSONL path"
    )
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="per-tool timeout in seconds"
    )
    parser.add_argument(
        "--rocjitsu-revision", help="override the auto-detected source revision"
    )
    parser.add_argument(
        "--llvm-revision", help="record the pinned LLVM source revision"
    )
    parser.add_argument("--allow-mismatches", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    paths = _inputs(args.input, args.corpus)
    if args.jobs < 1 or args.timeout <= 0:
        raise RuntimeError("--jobs and --timeout must be positive")

    project_root = Path(__file__).resolve().parents[2]
    revisions = {
        "rocjitsu": args.rocjitsu_revision or _git_revision(project_root),
        "llvm": args.llvm_revision
        or _revision([str(args.llvm_mc), "--version"], "unknown"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    mismatches = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = executor.map(
            lambda path: run_one(
                path,
                args.decoder,
                args.llvm_mc,
                args.target,
                args.input_qualification,
                args.timeout,
            ),
            paths,
        )
        with args.output.open("w", encoding="utf-8") as output:
            for result in results:
                if not result["categories"]:
                    continue
                mismatches += 1
                for category in result["categories"]:
                    counts[category] = counts.get(category, 0) + 1
                result["revisions"] = revisions
                json.dump(result, output, sort_keys=True)
                output.write("\n")

    summary = {
        "inputs": len(paths),
        "target": args.target,
        "input_qualification": args.input_qualification,
        "mismatches": mismatches,
        "categories": counts,
        "output": str(args.output),
        "revisions": revisions,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if mismatches == 0 or args.allow_mismatches else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"decode_differential.py: {error}", file=sys.stderr)
        raise SystemExit(2) from error

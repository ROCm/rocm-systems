#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Extract deterministic 16-byte AFL++ seeds from ELF .text sections."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

WINDOW_SIZE = 16
ALIGNMENT = 4
ELF_MACHINE_AMDGPU = 224
EF_AMDGPU_MACH = 0x0FF


def input_files(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(
                candidate for candidate in path.rglob("*") if candidate.is_file()
            )
        elif path.is_file():
            result.append(path)
        else:
            raise RuntimeError(f"input does not exist: {path}")
    return sorted(set(result))


def extract_text(llvm_objcopy: Path, source: Path, destination: Path) -> bool:
    process = subprocess.run(
        [str(llvm_objcopy), f"--dump-section=.text={destination}", str(source)],
        text=True,
        capture_output=True,
        check=False,
    )
    return process.returncode == 0 and destination.is_file()


def elf_identity(source: Path) -> tuple[int, int] | None:
    try:
        with source.open("rb") as stream:
            header = stream.read(52)
    except OSError:
        return None
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    if header[5] == 1:
        byteorder = "little"
    elif header[5] == 2:
        byteorder = "big"
    else:
        return None
    if header[4] == 1:
        flags_offset = 36
    elif header[4] == 2:
        flags_offset = 48
    else:
        return None
    if len(header) < flags_offset + 4:
        return None
    machine = int.from_bytes(header[18:20], byteorder)
    flags = int.from_bytes(header[flags_offset : flags_offset + 4], byteorder)
    return machine, flags & EF_AMDGPU_MACH


def add_text_windows(
    selected: list[tuple[int, bytes, str, int]],
    selected_windows: set[bytes],
    source: Path,
    data: bytes,
    max_seeds: int,
    max_windows_per_object: int,
) -> None:
    positions = max(0, (len(data) - WINDOW_SIZE) // ALIGNMENT + 1)
    step = max(1, (positions + max_windows_per_object - 1) // max_windows_per_object)
    for position in range(0, positions, step):
        offset = position * ALIGNMENT
        window = data[offset : offset + WINDOW_SIZE]
        if window in selected_windows:
            continue
        score = int.from_bytes(hashlib.blake2b(window, digest_size=8).digest(), "big")
        item = (-score, window, str(source), offset)
        if len(selected) < max_seeds:
            heapq.heappush(selected, item)
            selected_windows.add(window)
        elif item > selected[0]:
            removed = heapq.heapreplace(selected, item)
            selected_windows.remove(removed[1])
            selected_windows.add(window)


def tool_version(tool: Path) -> str:
    process = subprocess.run(
        [str(tool), "--version"], text=True, capture_output=True, check=False
    )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    return " | ".join(lines[:2]) if process.returncode == 0 and lines else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="ELF file or recursive corpus directory; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llvm-objcopy", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=4096)
    parser.add_argument("--max-windows-per-object", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_seeds < 1 or args.max_windows_per_object < 1:
        raise RuntimeError("seed limits must be positive")
    if args.output.exists():
        if not args.output.is_dir():
            raise RuntimeError(f"output path is not a directory: {args.output}")
        if any(args.output.iterdir()):
            raise RuntimeError(f"output directory is not empty: {args.output}")

    # The heap retains the globally lowest content hashes, making selection
    # deterministic and independent of filesystem enumeration details.
    selected: list[tuple[int, bytes, str, int]] = []
    selected_windows: set[bytes] = set()
    source_machines: dict[str, int] = {}
    examined_objects = 0
    extracted_objects = 0
    extraction_failures = 0
    skipped_inputs = 0
    skipped_elf_machines: set[int] = set()
    with tempfile.TemporaryDirectory(prefix="rj-decode-seeds-") as temporary:
        temporary_path = Path(temporary)
        for object_index, source in enumerate(input_files(args.input)):
            examined_objects += 1
            identity = elf_identity(source)
            if identity is None or identity[0] != ELF_MACHINE_AMDGPU:
                skipped_inputs += 1
                if identity is not None:
                    skipped_elf_machines.add(identity[0])
                continue
            source_machines[str(source)] = identity[1]
            text_path = temporary_path / f"{object_index}.text"
            if not extract_text(args.llvm_objcopy, source, text_path):
                extraction_failures += 1
                continue
            extracted_objects += 1
            data = text_path.read_bytes()
            add_text_windows(
                selected,
                selected_windows,
                source,
                data,
                args.max_seeds,
                args.max_windows_per_object,
            )

    if not selected:
        observed = ", ".join(str(machine) for machine in sorted(skipped_elf_machines))
        detail = f"; observed e_machine values: {observed}" if observed else ""
        raise RuntimeError(
            f"no 16-byte AMDGPU .text windows found (examined {examined_objects}, "
            f"skipped {skipped_inputs} non-AMDGPU or malformed ELF inputs, "
            f"{extraction_failures} .text extraction failures{detail})"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (_, window, source, offset) in enumerate(sorted(selected, reverse=True)):
        name = f"{index:05d}_{hashlib.sha256(window).hexdigest()[:16]}.bin"
        (args.output / name).write_bytes(window)
        manifest.append(
            {
                "file": name,
                "source": source,
                "text_offset": offset,
                "ef_amdgpu_mach": source_machines[source],
            }
        )
    manifest_path = args.output.parent / f"{args.output.name}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "window_size": WINDOW_SIZE,
                "alignment": ALIGNMENT,
                "examined_objects": examined_objects,
                "objects_with_text": extracted_objects,
                "extraction_failures": extraction_failures,
                "skipped_inputs": skipped_inputs,
                "skipped_elf_machines": sorted(skipped_elf_machines),
                "llvm_objcopy": str(args.llvm_objcopy),
                "llvm_objcopy_version": tool_version(args.llvm_objcopy),
                "seeds": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "examined_objects": examined_objects,
                "objects_with_text": extracted_objects,
                "extraction_failures": extraction_failures,
                "skipped_inputs": skipped_inputs,
                "seeds": len(manifest),
                "output": str(args.output),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"prepare_decode_seeds.py: {error}") from error

#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reject instruction-code register access backdoors.

Instruction emulators should acquire register values through the register access
facade or operand APIs, not by pairing raw storage with manual plugin hooks. Raw
VGPR storage remains valid in VM/storage code paths such as memory completion,
checkpointing, and the facade implementation itself.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
INSTRUCTION_ROOT = (
    REPO_ROOT / "emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu"
)
RAW_STORAGE = re.compile(r"\braw_vgpr_(?:data|reg)\s*(?:\(|<)")
INSTRUCTION_BACKDOOR = re.compile(
    r"\b(?:cu|c)\.(?:read|write)_vgpr\s*\(|\bwf\.cu\(\)\.(?:read|write)_vgpr\s*\(|\bSimdAccess::"
)


def checks_physical_vgpr_io(path: pathlib.Path) -> bool:
    try:
        rel = path.relative_to(INSTRUCTION_ROOT)
    except ValueError:
        return False
    return True


def main() -> int:
    failures: list[str] = []
    for path in sorted(INSTRUCTION_ROOT.rglob("*")):
        if path.suffix not in {".h", ".hpp", ".cpp", ".cc", ".c"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if RAW_STORAGE.search(line):
                rel = path.relative_to(REPO_ROOT)
                failures.append(f"{rel}:{line_number}: raw storage: {line.strip()}")
            if checks_physical_vgpr_io(path) and INSTRUCTION_BACKDOOR.search(line):
                rel = path.relative_to(REPO_ROOT)
                failures.append(
                    f"{rel}:{line_number}: physical VGPR backdoor: {line.strip()}"
                )

    if failures:
        print(
            "Instruction code must not call raw VGPR storage directly, and non-gfx1250 AMDGPU "
            "instruction helpers must not bypass RegisterAccess for physical VGPR I/O.",
            file=sys.stderr,
        )
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

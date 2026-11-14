#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regenerate the checked-in ROCr second-level trap-handler text sections.

The adjacent assembly is copied without modification from rocm-systems
projects/rocr-runtime/runtime/hsa-runtime/core/runtime/trap_handler at
f32de060ac5167e1459998e96c7ceae8a36e3999. Its NCSA license is retained in the
sources and LICENSE. Recipes follow that directory's CMakeLists.txt. The
checked-in binaries were generated with AMD clang 23.0.0git from core-7.14.
"""

import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


RECIPES = (
    ("gfx900", "9", "", None),
    ("gfx942", "942", "", None),
    ("gfx950", "950", "", None),
    ("gfx1010", "1010", "", None),
    ("gfx1030", "10", "", None),
    ("gfx1100", "11", "", None),
    ("gfx1200", "12", "_gfx12", None),
    ("gfx1250", "1250", "_gfx12", None),
    ("gfx1250", "1250_rev0", "_gfx12", "HAVE_VOP3PX_FIXUP=1"),
)
SOURCE_DIGESTS = {
    "trap_handler.s": "dc17a8606adb3f7b5d09f717a79d6d8676474e6de42a6b49e41bcfae71cffdb1",
    "trap_handler_gfx12.s": "0cd4118af7b9b3317f47fbe49fc4cc17b54dcb918a24dc4ed10cf25660cb5664",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llvm_bin", type=Path, help="directory containing clang and llvm-objcopy")
    parser.add_argument("--check", action="store_true", help="compare without changing binaries")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name, digest in SOURCE_DIGESTS.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            parser.error(f"{name} differs from the recorded source pin")
    with tempfile.TemporaryDirectory(prefix="hsa-traps-") as directory:
        temporary = Path(directory)
        for target, postfix, suffix, definition in RECIPES:
            obj = temporary / f"trap_{postfix}.hsaco"
            binary = temporary / f"trap_{postfix}.bin"
            command = [
                str(args.llvm_bin / "clang"), "-x", "assembler",
                "-target", "amdgcn-amd-amdhsa", f"-mcpu={target}",
            ]
            if definition is not None:
                command.append(f"-Wa,-defsym,{definition}")
            subprocess.run(
                command + ["-o", str(obj), str(root / f"trap_handler{suffix}.s")],
                check=True,
            )
            subprocess.run(
                [str(args.llvm_bin / "llvm-objcopy"), f"--dump-section=.text={binary}", str(obj)],
                check=True,
            )
            destination = root / binary.name
            content = binary.read_bytes()
            if args.check:
                if destination.read_bytes() != content:
                    parser.error(f"{destination.name} differs from regenerated trap code")
            else:
                destination.write_bytes(content)
            print(f"{destination.name}: {len(content)} bytes")


if __name__ == "__main__":
    main()

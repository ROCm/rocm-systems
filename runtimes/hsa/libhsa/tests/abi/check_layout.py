#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare HSA public C layouts with Rust and call the local finalizer stub."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
RUNTIMES = HERE.parents[3]
LIBRARY = Path(os.environ.get("CARGO_TARGET_DIR", HERE.parents[3] / "target")) / "release/librocddi_runtime.so"
HEADERS = RUNTIMES / "api-headers/include"
EXPECTED_RECORDS = 26


def run(command: list[str], environment: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, env=environment
    )
    if completed.returncode:
        raise RuntimeError(f"{command[0]} failed:\n{completed.stderr}")
    return completed.stdout


def main() -> int:
    if not LIBRARY.is_file():
        raise FileNotFoundError(f"build the release workspace first: {LIBRARY}")
    with tempfile.TemporaryDirectory(prefix="rocddi-hsa-abi-") as temporary:
        output = Path(temporary)
        c_binary = output / "c_layout"
        rust_binary = output / "rust_layout"
        (output / "libhsa-runtime64.so.1").symlink_to(LIBRARY)
        run(
            [
                os.environ.get("CC", "cc"),
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(HEADERS),
                str(HERE / "layout.c"),
                "-L",
                str(LIBRARY.parent),
                "-lrocddi_runtime",
                "-o",
                str(c_binary),
            ]
        )
        run(
            [
                os.environ.get("RUSTC", "rustc"),
                "--edition=2024",
                str(HERE / "layout.rs"),
                "-o",
                str(rust_binary),
            ]
        )
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(output)
        c_lines = run([str(c_binary)], environment).splitlines()
        rust_lines = run([str(rust_binary)]).splitlines()
    if len(c_lines) != EXPECTED_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_RECORDS} HSA ABI records, got {len(c_lines)}"
        )
    if c_lines != rust_lines:
        differences = difflib.unified_diff(
            c_lines, rust_lines, fromfile="C headers", tofile="Rust FFI", lineterm=""
        )
        print("\n".join(differences), file=sys.stderr)
        return 1
    print(f"{len(c_lines)} HSA C/Rust ABI layout records and local finalizer call pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

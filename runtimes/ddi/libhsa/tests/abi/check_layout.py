#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compare HSA public C layouts with Rust and call the local finalizer stub."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
RUNTIMES = HERE.parents[3]
LIBRARY = (
    Path(os.environ.get("CARGO_TARGET_DIR", RUNTIMES / "target"))
    / "release"
    / ("rocddi_runtime.dll" if os.name == "nt" else "librocddi_runtime.so")
).resolve()
# Both frontends use the workspace's native compiler adapter.
sys.path.insert(0, str(RUNTIMES / "cmake"))
from abi_probe import compile_c, run

HEADERS = RUNTIMES / "api-headers/include"
EXPECTED_RECORDS = 34


def main() -> int:
    if not LIBRARY.is_file():
        raise FileNotFoundError(f"build the release workspace first: {LIBRARY}")
    with tempfile.TemporaryDirectory(prefix="rocddi-hsa-abi-") as temporary:
        output = Path(temporary)
        c_binary = output / ("c_layout.exe" if os.name == "nt" else "c_layout")
        rust_binary = output / ("rust_layout.exe" if os.name == "nt" else "rust_layout")
        if os.name != "nt":
            (output / "libhsa-runtime64.so.1").symlink_to(LIBRARY)
        compile_c(HERE / "layout.c", c_binary, HEADERS, LIBRARY)
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
        search = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        environment[search] = os.pathsep.join(
            [str(output), str(LIBRARY.parent), environment.get(search, "")]
        )
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
    print(
        f"{len(c_lines)} HSA C/Rust ABI layout records and native handle "
        "and finalizer calls pass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

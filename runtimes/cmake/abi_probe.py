# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Native C compilation shared by the independent ABI probe runners."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


def run(command: list[str], environment: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, env=environment
    )
    if completed.returncode:
        raise RuntimeError(
            f"{command[0]} failed:\n{completed.stdout}{completed.stderr}"
        )
    return completed.stdout


def compile_c(source: Path, output: Path, headers: Path, library: Path | None = None) -> None:
    """Build with the native compiler; CC names one executable, possibly spaced."""
    compiler = os.environ.get("CC", "cl" if os.name == "nt" else "cc")
    msvc = Path(compiler).stem.lower() in {"cl", "clang-cl"}
    if msvc:
        command = [
            compiler, "/nologo", "/std:c11", "/W4", "/WX",
            f"/I{headers}", str(source), f"/Fe:{output}",
            f"/Fo:{output.with_suffix('.obj')}",
        ]
        if library is not None:
            command.append(str(library.with_suffix(".dll.lib")))
    else:
        command = [
            compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-I", str(headers), str(source), "-o", str(output),
        ]
        if library is not None:
            command.extend(["-L", str(library.parent), "-lrocddi_runtime"])
    run(command)

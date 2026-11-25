#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebuild private compute-copy kernels with a provisioned ROCm clang.

The checked-in objects use AMD clang 23.0.0git, LLVM revision
46fcb339fb61119b337f973c7ca9e710a319fdd0. Normal builds consume those objects
without requiring the GPU compiler. Generic code objects require ABI v6.
"""

import argparse
import pathlib
import subprocess

TARGETS = (
    "gfx908", "gfx90a", "gfx9-4-generic", "gfx10-1-generic",
    "gfx10-3-generic", "gfx11-generic",
    "gfx12-generic", "gfx12-5-generic",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", default="amdclang")
    arguments = parser.parse_args()
    directory = pathlib.Path(__file__).resolve().parent
    for target in TARGETS:
        subprocess.run(
            [arguments.clang, "-O2", "-x", "cl", "-Xclang",
             "-finclude-default-header", "-cl-std=CL2.0", "-target",
             "amdgcn-amd-amdhsa", f"-mcpu={target}",
             "-mcode-object-version=6", "-nogpulib", "-o", str(directory / f"{target}.hsaco"),
             str(directory / "copy.cl")],
            check=True,
        )


if __name__ == "__main__":
    main()

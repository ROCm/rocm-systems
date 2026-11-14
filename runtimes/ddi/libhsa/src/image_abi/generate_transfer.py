#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebuild image-transfer code objects using a provisioned ROCm clang.

The checked-in objects were generated with AMD clang 23.0.0git, ROCm LLVM
revision 46fcb339fb61119b337f973c7ca9e710a319fdd0. The build consumes the checked-in
objects and does not require a GPU compiler. Run with --clang /path/to/clang
after an intentional source or compiler update, then run the code-object tests.
"""

import argparse
import pathlib
import subprocess

TARGETS = ("gfx908", "gfx90a", "gfx9-4-generic", "gfx10-1-generic",
           "gfx10-3-generic", "gfx11-generic", "gfx1200", "gfx1201")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", default="amdclang")
    arguments = parser.parse_args()
    directory = pathlib.Path(__file__).resolve().parent
    for target in TARGETS:
        subprocess.run(
            [
                arguments.clang, "-O2", "-x", "cl", "-Xclang",
                "-finclude-default-header", "-cl-std=CL2.0",
                "-target", "amdgcn-amd-amdhsa", f"-mcpu={target}",
                f"-mcode-object-version={6 if 'generic' in target else 4}", "-o",
                str(directory / f"image_transfer_{target}.hsaco"),
                str(directory / "image_transfer.cl"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

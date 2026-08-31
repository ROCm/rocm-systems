# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Launch the test-torch-trace-collector gtest with its shared library paths.

Torch and ROCm live wherever the machine running the test put them, which is
not where the build machine had them, so the search mirrors the CMake logic at
run time instead of baking paths into the test definition.
"""

import os
import sys
from pathlib import Path

SKIP_RETURN_CODE = 125

_ROCM_ROOTS = (os.environ.get("ROCM_PATH", ""), "/opt/rocm")


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return SKIP_RETURN_CODE


def _torch_lib_dirs() -> list[str]:
    """Torch link dirs plus the wheel lib dir, which some wheels omit."""
    import torch
    from torch.utils import cpp_extension

    dirs = list(cpp_extension.library_paths())
    dirs.append(str(Path(torch.__file__).parent / "lib"))
    return dirs


def _first_dir_with(suffixes: tuple[str, ...], pattern: str) -> list[str]:
    for root in _ROCM_ROOTS:
        if not root:
            continue
        for suffix in suffixes:
            candidate = Path(root) / suffix
            if next(candidate.glob(pattern), None) is not None:
                return [str(candidate)]
    return []


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <gtest-binary>", file=sys.stderr)
        return 2

    binary = Path(argv[1]).resolve()
    if not binary.is_file():
        return _skip(f"{binary} not built")

    try:
        lib_dirs = _torch_lib_dirs()
    except ImportError as exc:
        return _skip(f"torch not importable: {exc}")

    lib_dirs += _first_dir_with(
        ("lib/host-math/lib", "lib", "lib64"), "librocm-openblas.so*"
    )
    lib_dirs += _first_dir_with(("lib", "lib64"), "librocprofiler-sdk-roctx.so*")

    env = dict(os.environ)
    inherited = env.get("LD_LIBRARY_PATH", "")
    if inherited:
        lib_dirs.append(inherited)
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)

    os.execve(str(binary), [str(binary)], env)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

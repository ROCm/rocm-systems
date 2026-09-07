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

try:
    import torch
    from torch.utils import cpp_extension
except ImportError as exc:
    torch = None
    cpp_extension = None
    torch_import_error = exc
else:
    torch_import_error = None

SKIP_RETURN_CODE = 125

ROCM_ROOTS = tuple(
    root for root in (os.environ.get("ROCM_PATH", ""), "/opt/rocm") if root
)


def skip_with_reason(reason: str) -> int:
    print(f"SKIP: {reason}")
    return SKIP_RETURN_CODE


def torch_lib_dirs() -> list[str]:
    """Torch link dirs plus the wheel lib dir, which some wheels omit."""
    dirs = list(cpp_extension.library_paths())
    dirs.append(str(Path(torch.__file__).parent / "lib"))
    return dirs


def first_rocm_dir_matching(suffixes: tuple[str, ...], pattern: str) -> list[str]:
    for root in ROCM_ROOTS:
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
        return skip_with_reason(f"{binary} not built")

    if torch is None:
        return skip_with_reason(f"torch not importable: {torch_import_error}")

    lib_dirs = torch_lib_dirs()

    lib_dirs += first_rocm_dir_matching(
        ("lib/host-math/lib", "lib", "lib64"), "librocm-openblas.so*"
    )
    lib_dirs += first_rocm_dir_matching(
        ("lib", "lib64"), "librocprofiler-sdk-roctx.so*"
    )

    env = dict(os.environ)
    inherited = env.get("LD_LIBRARY_PATH", "")
    if inherited:
        lib_dirs.append(inherited)
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)

    os.execve(str(binary), [str(binary)], env)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

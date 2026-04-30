#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def derive_rocm_path(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        test_script = candidate / "share" / "hsa-amd-aqlprofile" / "run_tests.sh"
        if test_script.is_file():
            return candidate
    raise RuntimeError(
        "Could not derive ROCM_PATH from an installed hsa-amd-aqlprofile test "
        "layout. Set ROCM_PATH explicitly."
    )


def get_rocm_lib_dir(rocm_path: Path) -> Path:
    for name in ("lib", "lib64"):
        candidate = rocm_path / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find ROCm library directory under {rocm_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    test_script = rocm_path / "share" / "hsa-amd-aqlprofile" / "run_tests.sh"
    if not test_script.is_file():
        raise FileNotFoundError(f"Could not find test script: {test_script}")

    env = os.environ.copy()
    lib_path = get_rocm_lib_dir(rocm_path)
    old_ld_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{lib_path}{os.pathsep}{old_ld_library_path}"
        if old_ld_library_path
        else str(lib_path)
    )

    cmd = [str(test_script)]
    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")

    subprocess.run(
        cmd,
        cwd=rocm_path,
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()

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
    if script_dir.name == "hsa-amd-aqlprofile" and script_dir.parent.name == "share":
        return script_dir.parent.parent
    output_artifacts_dir = os.getenv("OUTPUT_ARTIFACTS_DIR")
    if output_artifacts_dir:
        return Path(output_artifacts_dir)
    return script_dir.parent.parent


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path = Path(
        os.environ.get("ROCM_PATH", derive_rocm_path(script_dir))
    ).resolve()
    test_script = rocm_path / "share" / "hsa-amd-aqlprofile" / "run_tests.sh"

    env = os.environ.copy()
    lib_path = rocm_path / "lib"
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

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

TEST_FILTER = (
    "not TestOpenMPTarget and not (TestTranspose and runtime_instrument) "
    "and not TestGPUConnect"
)


def derive_rocm_path(script_dir: Path) -> Path:
    if script_dir.name == "tests" and script_dir.parent.name == "rocprofiler-systems":
        if script_dir.parent.parent.name == "share":
            return script_dir.parent.parent.parent
    rocm_path = os.getenv("ROCM_PATH")
    if rocm_path:
        return Path(rocm_path)
    return script_dir.parent.parent.parent


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path = Path(
        os.environ.get("ROCM_PATH", derive_rocm_path(script_dir))
    ).resolve()
    rocm_bin_dir = Path(os.environ.get("ROCM_BIN_DIR", rocm_path / "bin")).resolve()
    tests_dir = rocm_path / "share" / "rocprofiler-systems" / "tests"

    env = os.environ.copy()
    existing_path = env.get("PATH", "")
    env["PATH"] = (
        f"{rocm_bin_dir}{os.pathsep}{existing_path}"
        if existing_path
        else str(rocm_bin_dir)
    )
    env["ROCM_PATH"] = str(rocm_path)
    env["ROCPROFSYS_INSTALL_DIR"] = str(rocm_path)

    examples_lib_dir = rocm_path / "share" / "rocprofiler-systems" / "examples" / "lib"
    existing_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{examples_lib_dir}{os.pathsep}{existing_ld_path}"
        if existing_ld_path
        else str(examples_lib_dir)
    )

    pytest_package_exec = tests_dir / "rocprofsys-tests.pyz"
    cmd = [
        sys.executable,
        str(pytest_package_exec),
        "-k",
        TEST_FILTER,
        "--junit-xml=junit.xml",
        "--ci-mode",
        "--log-cli-level=info",
    ]

    logging.info(f"++ Exec [{rocm_path}]$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def derive_rocm_path(script_dir: Path) -> Path:
    if script_dir.name == "test" and script_dir.parent.name == "rocdecode":
        if script_dir.parent.parent.name == "share":
            return script_dir.parent.parent.parent
    rocm_path = os.getenv("ROCM_PATH")
    if rocm_path:
        return Path(rocm_path)
    return script_dir.parent.parent.parent


def setup_env(env: dict[str, str], rocm_path: Path) -> None:
    env["ROCM_PATH"] = str(rocm_path)
    logging.info(f"++ rocdecode setting ROCM_PATH={rocm_path}")
    if platform.system() == "Linux":
        hip_lib_path = rocm_path / "lib"
        logging.info(f"++ rocdecode setting LD_LIBRARY_PATH={hip_lib_path}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{hip_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = str(hip_lib_path)
    else:
        logging.info("++ rocdecode tests only supported on Linux")
        sys.exit(0)


def execute_tests(env: dict[str, str], test_source_dir: Path, build_dir: Path) -> None:
    if not test_source_dir.is_dir():
        raise FileNotFoundError(f"rocdecode tests not found in {test_source_dir}")

    build_dir.mkdir(parents=True, exist_ok=True)

    # rocdecode tests are shipped as CMake source and must be built on the target
    # machine. This serves two purposes:
    # 1. Verifies that the installed rocdecode headers and libraries are functional.
    # 2. Some test dependencies (e.g. video codec libraries) are not bundled in the
    #    TheRock artifacts and must be linked from the system at build time.
    cmd = [
        "cmake",
        "-GNinja",
        str(test_source_dir),
    ]
    logging.info(f"++ Exec [{build_dir}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=build_dir, check=True, env=env)

    cmd = [
        "ctest",
        "-N",
    ]
    logging.info(f"++ Exec [{build_dir}]$ {shlex.join(cmd)}")
    ctest_list = subprocess.run(
        cmd,
        cwd=build_dir,
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    logging.info(ctest_list.stdout)
    match = re.search(r"Total Tests:\s*(\d+)", ctest_list.stdout)
    if match is None:
        raise RuntimeError(
            "Failed to determine CTest test count from `ctest -N` output"
        )
    if int(match.group(1)) == 0:
        raise RuntimeError("CTest discovered zero rocdecode tests")

    cmd = [
        "ctest",
        "--extra-verbose",
        "--output-on-failure",
    ]
    logging.info(f"++ Exec [{build_dir}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=build_dir, check=True, env=env)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path = Path(
        os.environ.get("ROCM_PATH", derive_rocm_path(script_dir))
    ).resolve()
    test_source_dir = rocm_path / "share" / "rocdecode" / "test"
    build_dir = Path(os.environ.get("TEST_BUILD_DIR", Path.cwd() / "rocdecode-test"))
    build_dir = build_dir.resolve()

    env = os.environ.copy()
    setup_env(env, rocm_path)
    execute_tests(env, test_source_dir, build_dir)


if __name__ == "__main__":
    main()

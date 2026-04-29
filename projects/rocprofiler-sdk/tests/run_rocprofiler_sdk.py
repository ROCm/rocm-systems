# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_NAME = "rocprofiler-sdk"

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run installed rocprofiler-sdk tests.")
    parser.add_argument(
        "--rocm-path",
        type=Path,
        default=os.getenv("ROCM_PATH"),
        help="ROCm install prefix. Defaults to ROCM_PATH or the runner location.",
    )
    parser.add_argument(
        "--tests-path",
        type=Path,
        help="Installed rocprofiler-sdk tests directory.",
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        help="Relative or absolute CMake build directory for the SDK tests.",
    )
    parser.add_argument(
        "--parallel",
        default="8",
        help="Parallelism passed to CMake build and CTest.",
    )
    return parser.parse_args()


def get_rocm_path(
    script_path: Path, rocm_path: Path | None, tests_path: Path | None
) -> Path:
    if rocm_path is not None:
        return rocm_path.resolve()

    # Installed at <prefix>/share/rocprofiler-sdk/tests.
    if (
        script_path.parent.name == "tests"
        and script_path.parents[1].name == PROJECT_NAME
    ):
        return script_path.parents[3].resolve()

    if tests_path is not None:
        resolved_tests_path = tests_path.resolve()
        if (
            resolved_tests_path.name == "tests"
            and resolved_tests_path.parent.name == PROJECT_NAME
        ):
            return resolved_tests_path.parents[2].resolve()

    raise SystemExit("ROCM_PATH is required when the runner is not installed.")


def prepend_env_path(env: dict[str, str], key: str, paths: list[Path]) -> None:
    existing = env.get(key)
    values = [str(path) for path in paths]
    if existing:
        values.append(existing)
    env[key] = os.pathsep.join(values)


def get_llvm_bin_path(rocm_path: Path) -> Path:
    candidates = [
        rocm_path / "llvm" / "bin",
        rocm_path / "lib" / "llvm" / "bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def setup_env(rocm_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    sdk_path = rocm_path / "share" / PROJECT_NAME

    env["ROCM_PATH"] = str(rocm_path)
    env["HIP_PATH"] = str(rocm_path)
    env["HIP_PLATFORM"] = "amd"
    env["ROCPROFILER_METRICS_PATH"] = str(sdk_path)

    prepend_env_path(
        env,
        "LD_LIBRARY_PATH",
        [rocm_path / "lib", rocm_path / "lib" / "rocm_sysdeps" / "lib"],
    )

    return env


def get_build_dir(tests_path: Path, build_dir: str) -> Path:
    path = Path(build_dir)
    if path.is_absolute():
        return path
    return tests_path / path


def cmake_config(
    rocm_path: Path,
    tests_path: Path,
    build_dir: Path,
    env: dict[str, str],
) -> None:
    sysdeps_path = rocm_path / "lib" / "rocm_sysdeps"
    llvm_bin_path = get_llvm_bin_path(rocm_path)
    clang_path = llvm_bin_path / "amdclang"
    clang_plus_path = llvm_bin_path / "amdclang++"

    cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_PREFIX_PATH={rocm_path};{sysdeps_path}",
        f"-DCMAKE_HIP_COMPILER={clang_plus_path}",
        f"-DCMAKE_C_COMPILER={clang_path}",
        f"-DCMAKE_CXX_COMPILER={clang_plus_path}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]

    logging.info("++ Exec [%s]$ %s", tests_path, shlex.join(cmd))
    subprocess.run(cmd, cwd=tests_path, check=True, env=env)


def cmake_build(
    tests_path: Path, build_dir: Path, parallel: str, env: dict[str, str]
) -> None:
    cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--parallel",
        parallel,
    ]

    logging.info("++ Exec [%s]$ %s", tests_path, shlex.join(cmd))
    subprocess.run(cmd, cwd=tests_path, check=True, env=env)


def execute_tests(
    tests_path: Path, build_dir: Path, parallel: str, env: dict[str, str]
) -> None:
    cmd = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--parallel",
        parallel,
        "--output-on-failure",
    ]

    logging.info("++ Exec [%s]$ %s", tests_path, shlex.join(cmd))
    subprocess.run(cmd, cwd=tests_path, check=True, env=env)


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    rocm_path = get_rocm_path(script_path, args.rocm_path, args.tests_path)
    tests_path = (
        args.tests_path.resolve()
        if args.tests_path is not None
        else rocm_path / "share" / PROJECT_NAME / "tests"
    )
    build_dir = get_build_dir(tests_path, args.build_dir)
    env = setup_env(rocm_path)

    cmake_config(rocm_path, tests_path, build_dir, env)
    cmake_build(tests_path, build_dir, args.parallel, env)
    execute_tests(tests_path, build_dir, args.parallel, env)


if __name__ == "__main__":
    main()

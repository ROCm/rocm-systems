#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the rocprofiler-systems project's installed CTest suite.

This runner is installed with the rocprofiler-systems test payload. The suite
is driven by CTest, whose test definitions are generated from the project's
pytest suite (see share/rocprofiler-systems/tests/CTestTestfile.cmake). This
script discovers the installed test directory and invokes ``ctest`` against it.
"""

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)

PROJECT_NAME = "rocprofiler-systems"

# Test tiers stamped onto each generated CTest test as a label by the pytest
# conftest (see tests/test_categories.yaml). Selecting a tier with `ctest -L`
# already accounts for the always-excluded tests/labels, so the runner does not
# need to reproduce the regex/label exclusion lists here.
DEFAULT_TEST_TIER = "standard"
TEST_TIERS = ("quick", "standard", "comprehensive", "full")

HELP_EPILOG = f"""\
This script runs the {PROJECT_NAME} project CTest suite from an installed
payload. When run from an installed location, it discovers:

  <install-prefix>/share/{PROJECT_NAME}/tests

Tests are tagged with tier labels (quick/standard/comprehensive/full) at
generate time, so a tier is selected via `ctest -L <tier>`.

The install prefix is derived from this script's own installed location, so it
does not need to be configured.

Environment variables:
  ROCM_PATH     ROCm dependency prefix. Defaults to /opt/rocm.
  ROCM_BIN_DIR  Directory containing ROCm command-line tools.
  TEST_TYPE     Test tier: one of {", ".join(TEST_TIERS)}
                (default: {DEFAULT_TEST_TIER}).
"""


def format_command(cmd) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the {PROJECT_NAME} project CTest suite.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args()


def derive_install_dir(tests_dir: Path) -> Path:
    # This script is installed at <prefix>/share/<PROJECT_NAME>/tests, so the
    # install prefix is three levels up from the directory it lives in.
    if (
        tests_dir.name == "tests"
        and tests_dir.parent.name == PROJECT_NAME
        and tests_dir.parent.parent.name == "share"
    ):
        return tests_dir.parent.parent.parent
    raise RuntimeError(
        f"{Path(__file__).name} must be run from its installed location "
        f"(<prefix>/share/{PROJECT_NAME}/tests); "
        f"could not derive the install prefix from {tests_dir}."
    )


def prepend_path(env, name: str, path: Path) -> None:
    existing_value = env.get(name, "")
    path_value = str(path)
    env[name] = (
        f"{path_value}{os.pathsep}{existing_value}" if existing_value else path_value
    )


def build_env(
    install_dir: Path, tests_dir: Path, rocm_path: Path, rocm_bin_dir: Path
) -> dict[str, str]:
    env = os.environ.copy()
    prepend_path(env, "PATH", rocm_bin_dir)
    prepend_path(env, "PATH", install_dir / "bin")
    env["ROCM_PATH"] = str(rocm_path)
    env["ROCPROFSYS_INSTALL_DIR"] = str(install_dir)
    env["ROCPROFSYS_TEST_DIR"] = str(tests_dir)

    examples_lib_dir = install_dir / "share" / PROJECT_NAME / "examples" / "lib"
    prepend_path(env, "LD_LIBRARY_PATH", examples_lib_dir)
    return env


def resolve_tier() -> str:
    tier = os.getenv("TEST_TYPE", DEFAULT_TEST_TIER).lower()
    if tier not in TEST_TIERS:
        raise ValueError(
            f"Unknown TEST_TYPE {tier!r}. Expected one of: {', '.join(TEST_TIERS)}."
        )
    return tier


def run_tests(tests_dir: Path, env: dict[str, str]) -> None:
    tier = resolve_tier()

    ctest_base = ["ctest", "--test-dir", str(tests_dir)]

    # Dump the pytest configuration first; a failure here indicates a broken
    # setup
    config_cmd = ctest_base + [
        "--verbose",
        "--tests-regex",
        f"^{PROJECT_NAME}-pytest-config$",
    ]
    logging.info(f"++ Exec [{tests_dir}]$ {format_command(config_cmd)}")
    subprocess.run(config_cmd, cwd=tests_dir, check=True, env=env)

    # The tier label already encodes the always-excluded tests/labels, so we
    # only need to select it.
    cmd = ctest_base + [
        "--output-on-failure",
        "--label-regex",
        tier,
        "--repeat",
        "until-pass:3",
    ]

    logging.info(f"++ Exec [{tests_dir}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=tests_dir, check=True, env=env)


def main() -> None:
    parse_args()
    tests_dir = Path(__file__).resolve().parent
    install_dir = derive_install_dir(tests_dir)

    rocm_path = Path(os.getenv("ROCM_PATH") or "/opt/rocm").resolve()
    rocm_bin_dir = Path(os.getenv("ROCM_BIN_DIR") or rocm_path / "bin").resolve()

    ctest_file = tests_dir / "CTestTestfile.cmake"
    if not ctest_file.is_file():
        raise FileNotFoundError(
            f"Could not find the generated CTest suite: {ctest_file}. "
            "Ensure the project was built and installed with "
            "ROCPROFSYS_INSTALL_TESTING=ON."
        )

    env = build_env(install_dir, tests_dir, rocm_path, rocm_bin_dir)
    run_tests(tests_dir, env=env)


if __name__ == "__main__":
    main()

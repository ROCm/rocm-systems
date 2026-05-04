#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the amd-smi project's installed amdsmitst test suite.

amdsmitst requires GPU device access (/dev/kfd, /dev/dri), elevated
permissions, and execution on a ROCm-enabled system. GitHub-hosted CI
environments do not expose these capabilities, so this script is intended for
manual execution inside a privileged ROCm environment or container.
"""

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)

HELP_EPILOG = """\
This script runs the amd-smi project test suite from an installed ROCm payload.
When run from an installed location, it discovers:

  <rocm-prefix>/share/amd_smi/tests/amdsmitst

Environment variables:
  ROCM_PATH          ROCm install prefix override.
  TEST_TYPE         Use "quick" for the dynamic-metric quick-test subset.
  SHARD_INDEX       1-based GitHub Actions shard index.
  TOTAL_SHARDS      Total number of GitHub Actions shards.

The amdsmitst binary requires GPU device access and elevated permissions.
"""

INCLUDE_TESTS = [
    "amdsmitstReadOnly.*",
    "amdsmitstReadWrite.FanReadWrite",
    "amdsmitstReadWrite.TestOverdriveReadWrite",
    "amdsmitstReadWrite.TestPciReadWrite",
    "amdsmitstReadWrite.TestPowerReadWrite",
    "amdsmitstReadWrite.TestPerfCntrReadWrite",
    "amdsmitstReadWrite.TestEvtNotifReadWrite",
    "AmdSmiDynamicMetricTest.*",
]

EXCLUDE_TESTS = [
    "amdsmitstReadOnly.TempRead",
    "amdsmitstReadOnly.TestFrequenciesRead",
    "amdsmitstReadWrite.TestPowerReadWrite",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the amd-smi project amdsmitst test suite.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args()


def derive_rocm_path(script_dir: Path) -> Path:
    if script_dir.name == "tests" and script_dir.parent.name == "amd_smi":
        if script_dir.parent.parent.name == "share":
            return script_dir.parent.parent.parent
    raise RuntimeError(
        "Could not derive ROCM_PATH from an installed amd_smi test layout. "
        "Set ROCM_PATH explicitly."
    )


def build_test_filter(test_type: str) -> List[str]:
    if test_type == "quick":
        logging.info("Running quick tests only for amdsmitst")
        return ["--gtest_filter=AmdSmiDynamicMetricTest.*"]

    logging.info("Running full amdsmitst test suite")
    gtest_filter = f"{':'.join(INCLUDE_TESTS)}-{':'.join(EXCLUDE_TESTS)}"
    return [f"--gtest_filter={gtest_filter}"]


def main() -> None:
    parse_args()
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    test_dir = rocm_path / "share" / "amd_smi" / "tests"
    amdsmitst_bin = test_dir / "amdsmitst"
    if not amdsmitst_bin.is_file():
        raise FileNotFoundError(f"Could not find amdsmitst executable: {amdsmitst_bin}")

    env = os.environ.copy()
    env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
    env["GTEST_TOTAL_SHARDS"] = os.getenv("TOTAL_SHARDS", "1")

    cmd = [str(amdsmitst_bin)] + build_test_filter(os.getenv("TEST_TYPE", "full"))

    logging.info(f"++ Exec [{test_dir}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=test_dir, env=env, check=True)


if __name__ == "__main__":
    main()

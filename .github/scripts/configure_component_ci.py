"""
Determines which component CI jobs to run based on the GitHub event type
and the files changed in the pull request or push.
"""

import fnmatch
import logging
import os
import subprocess
from typing import Mapping

logging.basicConfig(level=logging.INFO)


def set_github_output(d: Mapping[str, str]):
    """Sets GITHUB_OUTPUT values.
    See https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/passing-information-between-jobs
    """
    logging.info(f"Setting github output:\n{d}")
    step_output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not step_output_file:
        logging.warning(
            "Warning: GITHUB_OUTPUT env var not set, can't set github outputs"
        )
        return
    with open(step_output_file, "a") as f:
        f.writelines(f"{k}={v}" + "\n" for k, v in d.items())


NVIDIA_PATTERNS = [
    "projects/clr/*",
    "projects/hip/*",
    "projects/hip-tests/*",
    "projects/hipother/*",
    ".github/workflows/component-ci.yml",
    ".github/workflows/hip-nvidia-ci.yml",
]

WSL_PATTERNS = [
    "projects/rocr-runtime/*",
    "projects/clr/*",
    "projects/hip/*",
    "projects/hip-tests/*",
    "shared/amdgpu-windows-interop/wkmi*",
    ".github/workflows/component-ci.yml",
    ".github/workflows/rocr-runtime-wsl.yml",
]

HIP_CONTRACT_TESTS_PATTERNS = [
    "projects/hip/include/hip/*",
    "projects/hip-tests/catch/contract/*",
    "projects/hip-tests/catch/tools/*",
    "projects/hip-tests/catch/config/configs/contract.yaml",
    "projects/hip-tests/catch/TEST_PLAN.md",
    ".github/workflows/component-ci.yml",
    ".github/workflows/hip-contract-coverage.yml",
]

ALL_JOBS = {"nvidia", "wsl", "hip-contract-tests"}


def get_changed_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def matches_any(files: list[str], patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(f, pattern) for f in files for pattern in patterns)


def main():
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    base_ref = os.environ.get("BASE_REF", "HEAD^")

    nvidia = False
    wsl = False
    hip_contract_tests = False

    if event_name == "workflow_dispatch":
        dispatch_jobs = os.environ.get("WORKFLOW_DISPATCH_JOBS", "all").strip()
        requested = ALL_JOBS if dispatch_jobs == "all" else set(dispatch_jobs.split())
        nvidia = "nvidia" in requested
        wsl = "wsl" in requested
        hip_contract_tests = "hip-contract-tests" in requested
    else:
        changed_files = get_changed_files(base_ref)
        logging.info("Changed files:\n" + "\n".join(changed_files))

        nvidia = matches_any(changed_files, NVIDIA_PATTERNS)
        wsl = matches_any(changed_files, WSL_PATTERNS)
        hip_contract_tests = matches_any(changed_files, HIP_CONTRACT_TESTS_PATTERNS)

    set_github_output(
        {
            "nvidia": str(nvidia).lower(),
            "wsl": str(wsl).lower(),
            "hip-contract-tests": str(hip_contract_tests).lower(),
        }
    )


if __name__ == "__main__":
    main()

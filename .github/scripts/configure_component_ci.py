"""
Determines which component CI jobs to run based on the GitHub event type
and the files changed in the pull request or push.
"""

import logging
import os
import re
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


def get_changed_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def matches_any(files: list[str], pattern: str) -> bool:
    regex = re.compile(pattern)
    return any(regex.search(f) for f in files)


def main():
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    base_ref = os.environ.get("BASE_REF", "HEAD^")

    nvidia = False
    wsl = False
    hip_contract_tests = False

    if event_name == "workflow_dispatch":
        nvidia = True
        wsl = True
        hip_contract_tests = True
    else:
        changed_files = get_changed_files(base_ref)
        logging.info("Changed files:\n" + "\n".join(changed_files))

        if matches_any(
            changed_files,
            r"^projects/(clr|hip|hip-tests|hipother)/|^\.github/workflows/(component-ci|hip-nvidia-ci)\.yml",
        ):
            nvidia = True

        if matches_any(
            changed_files,
            r"^projects/(rocr-runtime|clr|hip|hip-tests)/|^\.github/workflows/(component-ci|rocr-runtime-wsl)\.yml|^shared/amdgpu-windows-interop/wkmi",
        ):
            wsl = True

        if matches_any(
            changed_files,
            r"^projects/hip/include/hip/|^projects/hip-tests/catch/(contract|tools)/|^projects/hip-tests/catch/config/configs/contract\.yaml|^projects/hip-tests/catch/TEST_PLAN\.md|^\.github/workflows/(component-ci|hip-contract-coverage)\.yml",
        ):
            hip_contract_tests = True

    set_github_output(
        {
            "nvidia": str(nvidia).lower(),
            "wsl": str(wsl).lower(),
            "hip-contract-tests": str(hip_contract_tests).lower(),
        }
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Select the Linux CPU runner from TheRock build_runners.

TheRock's select_build_runner() reads build_runners from CI_CONFIG_PATH
(therock-ci-config) and falls back to the in-tree BUILD_RUNNER_LABELS.
This PoC uses that label for runs-on so the job lands on the same CPU
scale set as Linux builds, not on a GPU test runner.
"""

import os
import sys
from pathlib import Path


def therock_actions_dir() -> Path:
    override = os.environ.get("THEROCK_ROOT", "").strip()
    root = Path(override) if override else Path("TheRock")
    return root / "build_tools" / "github_actions"


def select_linux_build_runner() -> str:
    actions = therock_actions_dir()
    if not (actions / "amdgpu_family_matrix.py").is_file():
        raise SystemExit(f"TheRock github_actions not found at {actions}")
    sys.path.insert(0, str(actions))
    from amdgpu_family_matrix import select_build_runner

    label = select_build_runner("linux", "")
    if not label:
        raise SystemExit("select_build_runner returned an empty linux label")
    return label


def main() -> None:
    label = select_linux_build_runner()
    print(f"build_runs_on={label}")
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"build_runs_on={label}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

GPU_FAMILIES = [
    "gfx94x",
    "gfx950",
    "gfx90a",
    "gfx103x",
    "gfx110x",
    "gfx120x",
    "gfx1151",
]
TRIGGER_TYPES = ["presubmit", "postsubmit", "nightly"]


def warn(message: str) -> None:
    print(f"::warning::{message}")


def load_ci_configs(
    config_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not config_path.exists():
        warn("CI config checkout missing. Using fallback runner labels.")
        return {}, {}

    sys.path.insert(0, str(config_path))
    try:
        from ci_config_api import load_config_v1

        config = load_config_v1(config_path)
        all_families = config.get_gpu_families(TRIGGER_TYPES)
        gpu_configs = {
            family: all_families.get(family, {}).get("linux", {})
            for family in GPU_FAMILIES
        }
        return gpu_configs, config.build_runners
    except Exception as exc:
        warn(f"Failed to load CI config: {exc}. Using fallback runner labels.")
        return {}, {}


def load_gpu_configs(config_path: Path) -> dict[str, dict[str, Any]]:
    return load_ci_configs(config_path)[0]


def get_linux_build_runner(build_runners: dict[str, Any], fallback: str) -> str:
    return next(
        (
            runner["label"]
            for runner in build_runners.get("linux", {}).get("default", [])
            if runner.get("label") and runner.get("weight", 0) > 0
        ),
        fallback,
    )


def set_outputs(outputs: dict[str, str]) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output_file:
        for key, value in outputs.items():
            output_file.write(f"{key}={value}\n")


def main() -> None:
    config_path = Path(os.environ["CI_CONFIG_PATH"])
    gpu_configs, build_runners = load_ci_configs(config_path)

    outputs: dict[str, str] = {}
    fallback_linux_build_runner = os.environ.get("FALLBACK_LINUX_BUILD_RUNNER", "")
    linux_build_runner = get_linux_build_runner(
        build_runners, fallback_linux_build_runner
    )
    outputs["linux_build_runner"] = linux_build_runner
    print(f"Using Linux build runner: {linux_build_runner}")

    for family in GPU_FAMILIES:
        linux_config = gpu_configs.get(family, {})
        fallback_env = f"FALLBACK_{family.upper()}_RUNNER"
        fallback = os.environ.get(fallback_env, "")

        runner = linux_config.get("test-runs-on") or fallback
        outputs[f"{family}_runner"] = runner
        print(f"Using {family} runner: {runner}")

    fallback_sandbox = os.environ.get("FALLBACK_GFX94X_SANDBOX_RUNNER", "")
    gfx94x_config = gpu_configs.get("gfx94x", {})
    sandbox_runner = gfx94x_config.get("test-runs-on-sandbox") or fallback_sandbox
    outputs["gfx94x_sandbox_runner"] = sandbox_runner
    print(f"Using gfx94x sandbox runner: {sandbox_runner}")

    set_outputs(outputs)


if __name__ == "__main__":
    main()

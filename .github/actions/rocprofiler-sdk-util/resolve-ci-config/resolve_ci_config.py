#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def warn(message: str) -> None:
    print(f"::warning::{message}")


def load_gfx94x_linux_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        warn("CI config checkout missing. Using fallback runner labels.")
        return {}

    sys.path.insert(0, str(config_path))
    try:
        from ci_config_api import load_config_v1

        config = load_config_v1(config_path)
        return (
            config.get_gpu_families(["presubmit"])
            .get("gfx94x", {})
            .get("linux", {})
        )
    except Exception as exc:
        warn(f"Failed to load CI config: {exc}. Using fallback runner labels.")
        return {}


def set_outputs(outputs: dict[str, str]) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output_file:
        for key, value in outputs.items():
            output_file.write(f"{key}={value}\n")


def main() -> None:
    config_path = Path(os.environ["CI_CONFIG_PATH"])
    fallback_runner = os.environ["FALLBACK_GFX94X_RUNNER"]
    fallback_sandbox_runner = os.environ["FALLBACK_GFX94X_SANDBOX_RUNNER"]

    gfx94x_linux_config = load_gfx94x_linux_config(config_path)
    runner = gfx94x_linux_config.get("test-runs-on") or fallback_runner
    sandbox_runner = (
        gfx94x_linux_config.get("test-runs-on-sandbox") or fallback_sandbox_runner
    )

    print(f"Using gfx94x runner: {runner}")
    print(f"Using gfx94x sandbox runner: {sandbox_runner}")
    set_outputs(
        {
            "gfx94x_runner": runner,
            "gfx94x_sandbox_runner": sandbox_runner,
        }
    )


if __name__ == "__main__":
    main()

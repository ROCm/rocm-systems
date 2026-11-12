#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke test repeated VFIO socket options and selected-node cleanup."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    binary, config = sys.argv[1:3]
    with tempfile.TemporaryDirectory(prefix="rocjitsu-vfio-cli-") as directory:
        root = Path(directory)
        runtime = root / "runtime"
        socket = root / "gpu.sock"
        env = os.environ.copy()
        env["ROCJITSU_RUNTIME_DIR"] = str(runtime)

        # The second spelling aliases the first only after lexical normalization.
        result = subprocess.run(
            [
                binary,
                "--config",
                config,
                "--node",
                "tray7",
                "--vfio-socket",
                str(socket),
                "--vfio-socket",
                f"{root}/./gpu.sock",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        output = result.stdout + result.stderr
        expected = "socket paths resolve to the same lexical path"
        if result.returncode == 0 or expected not in output:
            print(
                f"expected duplicate socket rejection, got {result.returncode}:\n"
                f"{output}"
            )
            return 1
        if runtime.exists() and any(runtime.iterdir()):
            print(f"VFIO launch left effective config files in {runtime}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

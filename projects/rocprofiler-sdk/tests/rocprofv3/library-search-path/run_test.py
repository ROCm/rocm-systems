#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Check that profiling selects the local copy when rocm root libs contain a rival copy."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def check(label, command, expected, env, cwd):
    result = subprocess.run(
        command, env=env, cwd=cwd, text=True, capture_output=True, timeout=15
    )
    print(f"{label}: {result.stdout.strip()}", flush=True)
    assert result.returncode == 0, result.stderr
    assert "cannot be preloaded" not in result.stderr, result.stderr
    assert result.stdout.strip() == expected, (result.stdout, result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rocprofv3", "library-dir", "app", "rocm-library"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    # Preserve sanitizer preloads supplied by CTest for both launches.

    with tempfile.TemporaryDirectory(prefix="rocprofv3-library-origin-") as directory:
        root = Path(directory)
        lib = root / "lib"
        lib.mkdir()
        # A private ROCm root lets the test install its library without root access.
        for entry in args.library_dir.iterdir():
            if entry.name != "liblibrary_origin.so":
                (lib / entry.name).symlink_to(entry.resolve())
        (lib / "liblibrary_origin.so").symlink_to(args.rocm_library.resolve())

        check("Direct app", [args.app], "from local copy", env, root)
        check(
            "Explicit rocm root libs",
            [args.app],
            "from rocm root libs",
            dict(env, LD_LIBRARY_PATH=str(lib)),
            root,
        )
        check(
            "Through rocprofv3",
            [args.rocprofv3, "--marker-trace", "--rocm-root", root, "--", args.app],
            "from local copy",
            env,
            root,
        )


if __name__ == "__main__":
    main()

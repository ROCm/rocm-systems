# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise production provider CMake rules without building the simulator."""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def run(*args):
    process = subprocess.run(args, capture_output=True, text=True, check=False)
    if process.returncode:
        raise AssertionError(
            f"Command failed: {args}\n{process.stdout}{process.stderr}"
        )
    return process.stdout


def check_probe(build, config):
    directory = build / "bin" / config
    output = run(str(directory / "probe"))
    match = re.fullmatch(r"config=(.*?) digest=([0-9a-f]{64}) flag=(\d+)\n", output)
    if not match or match[1] != config:
        raise AssertionError(f"Wrong provider configuration: {output}")
    return match[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--ninja", required=True)
    parser.add_argument("--compiler", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "cmake" / "x86_v4_provider"
    common = (
        args.cmake,
        "-S",
        str(source),
        f"-DCMAKE_MAKE_PROGRAM={args.ninja}",
        f"-DCMAKE_CXX_COMPILER={args.compiler}",
    )
    with tempfile.TemporaryDirectory(prefix="rocjitsu-provider-build-") as temporary:
        root = Path(temporary)
        build = root / "multi"
        configs = ("Debug", "Release", "RelWithDebInfo", "CustomReview")
        multi = common + (
            "-B",
            str(build),
            "-G",
            "Ninja Multi-Config",
            "-DCMAKE_CROSS_CONFIGS=all",
            f"-DCMAKE_CONFIGURATION_TYPES={';'.join(configs)}",
        )
        run(*multi, "-DCMAKE_CXX_FLAGS_CUSTOMREVIEW=-DPROVIDER_FLAG_VALUE=7")
        run(
            args.cmake,
            "--build",
            str(build),
            "--config",
            "Debug",
            "--target",
            "probe:all",
        )
        digests = {config: check_probe(build, config) for config in configs}
        if len(set(digests.values())) != len(configs):
            raise AssertionError(f"Configurations share a build digest: {digests}")
        for config in configs:
            for kind in ("reloc", "local"):
                if not (build / f"provider.{config}.{kind}.o").is_file():
                    raise AssertionError(
                        f"Missing config-specific {kind} object: {config}"
                    )
            for other in configs:
                if other != config:
                    run(
                        str(build / "bin" / config / "probe"),
                        str(build / "bin" / other / "libprovider.so"),
                    )

        # The selected flags participate in its digest; unrelated config flags do not.
        run(*multi, "-DCMAKE_CXX_FLAGS_CUSTOMREVIEW=-DPROVIDER_FLAG_VALUE=9")
        run(
            args.cmake,
            "--build",
            str(build),
            "--config",
            "Debug",
            "--target",
            "probe:all",
        )
        updated = {config: check_probe(build, config) for config in configs}
        if updated["CustomReview"] == digests["CustomReview"] or any(
            updated[config] != digests[config] for config in configs[:-1]
        ):
            raise AssertionError(
                "Digest did not track only the selected configuration flags"
            )

        for index, config in enumerate(("Release", "", "CustomReview")):
            single = root / f"single-{index}"
            run(
                *common,
                "-B",
                str(single),
                "-G",
                "Ninja",
                f"-DCMAKE_BUILD_TYPE={config}",
                "-DCMAKE_CXX_FLAGS_CUSTOMREVIEW=-DPROVIDER_FLAG_VALUE=7",
            )
            run(args.cmake, "--build", str(single), "--target", "probe")
            check_probe(single, config)
        print(
            "Passed: cross-config objects, matching/mixed digests, custom flags, single/empty config"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

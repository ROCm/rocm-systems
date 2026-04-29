# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path

PROJECT_NAME = "rocprofiler-compute"

EXCLUDED_TESTS = [
    "test_profile_live_attach_detach",
]

QUICK_TESTS = [
    "test_autogen_config",
    "test_utils",
    "test_num_xcds_cli_output",
    "test_num_xcds_spec_class",
    "test_L1_cache_counters",
    "test_analyze_workloads",
    "test_analyze_commands",
    "test_metric_validation",
    "test_profile_iteration_multiplexing_1",
]

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run installed rocprofiler-compute tests."
    )
    parser.add_argument(
        "--rocm-path",
        type=Path,
        default=os.getenv("ROCM_PATH"),
        help="ROCm install prefix. Defaults to ROCM_PATH or the runner location.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Installed rocprofiler-compute test directory.",
    )
    return parser.parse_args()


def get_rocm_path(
    script_path: Path, rocm_path: Path | None, test_dir: Path | None
) -> Path:
    if rocm_path is not None:
        return rocm_path.resolve()

    # Installed at <prefix>/libexec/rocprofiler-compute.
    if (
        script_path.parent.name == PROJECT_NAME
        and script_path.parent.parent.name == "libexec"
    ):
        return script_path.parents[2].resolve()

    if test_dir is not None and test_dir.name == PROJECT_NAME:
        resolved_test_dir = test_dir.resolve()
        if resolved_test_dir.parent.name == "libexec":
            return resolved_test_dir.parents[1].resolve()

    raise SystemExit("ROCM_PATH is required when the runner is not installed.")


def prepend_env_path(env: dict[str, str], key: str, paths: list[Path]) -> None:
    existing = env.get(key)
    values = [str(path) for path in paths]
    if existing:
        values.append(existing)
    env[key] = os.pathsep.join(values)


def setup_env(rocm_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_PATH"] = str(rocm_path)

    prepend_env_path(env, "PATH", [rocm_path / "bin"])
    prepend_env_path(
        env,
        "LD_LIBRARY_PATH",
        [rocm_path / "lib", rocm_path / "lib" / "rocm_sysdeps" / "lib"],
    )

    return env


def build_ctest_command() -> list[str]:
    shard_index = int(os.getenv("SHARD_INDEX", "1")) - 1
    total_shards = int(os.getenv("TOTAL_SHARDS", "1"))

    cmd = [
        "ctest",
        "--output-on-failure",
        "--verbose",
        "--exclude-regex",
        "|".join(EXCLUDED_TESTS),
        "--tests-information",
        f"{shard_index},,{total_shards}",
    ]

    if os.getenv("TEST_TYPE", "full") == "quick":
        cmd.extend(["--tests-regex", "|".join(QUICK_TESTS)])

    return cmd


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    rocm_path = get_rocm_path(script_path, args.rocm_path, args.test_dir)
    test_dir = (
        args.test_dir.resolve()
        if args.test_dir is not None
        else rocm_path / "libexec" / PROJECT_NAME
    )
    env = setup_env(rocm_path)
    cmd = build_ctest_command()

    logging.info("++ Exec [%s]$ %s", test_dir, shlex.join(cmd))
    subprocess.run(cmd, cwd=test_dir, check=True, env=env)


if __name__ == "__main__":
    main()

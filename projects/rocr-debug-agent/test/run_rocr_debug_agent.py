#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ROCm Debug Agent tests with configurable paths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (used when CLI args are not provided):
  ROCM_PATH                ROCm install prefix containing tests/rocm-debug-agent
  OUTPUT_ARTIFACTS_DIR     Fallback install prefix containing tests/rocm-debug-agent
        """,
    )
    parser.add_argument(
        "--test-bin",
        type=Path,
        help="Path to rocm-debug-agent-test binary.",
    )
    parser.add_argument(
        "--test-script",
        type=Path,
        help="Path to run-test.py script.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of test retry attempts (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Base delay in seconds between retries (default: 5).",
    )

    args = parser.parse_args()
    args_count = sum(arg is not None for arg in (args.test_bin, args.test_script))
    if args_count not in (0, 2):
        parser.error(
            "Either provide both arguments (--test-bin, --test-script) or none."
        )

    return args


def set_core_dump_limit() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        logger.info("[OK] Core dump limit set to 0.")
    except (ValueError, OSError) as e:
        logger.warning("[!] Failed to set core dump limit: %s", e)
        logger.warning("Core files may be generated and consume disk space.")


def validate_path(path: Path, path_type: str, must_exist: bool = True) -> Path:
    try:
        return path.resolve(strict=must_exist)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{path_type} does not exist: {path}") from e
    except (OSError, RuntimeError) as e:
        raise RuntimeError(f"Could not resolve {path_type} '{path}': {e}") from e


def get_default_paths() -> Dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    candidate_dirs = []
    if script_dir.name == "rocm-debug-agent" and script_dir.parent.name == "tests":
        candidate_dirs.append(script_dir)

    rocm_path = os.getenv("ROCM_PATH")
    if rocm_path:
        candidate_dirs.append(Path(rocm_path) / "tests" / "rocm-debug-agent")

    artifacts_dir = os.getenv("OUTPUT_ARTIFACTS_DIR")
    if artifacts_dir:
        candidate_dirs.append(Path(artifacts_dir) / "tests" / "rocm-debug-agent")

    for test_dir in candidate_dirs:
        test_bin = test_dir / "rocm-debug-agent-test"
        test_script = test_dir / "run-test.py"
        if test_bin.exists() and test_script.exists():
            return {
                "test_bin": validate_path(test_bin, "rocm-debug-agent-test"),
                "test_script": validate_path(test_script, "run-test.py"),
            }

    searched_dirs = ", ".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(
        f"rocm-debug-agent test bundle not found. Searched: {searched_dirs}"
    )


def get_python_executable() -> str:
    if not sys.executable:
        raise RuntimeError("Could not identify a valid Python executable path.")

    python_executable = Path(sys.executable)
    if not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")

    return str(python_executable)


def print_section(title: str) -> None:
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"{title:^80}")
    logger.info("=" * 80)


def run_tests(
    python_executable: str,
    test_script: Path,
    working_dir: Path,
    test_bin_dir: Path,
    env_vars: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> None:
    if env_vars is None:
        env_vars = os.environ.copy()

    cmd = [python_executable, str(test_script), str(test_bin_dir)]
    last_error: Optional[subprocess.CalledProcessError] = None

    for attempt in range(1, max_retries + 1):
        print_section(f"Running tests (attempt {attempt}/{max_retries})")
        logger.info("Exec [%s]$ %s", working_dir, shlex.join(cmd))

        start_time = time.perf_counter()
        try:
            subprocess.run(cmd, cwd=working_dir, check=True, env=env_vars)
        except subprocess.CalledProcessError as e:
            last_error = e
            duration = time.perf_counter() - start_time
            logger.error(
                "[X] Attempt %s/%s failed with exit code %s after %.2fs",
                attempt,
                max_retries,
                e.returncode,
                duration,
            )

            if attempt < max_retries:
                wait_time = attempt * retry_delay
                logger.info("Retrying in %ss...", wait_time)
                time.sleep(wait_time)
            continue

        duration = time.perf_counter() - start_time
        print_section(
            f"[OK] Tests succeeded on attempt {attempt}. Duration: {duration:.2f}s"
        )
        return

    raise RuntimeError(f"All {max_retries} attempts failed.") from last_error


def main() -> None:
    args = parse_arguments()

    print_section("Path discovery")
    if args.test_bin is not None:
        logger.info("Using paths from command-line arguments.")
        test_bin = validate_path(args.test_bin, "--test-bin")
        test_script = validate_path(args.test_script, "--test-script")
    else:
        logger.info("Using default paths from the installed test bundle.")
        defaults = get_default_paths()
        test_bin = defaults["test_bin"]
        test_script = defaults["test_script"]

    test_bin_dir = test_bin.parent
    logger.info("Test binary: %s", test_bin)
    logger.info("Test script: %s", test_script)
    logger.info("Test binary directory: %s", test_bin_dir)

    python_executable = get_python_executable()
    logger.info("Located Python executable: %s", python_executable)

    print_section("Disabling core file generation")
    set_core_dump_limit()

    run_tests(
        python_executable=python_executable,
        test_script=test_script,
        working_dir=test_bin_dir,
        test_bin_dir=test_bin_dir,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )


if __name__ == "__main__":
    main()

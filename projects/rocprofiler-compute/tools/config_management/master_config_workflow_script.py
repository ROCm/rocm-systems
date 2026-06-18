#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TOOLS_DIR = SCRIPT_DIR

SOC_ROOT = REPO_ROOT / "src" / "rocprof_compute_soc"
ANALYSIS_CONFIGS = SOC_ROOT / "analysis_configs"

GFX9_TEMPLATE = ANALYSIS_CONFIGS / "gfx9_config_template.yaml"
GFX11_TEMPLATE = ANALYSIS_CONFIGS / "gfx11_config_template.yaml"

PYTHON = sys.executable

VERIFY_SCRIPT = TOOLS_DIR / "verify_against_config_template.py"
HASH_CHECKER_SCRIPT = REPO_ROOT / "src" / "utils" / "hash_checker.py"


def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def fatal(msg):
    print(f"\nFATAL: {msg}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--hash-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.ci or args.hash_only:
        if not HASH_CHECKER_SCRIPT.exists():
            fatal("hash_checker.py not found")
        sys.exit(run([PYTHON, HASH_CHECKER_SCRIPT]))

    if not VERIFY_SCRIPT.exists():
        fatal("verify_against_config_template.py not found")

    rc = run([
        PYTHON,
        VERIFY_SCRIPT,
        ANALYSIS_CONFIGS,
        "--gfx9-template",
        GFX9_TEMPLATE,
        "--gfx11-template",
        GFX11_TEMPLATE,
    ])
    if rc != 0:
        fatal("Template / architecture verification failed")

    if args.validate_only:
        print("\nValidation successful.")
        sys.exit(0)

    print(
        "\nNo workflow selected.\n"
        "Use one of:\n"
        "  --validate-only\n"
        "  --hash-only / --ci\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()

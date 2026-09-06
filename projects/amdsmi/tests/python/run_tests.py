#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified test runner — run one tier, several, or all of them.

Tiers:
    unit           hardware-free logic and mocked-CLI suites
    integration    per-API suites that drive live devices
    functional     end-to-end behaviour suites
    cli            amd-smi command-line suites

Usage (installed):
    /opt/rocm/share/amd_smi/tests/python_unittest/run_tests.py --unit -v

Usage (source):
    tests/python/run_tests.py                        # every tier
    tests/python/run_tests.py --unit                 # one tier
    tests/python/run_tests.py --unit --integration   # several
"""

import argparse
import os
import sys

# Resolve the package root (the directory containing common/) regardless of
# where this script is installed or invoked from.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import common.common as common  # noqa: E402  (sys.path bootstrapped above)

TIERS = ("unit", "integration", "functional", "cli")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="run_tests.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for tier in TIERS:
        parser.add_argument(f"--{tier}", action="store_true", help=f"run the {tier} tier")

    # run_test_dir() re-reads these from sys.argv; they are declared here so a
    # mistyped option is refused rather than widening the run to every tier.
    passthrough = parser.add_argument_group("passed through to the runner")
    passthrough.add_argument("-v", "--verbose", action="count", default=0)
    passthrough.add_argument("-q", "--quiet", action="store_true")
    passthrough.add_argument("-b", "--buffer", action="store_true")
    passthrough.add_argument("-k", "--keyword", metavar="PATTERN")
    passthrough.add_argument("-x", "--exclude", metavar="PATTERN")
    passthrough.add_argument("-l", "--list", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    selected = [tier for tier in TIERS if getattr(args, tier)] or list(TIERS)
    common.run_test_dir(selected, f"AMD SMI Tests ({', '.join(selected)})", _here)


if __name__ == "__main__":
    main()

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

Options:
    -v / --verbose    Verbose output (show per-test names)
    -q / --quiet      Quiet output
    -b / --buffer     Buffer stdout/stderr during tests
    -k "pattern"      Only run tests matching the substring
    -x "pattern"      Skip tests matching the substring
    --list / -l       List all available tests without running them
"""

import os
import sys

# Resolve the package root (the directory containing common/) regardless of
# where this script is installed or invoked from.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import common.common as common  # noqa: E402  (sys.path bootstrapped above)

TIERS = ("unit", "integration", "functional", "cli")
_LONG_OPTIONS = frozenset(
    {"--verbose", "--quiet", "--buffer", "--keyword", "--exclude", "--list", "--help"}
)


def _unknown_options(argv):
    """Long options this runner does not define.

    Naming no tier runs every tier, so a mistyped tier would silently widen the
    run to the device-driven suites instead of narrowing it.
    """
    known = _LONG_OPTIONS | {f"--{tier}" for tier in TIERS}
    unknown = []
    skip = False
    for arg in argv[1:]:
        if skip:
            skip = False
        elif arg in ("-k", "--keyword", "-x", "--exclude"):
            skip = True
        elif arg.startswith("--") and arg not in known:
            unknown.append(arg)
    return unknown


if "-h" in sys.argv or "--help" in sys.argv:
    print(__doc__)
    sys.exit(0)

_bad = _unknown_options(sys.argv)
if _bad:
    print(f"error: unknown option(s): {', '.join(_bad)}", file=sys.stderr)
    print(f"tiers: {', '.join('--' + tier for tier in TIERS)}", file=sys.stderr)
    sys.exit(2)

selected = [tier for tier in TIERS if f"--{tier}" in sys.argv] or list(TIERS)

common.run_test_dir(selected, f"AMD SMI Tests ({', '.join(selected)})", _here)

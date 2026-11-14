# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Argument validation shared by the launcher and ROCgdb's Python commands."""

import argparse
from decimal import Decimal
import re


def parse_duration(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(us|ms|s)", value)
    if not match:
        raise argparse.ArgumentTypeError("duration needs units: 100us, 10ms, or 1s")
    ns = int(Decimal(match[1]) * {"us": 1000, "ms": 1000000, "s": 1000000000}[match[2]])
    if not 0 < ns <= 3600 * 1000000000:
        raise argparse.ArgumentTypeError("duration must be between 1ns and 1h")
    return ns


def parse_skip(value):
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--skip requires a nonnegative integer")
    if not 0 <= count <= 2147483647:
        raise argparse.ArgumentTypeError("--skip must be between 0 and 2147483647")
    return count

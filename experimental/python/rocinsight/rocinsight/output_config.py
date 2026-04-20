#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""
Minimal output configuration for rocinsight.

Pure-Python replacement for rocpd's output_config (which required libpyrocpd).
Provides the same interface used by analyze.py: output_file, output_path,
and add_args().
"""

import argparse


class output_config:
    """Simple data class holding output file/directory settings."""

    def __init__(
        self,
        output_file=None,
        output_path=None,
        **_ignored,
    ):
        self.output_file = output_file
        self.output_path = output_path

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add --output / --dir flags to the given parser."""
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        default=None,
        metavar="NAME",
        help=(
            "Base name for output file (without extension). "
            "Extension is added automatically based on --format."
        ),
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest="output_path",
        default=None,
        metavar="DIR",
        help="Directory for output file (default: current directory).",
    )

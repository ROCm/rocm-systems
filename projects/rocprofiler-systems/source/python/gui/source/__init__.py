#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import absolute_import

__author__ = "AMD Research"
__copyright__ = "Copyright 2023, Advanced Micro Devices, Inc."
__license__ = "MIT"
__maintainer__ = "AMD Research"
__status__ = "Development"


def _get_version():
    import os
    from pathlib import Path

    this_dir = Path(__file__).resolve().parent
    ver_path = os.path.join(f"{this_dir}", "VERSION")
    if os.path.exists(ver_path):
        with open(ver_path, "r") as f:
            return f.read().strip("\n")
    return "???"


__version__ = _get_version()

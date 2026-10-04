# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared paths, tier names, and pinned download URLs."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import NoReturn

NIGHTLY_TARBALL_INDEX = "https://nightly.repo.amd.com/rocm/core/tarball/"

NIGHTLY_TARBALL_BASE = "https://nightly.repo.amd.com/rocm/core/tarball"

ROCM_SYSTEMS_REPO = "https://github.com/ROCm/rocm-systems.git"

PROJECT_SUBDIR = "projects/rocprofiler-systems"

REQUIRED_ROCPROFSYS_BINARIES = [
    "rocprof-sys-run",
    "rocprof-sys-instrument",
    "rocprof-sys-sample",
    "rocprof-sys-avail",
    "rocprof-sys-causal",
]

TIER_ORDER = ["quick", "standard", "comprehensive", "full"]

TEST_CATEGORIES_REL = Path("tests") / "test_categories.yaml"

MULTIARCH_VARIANT = "multiarch"

DEFAULT_MIN_FREE_GB = 40

TRACE_PROCESSOR_SHELL_URL = (
    "https://commondatastorage.googleapis.com/perfetto-luci-artifacts/"
    "v47.0/linux-amd64/trace_processor_shell"
)

TRACE_PROCESSOR_SHELL_SHA256 = (
    "832425c3c7934904d1e0ec1721beb51423de7dbcf399a899973f2b6b464603fa"
)

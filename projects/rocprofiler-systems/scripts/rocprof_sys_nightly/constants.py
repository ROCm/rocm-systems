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

RELEASE_TARBALL_BASE = "https://rc.repo.amd.com/rocm/core/tarball"

TARBALL_CHANNELS = {
    "nightly": NIGHTLY_TARBALL_BASE,
    "release": RELEASE_TARBALL_BASE,
}


TARBALL_CHOICES = ("nightly", "tests", "release")


def tarball_selection(choice: str) -> tuple[str, bool]:
    """Map a ``--tarball`` value to ``(index channel, include tests archive)``.

    ``tests`` stays on the nightly index and also extracts the matching
    ``-tests-`` archive. That archive is sample media, so the dist tarball is
    still downloaded first.
    """
    if choice == "nightly":
        return "nightly", False
    if choice == "tests":
        return "nightly", True
    if choice == "release":
        return "release", False
    known = ", ".join(TARBALL_CHOICES)
    raise ValueError(f"unknown tarball choice {choice!r}; expected {known}")


def tarball_urls(channel: str) -> tuple[str, str]:
    """Return ``(index_url, archive_base)`` for a tarball channel.

    ``index_url`` has a trailing slash. ``archive_base`` does not.
    """
    try:
        base = TARBALL_CHANNELS[channel]
    except KeyError:
        known = ", ".join(sorted(TARBALL_CHANNELS))
        raise ValueError(
            f"unknown tarball channel {channel!r}; expected {known}"
        ) from None
    return base + "/", base


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

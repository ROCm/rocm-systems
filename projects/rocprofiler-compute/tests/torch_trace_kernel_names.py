# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Kernel name normalization for torch trace coverage tests."""

from __future__ import annotations

import re
from typing import Set

_KERNEL_SUFFIX_RE = re.compile(r"(?:\s*(?:\[clone \.[^\]]+\]|\.kd))+\s*$")


def normalize_kernel_name(name: str) -> str:
    """Return ``name`` without trailing ``.kd`` or ``[clone ...]`` suffixes."""
    return _KERNEL_SUFFIX_RE.sub("", name).strip()


def normalize_kernel_names(names: Set[str]) -> Set[str]:
    """Return the normalized form of each name in ``names``."""
    return {normalize_kernel_name(name) for name in names}

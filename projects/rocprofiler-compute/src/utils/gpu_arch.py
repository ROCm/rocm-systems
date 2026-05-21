# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Lightweight GPU architecture canonicalization helpers."""

from __future__ import annotations

from typing import Optional


def canonical_gpu_arch(gpu_arch: Optional[str]) -> Optional[str]:
    """Map LLVM GPU targets that share one SoC and analysis config tree."""
    if gpu_arch is None:
        return None
    if gpu_arch == "gfx1152":
        return "gfx1151"
    return gpu_arch


def canonical_config_arch(gpu_arch: Optional[str]) -> Optional[str]:
    """Map GPU architectures to the shared analysis-config directory name."""
    if gpu_arch is None:
        return None
    if gpu_arch.startswith("gfx115"):
        return "gfx115x"
    return canonical_gpu_arch(gpu_arch)

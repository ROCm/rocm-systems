"""Shared PMC block-limit helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from perfxpert.knowledge import load_yaml

PMC_BLOCK_LIMIT_DEFAULT = 4


@lru_cache(maxsize=1)
def _limits_data() -> Dict[str, Dict[str, Any]]:
    return load_yaml("pmc_limits")


def default_block_limits() -> Dict[str, int]:
    """Return conservative default limits by block."""
    return {
        block: int(info.get("limit", PMC_BLOCK_LIMIT_DEFAULT))
        for block, info in _limits_data().get("per_block_limits", {}).items()
    }


def pmc_block(counter: str) -> str:
    """Return the hardware block name for a counter."""
    return counter.split("_", 1)[0]


def pmc_block_limit(block: str, gpu_arch: str = "") -> int:
    """Return the per-pass counter limit for ``block`` on ``gpu_arch``."""
    data = _limits_data()
    arch_limits = data.get("gpu_arch_limits", {})
    if gpu_arch in arch_limits and block in arch_limits[gpu_arch]:
        return int(arch_limits[gpu_arch][block])

    info = data.get("per_block_limits", {}).get(block, {})

    # Backward-compatible reader for older knowledge files.
    arch_limits = info.get("arch_limits") if isinstance(info, dict) else {}
    if isinstance(arch_limits, dict) and gpu_arch in arch_limits:
        return int(arch_limits[gpu_arch])

    legacy_key = f"{gpu_arch}_limit" if gpu_arch else ""
    if legacy_key and legacy_key in info:
        return int(info[legacy_key])

    return int(info.get("limit", PMC_BLOCK_LIMIT_DEFAULT))

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Julia Tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

import subprocess
from functools import lru_cache
from typing import Optional

pytestmark = [pytest.mark.julia]


# =============================================================================
# Helper Functions
# =============================================================================


@lru_cache(maxsize=1)
def _resolve_julia_lib_paths(julia_binary: str) -> Optional[tuple[str, str]]:
    """Resolve Julia library paths by invoking the Julia binary.

    Results are cached so the subprocess is only run once per session.

    Args:
        julia_binary: Path to the Julia binary (string for hashability).

    Returns:
        Tuple of (lib/julia path, lib path), or None on failure.
    """

    # Run Julia to get the correct library paths
    # This is necessary because 'julia' may be a launcher, not the actual binary
    try:
        r1 = subprocess.run(
            [julia_binary, "-e", 'println(joinpath(Sys.BINDIR, "..", "lib", "julia"))'],
            capture_output=True,
            text=True,
            timeout=10,
        )

        r2 = subprocess.run(
            [julia_binary, "-e", 'println(joinpath(Sys.BINDIR, "..", "lib"))'],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if r1.returncode != 0 or r2.returncode != 0:
            return None
        julia_lib_julia = (r1.stdout or "").strip()
        julia_lib = (r2.stdout or "").strip()
        if not julia_lib_julia or not julia_lib:
            return None

    except (subprocess.SubprocessError, OSError):
        return None

    return (julia_lib_julia, julia_lib)


# =============================================================================
# Julia Fixtures
# =============================================================================


@pytest.fixture
def julia_lib_paths(rocprof_config) -> Optional[tuple[str, str]]:
    """Resolve the Julia-specific library paths, or None if unavailable."""
    if not rocprof_config.capabilities.julia_exec:
        return None

    return _resolve_julia_lib_paths(str(rocprof_config.capabilities.julia_exec))


# =============================================================================
# Julia Tests
# =============================================================================


class TestJulia(RocprofsysTest):
    @pytest.mark.hpc
    @pytest.mark.gpu
    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_vecadd(self, mode, julia_lib_paths, rocprof_config):
        if julia_lib_paths is None:
            pytest.skip("Unable to resolve Julia library paths")

        paths = [julia_lib_paths[0], julia_lib_paths[1], self.library_path]
        julia_environment = {
            "ROCPROFSYS_TRACE": "ON",
            "ROCPROFSYS_PROFILE": "ON",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "ROCPROFSYS_USE_PID": "OFF",
            "ROCPROFSYS_ROCM_DOMAINS": "hip_api,hsa_api,kernel_dispatch,memory_copy",
            "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count,peak_rss",
            "ROCPROFSYS_COUT_OUTPUT": "ON",
            "LD_LIBRARY_PATH": ":".join(filter(None, paths)),
        }
        result = self.run_test(
            mode,
            "vecadd.jl",
            env=julia_environment,
            pre_run_args=[str(rocprof_config.capabilities.julia_exec)],
        )
        self.assert_regex(result, pass_regex=["PASSED!"])

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Lulesh tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.lulesh]

# =============================================================================
# Lulesh fixtures
# =============================================================================


@pytest.fixture
def lulesh_base_env() -> dict[str, str]:
    return {
        "ROCPROFSYS_USE_KOKKOSP": "ON",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
        "ROCPROFSYS_SAMPLING_FREQ": "50",
        "ROCPROFSYS_KOKKOSP_PREFIX": "[kokkos]",
    }


# =============================================================================
# Lulesh tests
# =============================================================================


# TODO: LULESH_USE_HIP does not currently work properly out of the box, tofix
@pytest.mark.mpi_optional("lulesh")
class TestLulesh(RocprofsysTest):
    @pytest.mark.parametrize(
        "mode",
        ["baseline", "sampling", "binary_rewrite", "runtime_instrument", "sys_run"],
    )
    def test(self, mode, lulesh_base_env):
        env = lulesh_base_env.copy()
        env["KOKKOS_TOOLS_LIBS"] = "librocprof-sys-dl.so"
        result = self.run_test(
            mode,
            "lulesh",
            env=env,
            run_args=["-i", "5", "-s", "20", "-p"],
            rewrite_args=["-e", "-v", "2", "--label", "file", "line", "return", "args"],
            runtime_args=[
                "-e",
                "-v",
                "1",
                "--label",
                "file",
                "line",
                "return",
                "args",
                "-ME",
                "lib(gomp|m-)",
            ],
            mpi_ranks=8,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=[r"\|_\[kokkos\] [a-zA-Z]"],
            runtime_pass_regex=[r"\|_\[kokkos\] [a-zA-Z]"],
        )

    @pytest.mark.baseline
    @pytest.mark.parametrize("lib", ["librocprof-sys", "librocprof-sys-dl"])
    def test_baseline_kokkosp(self, lib, lulesh_base_env):
        env = lulesh_base_env.copy()
        env["KOKKOS_TOOLS_LIBS"] = f"{lib}.so"
        result = self.run_test(
            "baseline",
            "lulesh",
            env=env,
            run_args=["-i", "10", "-s", "20", "-p"],
            mpi_ranks=8,
        )
        self.assert_regex(result, pass_regex=[r"\|_\[kokkos\] [a-zA-Z]"])

    @pytest.mark.parametrize(
        "mode", ["sampling", "binary_rewrite", "runtime_instrument", "sys_run"]
    )
    def test_kokkosp(self, mode):
        env = {"ROCPROFSYS_USE_KOKKOSP": "ON"}
        result = self.run_test(
            mode,
            "lulesh",
            env=env,
            run_args=["-i", "10", "-s", "20", "-p"],
            rewrite_args=["-e", "-v", "2"],
            runtime_args=[
                "-e",
                "-v",
                "1",
                "--label",
                "file",
                "line",
                "return",
                "args",
                "-ME",
                "lib(gomp|m-)",
            ],
            mpi_ranks=8,
        )
        self.assert_regex(result)

    @pytest.mark.parametrize(
        "mode", ["sampling", "binary_rewrite", "runtime_instrument", "sys_run"]
    )
    def test_perfetto(self, mode, perfetto_env):
        env = perfetto_env.copy()
        env["ROCPROFSYS_USE_KOKKOSP"] = "OFF"
        result = self.run_test(
            mode,
            "lulesh",
            env=env,
            run_args=["-i", "10", "-s", "20", "-p"],
            rewrite_args=["-e", "-v", "2"],
            runtime_args=[
                "-e",
                "-v",
                "1",
                "-l",
                "--dynamic-callsites",
                "--traps",
                "--allow-overlapping",
                "-ME",
                "libgomp",
            ],
            mpi_ranks=8,
        )
        self.assert_regex(result)

    @pytest.mark.parametrize(
        "mode",
        ["baseline", "sampling", "binary_rewrite", "runtime_instrument", "sys_run"],
    )
    def test_timemory(self, mode, timemory_env):
        env = timemory_env.copy()
        env["ROCPROFSYS_USE_KOKKOSP"] = "OFF"
        result = self.run_test(
            mode,
            "lulesh",
            env=env,
            run_args=["-i", "2", "-s", "20", "-p"],
            rewrite_args=[
                "-e",
                "-v",
                "2",
                "-l",
                "--dynamic-callsites",
                "--traps",
                "--allow-overlapping",
            ],
            runtime_args=[
                "-e",
                "-v",
                "1",
                "-l",
                "--dynamic-callsites",
                "-ME",
                "libgomp",
                "--env",
                "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock peak_rss",
            ],
            mpi_ranks=8,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_fail_regex=["0 instrumented loops in procedure"],
        )

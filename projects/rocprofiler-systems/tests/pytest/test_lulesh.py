# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Lulesh tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

# These binaries link against libgomp
pytestmark = [pytest.mark.lulesh, pytest.mark.openmp]

# =============================================================================
# Lulesh fixtures
# =============================================================================


@pytest.fixture
def lulesh_base_env() -> dict[str, str]:
    env = {
        "ROCPROFSYS_USE_KOKKOSP": "ON",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
        "ROCPROFSYS_SAMPLING_FREQ": "50",
        "ROCPROFSYS_KOKKOSP_PREFIX": "[kokkos]",
    }
    return env


# =============================================================================
# Lulesh tests
# =============================================================================


# TODO: LULESH_USE_HIP does not currently work properly out of the box, tofix
# TODO: LULESH_USE_MPI does not currently work, tofix
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
            binary_rewrite_args=[
                "-e",
                "-v",
                "2",
                "--label",
                "file",
                "line",
                "return",
                "args",
            ],
            runtime_instrument_args=[
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
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_pass_regex=[r"\|_\[kokkos\] [a-zA-Z]"],
            runtime_instrument_pass_regex=[r"\|_\[kokkos\] [a-zA-Z]"],
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
            binary_rewrite_args=["-e", "-v", "2"],
            runtime_instrument_args=[
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
            binary_rewrite_args=["-e", "-v", "2"],
            runtime_instrument_args=[
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
            binary_rewrite_args=[
                "-e",
                "-v",
                "2",
                "-l",
                "--dynamic-callsites",
                "--traps",
                "--allow-overlapping",
            ],
            runtime_instrument_args=[
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
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_fail_regex=["0 instrumented loops in procedure"],
        )


# =============================================================================
# Kokkos deep-copy tracking
# =============================================================================


@pytest.mark.sampling
@pytest.mark.timeout(300)
@pytest.mark.class_name("lulesh-kokkosp-deep-copy")
@pytest.mark.rocpd("lulesh_base_env")
class TestLuleshKokkospDeepCopy(RocprofsysTest):
    """--kokkosp-deep-copy records a region per Kokkos::deep_copy.

    Tests:  the flag turns Kokkos deep copies into their own trace regions.
    Input:  --kokkosp-deep-copy on lulesh -i 5 -s 20 -p, with KOKKOS_TOOLS_LIBS
            pointing at librocprof-sys-dl.so.
    Expect: ROCPROFSYS_KOKKOSP_DEEP_COPY=true and regions ending in
            "[deep_copy]" in both the Perfetto trace and ROCpd.

    lulesh : deep-copy tracking hangs off the
    Kokkos profiling hooks, so a workload that never calls into Kokkos
    leaves the flag a no-op and any assertion on it vacuous.

    The regions are named "<prefix> <dst space> <- <src name> [deep_copy]".
    Match that suffix, not the bare substring "deep_copy": Kokkos labels
    several of its own fences "Kokkos::deep_copy: ... [fence]" and those
    are recorded with the flag off too, so a looser pattern would pass
    against a build that ignored the flag.
    """

    DEEP_COPY_FLAGS = ["--kokkosp-deep-copy"]
    RUN_ARGS = ["-i", "5", "-s", "20", "-p"]

    def test_regions(self, lulesh_base_env, validation_rules_dir):
        env = lulesh_base_env.copy()
        env["KOKKOS_TOOLS_LIBS"] = "librocprof-sys-dl.so"
        result = self.run_test(
            "sampling",
            "lulesh",
            env=env,
            run_args=self.RUN_ARGS,
            sampling_args=self.DEEP_COPY_FLAGS,
        )
        # The "[deep_copy]" regions below are what prove the flag took effect.
        # A pass_regex on the echoed ROCPROFSYS_KOKKOSP_DEEP_COPY would only
        # show the launcher set it before exec.
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Kokkos deep copies recorded in the Perfetto trace",
            label_substrings=["[deep_copy]"],
        )
        self.assert_rocpd(
            result,
            subtest_name="Kokkos deep copies recorded in ROCpd",
            rules_files=[
                validation_rules_dir / "lulesh" / "kokkosp-deep-copy-rules.json"
            ],
        )

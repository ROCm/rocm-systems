# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KFD Event Tests using the unified-memory example.

Validates that KFD page fault, page migration, queue eviction, and
unmap-from-GPU events are correctly captured in both Perfetto traces
and the ROCpd database.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.xnack,
    pytest.mark.hip,
    pytest.mark.kfd,
]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def kfd_environment() -> dict[str, str]:
    """Environment variables for KFD event tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,kfd_events",
        "ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING": "ON",
        "ROCPROFSYS_USE_AMD_SMI": "OFF",
        "HSA_XNACK": "1",
    }


@pytest.fixture
def kfd_rules(validation_rules_dir) -> list[Path]:
    """Get validation rules for KFD event tests."""
    rules_dir = validation_rules_dir / "kfd"
    return [
        rules_dir / "kfd-rules.json",
    ]


# =============================================================================
# KFD Event Tests
# =============================================================================


class TestKFD(RocprofsysTest):
    """KFD event tracing tests using the unified-memory HIP example.

    The unified-memory example exercises managed memory (hipMallocManaged)
    with various access patterns designed to reliably trigger:
      - Page faults (read/write, migrated/updated)
      - Page migrations (prefetch, GPU pagefault, CPU pagefault)
      - Queue evictions (SVM under memory pressure)
      - Unmap-from-GPU events (MMU notify, migrate, unmap from CPU)
    """

    UM_DEFAULT_TEST_PASS_REGEX = ["9 tests completed"]
    UM_ALL_TEST_PASS_REGEX = ["19 tests completed"]

    run_args = ["-s", "32", "-p", "256", "-i", "4"]

    @pytest.mark.timeout(120)
    @pytest.mark.rocpd("kfd_environment")
    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_events(self, mode, kfd_environment, kfd_rules, gpu_info):
        """Run unified-memory and validate KFD events in Perfetto + ROCpd."""
        is_apu = "apu" in gpu_info.categories

        result = self.run_test(
            mode,
            target="unified-memory",
            env=kfd_environment,
            run_args=self.run_args,
            check_target_arch=True,
        )

        self.assert_regex(
            result,
            subtest_name="Unified-memory completion check",
            pass_regex=self.UM_DEFAULT_TEST_PASS_REGEX,
        )

        self.assert_perfetto(
            result,
            subtest_name="Perfetto KFD page fault validation",
            categories=["rocm_kfd_page_fault"],
            print_output=True,
            pass_regex=[r"PAGE_FAULT"],
        )

        if not is_apu:
            self.assert_perfetto(
                result,
                subtest_name="Perfetto KFD page migrate validation",
                categories=["rocm_kfd_page_migrate"],
                print_output=True,
                pass_regex=[r"PAGE_MIGRATE"],
            )

        self.assert_perfetto(
            result,
            subtest_name="Perfetto KFD queue validation",
            categories=["rocm_kfd_queue"],
            print_output=True,
            pass_regex=[r"QUEUE_EVICT_SVM"],
        )

        self.assert_perfetto(
            result,
            subtest_name="Perfetto KFD unmap from GPU validation",
            categories=["rocm_kfd_event_unmap_from_gpu"],
            print_output=True,
            pass_regex=[r"UNMAP_FROM_GPU"],
        )

        self.assert_rocpd(
            result,
            subtest_name="ROCpd KFD event validation",
            rules_files=kfd_rules,
            gpu_category_to_skip=["apu"] if is_apu else None,
        )

    @pytest.mark.timeout(120)
    @pytest.mark.rocpd("kfd_environment")
    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_prefetch_events(self, mode, kfd_environment, kfd_rules, gpu_info):
        """Focused test for prefetch-driven page migrations.

        Uses more prefetch iterations to generate a high volume of
        PAGE_MIGRATE_PREFETCH events.
        """
        is_apu = "apu" in gpu_info.categories
        env = kfd_environment.copy()

        result = self.run_test(
            mode,
            target="unified-memory",
            env=env,
            run_args=["-s", "32", "-p", "256", "-i", "8"],
            check_target_arch=True,
        )

        self.assert_regex(
            result,
            subtest_name="Unified-memory completion check",
            pass_regex=self.UM_DEFAULT_TEST_PASS_REGEX,
        )

        if not is_apu:
            self.assert_perfetto(
                result,
                subtest_name="Perfetto KFD prefetch migration validation",
                categories=["rocm_kfd_page_migrate"],
                print_output=True,
                pass_regex=[r"PAGE_MIGRATE"],
            )

        self.assert_perfetto(
            result,
            subtest_name="Perfetto KFD queue validation",
            categories=["rocm_kfd_queue"],
            print_output=True,
            pass_regex=[r"QUEUE_EVICT_SVM"],
        )

        if not is_apu:
            self.assert_perfetto(
                result,
                subtest_name="Perfetto KFD combined event validation",
                categories=["rocm_kfd_page_fault", "rocm_kfd_page_migrate"],
                print_output=True,
                pass_regex=[r"PAGE_FAULT", r"PAGE_MIGRATE"],
            )
        else:
            self.assert_perfetto(
                result,
                subtest_name="Perfetto KFD combined event validation",
                categories=["rocm_kfd_page_fault"],
                print_output=True,
                pass_regex=[r"PAGE_FAULT"],
            )

        self.assert_rocpd(
            result,
            subtest_name="ROCpd KFD prefetch validation",
            rules_files=kfd_rules,
            gpu_category_to_skip=["apu"] if is_apu else None,
        )

    @pytest.mark.timeout(180)
    @pytest.mark.rocpd("kfd_environment")
    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_memory_pressure(self, mode, kfd_environment, kfd_rules, gpu_info):
        """Stress test with high memory pressure to trigger queue evictions.

        Uses larger pressure allocation to maximize the chance of
        triggering SVM queue eviction and TTM eviction events. The
        unified-memory program auto-scales pressure to at least 25%
        of GPU VRAM (capped at 4 GB).
        """
        is_apu = "apu" in gpu_info.categories
        env = kfd_environment.copy()

        result = self.run_test(
            mode,
            target="unified-memory",
            env=env,
            run_args=["-a", "-s", "64", "-p", "512", "-i", "4"],
            check_target_arch=True,
        )

        self.assert_regex(
            result,
            subtest_name="Unified-memory completion check",
            pass_regex=self.UM_ALL_TEST_PASS_REGEX,
        )

        self.assert_perfetto(
            result,
            subtest_name="Perfetto KFD page fault validation (pressure)",
            categories=["rocm_kfd_page_fault"],
            print_output=True,
            pass_regex=[r"PAGE_FAULT"],
        )

        if not is_apu:
            self.assert_perfetto(
                result,
                subtest_name="Perfetto KFD page migrate validation (pressure)",
                categories=["rocm_kfd_page_migrate"],
                print_output=True,
                pass_regex=[r"PAGE_MIGRATE"],
            )

        self.assert_perfetto(
            result,
            subtest_name="Perfetto KFD queue validation (pressure)",
            categories=["rocm_kfd_queue"],
            print_output=True,
            pass_regex=[r"QUEUE_EVICT_SVM"],
        )

        self.assert_perfetto(
            result,
            subtest_name="Perfetto KFD unmap validation (pressure)",
            categories=["rocm_kfd_event_unmap_from_gpu"],
            print_output=True,
            pass_regex=[r"UNMAP_FROM_GPU"],
        )

        self.assert_rocpd(
            result,
            subtest_name="ROCpd KFD pressure validation",
            rules_files=kfd_rules,
            gpu_category_to_skip=["apu"] if is_apu else None,
        )

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Regression tests for help-output discoverability in rocprof-sys-run /
rocprof-sys-sample. Each test pins a user-facing defect found in the
help-system audit; failures here indicate a regression in the flag /
topic registration contract.

See planning/bugfix-help-discoverability.md for the full plan.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.presets]

TARGETS = [
    pytest.param("rocprof-sys-run", marks=pytest.mark.sys_run, id="run"),
    pytest.param("rocprof-sys-sample", marks=pytest.mark.sampling, id="sample"),
]


# ============================================================================
# --selected-regions belongs to the tracing topic.
# ----------------------------------------------------------------------------
# Its config setting carries {trace, profile, perfetto, rocpd, timemory,
# rocm} categories, but the registration order in argparse.cpp places it
# inside the GENERAL OPTIONS start_group — invisible to --help=tracing
# and --help=rocm.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("help-tracing-topic-coverage")
class TestTracingTopicListsSelectedRegions(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_tracing_topic_lists_selected_regions(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--help=tracing"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--selected-regions"],
        )


# ============================================================================
# --use-amd-smi must surface under the gpu and rocm domain helps.
# ----------------------------------------------------------------------------
# Timemory auto-derives a --<lower-env-name> flag for every registered
# ROCPROFSYS_* setting, so --use-amd-smi=true|false already works at
# runtime and appears under --help=backend. But users searching
# --help=gpu / --help=rocm (the natural paths for "GPU SMI sampling")
# never see it. The fix extends get_domain_help_map() to cross-list the
# flag into gpu/rocm/process domain helps.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("help-use-amd-smi-cross-domain")
class TestUseAmdSmiVisibleUnderGpuDomain(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_gpu_domain_lists_use_amd_smi(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--help=gpu"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--use-amd-smi"],
        )

    @pytest.mark.parametrize("target", TARGETS)
    def test_rocm_domain_lists_use_amd_smi(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--help=rocm"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--use-amd-smi"],
        )


# ============================================================================
# --rocm help text advertises kfd_events.
# ----------------------------------------------------------------------------
# kfd_events is recognized at runtime by
# source/lib/core/rocprofiler-sdk.cpp (ROCm 7.13+) but the user-facing
# --rocm help string in argument_registration.hpp predates the feature
# and never mentions the value.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("help-rocm-lists-kfd-events")
class TestRocmHelpListsKfdEvents(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_rocm_topic_help_mentions_kfd_events(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--help=rocm"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"kfd_events"],
        )

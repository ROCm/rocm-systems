# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for GPU-specific test counter selection.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest
from rocprofsys import GPUInfo

pytestmark = [pytest.mark.pytest_impl]


@pytest.mark.class_name("gpu-info")
class TestGPUInfo(RocprofsysTest):
    def test_gfx1250_uses_gfx1250_counter_set(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx1250"],
            device_count=1,
            categories={"instinct"},
        )

        assert (
            gpu_info.rocm_events_for_test
            == "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TX_VCA_VCA_BUSY"
        )
        assert gpu_info.counter_names == [
            "GRBM_COUNT",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "TX_VCA_VCA_BUSY",
        ]
        assert gpu_info.expected_counter_files == [
            "rocprof-device-[0-9]-GRBM_COUNT.txt",
            "rocprof-device-[0-9]-SQ_WAVES.txt",
            "rocprof-device-[0-9]-SQ_INSTS_VALU.txt",
            "rocprof-device-[0-9]-TX_VCA_VCA_BUSY.txt",
        ]

    def test_mi300_and_later_keep_ta_ta_busy(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx942"],
            device_count=1,
            categories={"instinct"},
        )

        assert (
            gpu_info.rocm_events_for_test
            == "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY"
        )
        assert gpu_info.counter_names == [
            "GRBM_COUNT",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "TA_TA_BUSY",
        ]

    def test_non_mi300_non_gfx1250_keep_single_counter(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx1201"],
            device_count=1,
            categories={"radeon"},
        )

        assert gpu_info.rocm_events_for_test == "SQ_WAVES"
        assert gpu_info.counter_names == ["SQ_WAVES"]

#!/usr/bin/env python3
"""
Unit tests for rocpd/tracelens_port.py.

Run:
    ROCPD_SYS=/opt/rocm-7.2.0/lib/python3.12/site-packages
    ROCPD_SRC=/home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python
    PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}" pytest --noconftest test_tracelens_port.py -v
"""

import json
from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_cursor(rows):
    """Return a mock cursor whose .fetchall() returns rows."""
    cur = MagicMock()
    cur.fetchall.return_value = rows
    return cur


def _mock_conn():
    return MagicMock()


# ===========================================================================
# Task 1 Tests: pure functions (no DB)
# ===========================================================================

class TestMergeIntervals:
    def test_empty(self):
        from rocpd.tracelens_port import _merge_intervals
        assert _merge_intervals([]) == []

    def test_single(self):
        from rocpd.tracelens_port import _merge_intervals
        assert _merge_intervals([(0, 100)]) == [(0, 100)]

    def test_non_overlapping(self):
        from rocpd.tracelens_port import _merge_intervals
        result = _merge_intervals([(0, 50), (100, 150)])
        assert result == [(0, 50), (100, 150)]

    def test_overlapping(self):
        from rocpd.tracelens_port import _merge_intervals
        result = _merge_intervals([(0, 100), (50, 150)])
        assert result == [(0, 150)]

    def test_adjacent(self):
        from rocpd.tracelens_port import _merge_intervals
        result = _merge_intervals([(0, 100), (100, 200)])
        assert result == [(0, 200)]

    def test_contained(self):
        from rocpd.tracelens_port import _merge_intervals
        result = _merge_intervals([(0, 200), (50, 100)])
        assert result == [(0, 200)]

    def test_unsorted_input(self):
        from rocpd.tracelens_port import _merge_intervals
        result = _merge_intervals([(100, 200), (0, 50)])
        assert result == [(0, 50), (100, 200)]


class TestCategorizeKernelName:
    def test_gemm(self):
        from rocpd.tracelens_port import categorize_kernel_name
        assert categorize_kernel_name("sgemm_nn_kernel") == "GEMM"
        assert categorize_kernel_name("Cijk_Alik_Bljk_HHS_BH_SRVB") == "GEMM"
        assert categorize_kernel_name("rocblas_gemm_kernel") == "GEMM"

    def test_conv(self):
        from rocpd.tracelens_port import categorize_kernel_name
        assert categorize_kernel_name("conv2d_fwd_kernel") == "CONV"
        assert categorize_kernel_name("implicit_gemm_conv_v4r1") == "CONV"

    def test_sdpa(self):
        from rocpd.tracelens_port import categorize_kernel_name
        assert categorize_kernel_name("flash_attention_fwd") == "SDPA"
        assert categorize_kernel_name("fmha_v2_flash_attn") == "SDPA"
        assert categorize_kernel_name("scaled_dot_product_attention") == "SDPA"

    def test_nccl(self):
        from rocpd.tracelens_port import categorize_kernel_name
        assert categorize_kernel_name("ncclKernel_AllReduce_RING_LL") == "NCCL"
        assert categorize_kernel_name("rccl_AllGather_kernel") == "NCCL"

    def test_elementwise(self):
        from rocpd.tracelens_port import categorize_kernel_name
        assert categorize_kernel_name("vectorized_elementwise_kernel") == "Elementwise"
        assert categorize_kernel_name("gelu_activation_kernel") == "Elementwise"

    def test_normalization(self):
        from rocpd.tracelens_port import categorize_kernel_name
        assert categorize_kernel_name("layer_norm_fwd") == "Normalization"
        assert categorize_kernel_name("rms_norm_kernel") == "Normalization"

    def test_reduction(self):
        from rocpd.tracelens_port import categorize_kernel_name
        assert categorize_kernel_name("reduce_kernel") == "Reduction"
        assert categorize_kernel_name("softmax_fwd") == "Reduction"

    def test_other(self):
        from rocpd.tracelens_port import categorize_kernel_name
        assert categorize_kernel_name("reproducible_dispatch_count") == "Other"
        assert categorize_kernel_name("custom_kernel_xyz") == "Other"

    def test_case_insensitive(self):
        from rocpd.tracelens_port import categorize_kernel_name
        assert categorize_kernel_name("SGEMM_KERNEL") == "GEMM"
        assert categorize_kernel_name("Flash_Attention") == "SDPA"


class TestSubtractIntervals:
    def test_empty_a(self):
        from rocpd.tracelens_port import _subtract_intervals
        assert _subtract_intervals([], [(0, 100)]) == []

    def test_empty_b(self):
        from rocpd.tracelens_port import _subtract_intervals
        result = _subtract_intervals([(0, 100)], [])
        assert result == [(0, 100)]

    def test_b_fully_covers_a(self):
        from rocpd.tracelens_port import _subtract_intervals
        result = _subtract_intervals([(0, 100)], [(0, 100)])
        assert result == []

    def test_b_partially_overlaps_left(self):
        from rocpd.tracelens_port import _subtract_intervals
        # b covers [0,50], a is [0,100] → remaining [50, 100]
        result = _subtract_intervals([(0, 100)], [(0, 50)])
        assert result == [(50, 100)]

    def test_b_partially_overlaps_right(self):
        from rocpd.tracelens_port import _subtract_intervals
        # b covers [50,100], a is [0,100] → remaining [0, 50]
        result = _subtract_intervals([(0, 100)], [(50, 100)])
        assert result == [(0, 50)]

    def test_b_cuts_middle(self):
        from rocpd.tracelens_port import _subtract_intervals
        # b covers [40,60], a is [0,100] → remaining [0,40] and [60,100]
        result = _subtract_intervals([(0, 100)], [(40, 60)])
        assert result == [(0, 40), (60, 100)]

    def test_multiple_a_intervals(self):
        from rocpd.tracelens_port import _subtract_intervals
        # a=[0,50],[100,150], b=[25,125] → [0,25] and [125,150]
        result = _subtract_intervals([(0, 50), (100, 150)], [(25, 125)])
        assert result == [(0, 25), (125, 150)]

    def test_adjacent_boundary(self):
        from rocpd.tracelens_port import _subtract_intervals
        # b ends exactly at a_start → no overlap, a is preserved
        result = _subtract_intervals([(100, 200)], [(0, 100)])
        assert result == [(100, 200)]

    def test_no_overlap(self):
        from rocpd.tracelens_port import _subtract_intervals
        # b is entirely before a
        result = _subtract_intervals([(200, 300)], [(0, 100)])
        assert result == [(200, 300)]

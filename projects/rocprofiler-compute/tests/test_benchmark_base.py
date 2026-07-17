# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the HBM roofline benchmark kernel and dataset sizing."""

import pytest


class TestHbmKernelSource:
    """Verify the HBM kernel source string on gfx9 devices."""

    @pytest.fixture()
    def kernel_src(self):
        try:
            from roofline.benchmark.gfx9 import benchmark_gfx9_base
        except ImportError:
            pytest.skip("roofline benchmark module not importable")

        class FakeChild(benchmark_gfx9_base.Bench_gfx9):
            def __init__(self):
                pass

            def set_kernel_source(self):
                benchmark_gfx9_base.Bench_gfx9.set_kernel_source(self)

        obj = FakeChild()
        obj.set_kernel_source()
        return obj.hbm_bw_src

    def test_kernel_is_read_only(self, kernel_src):
        assert "HBM_bw" in kernel_src
        assert "__builtin_nontemporal_load" in kernel_src

    def test_kernel_uses_uint128(self, kernel_src):
        assert "__uint128_t" in kernel_src

    def test_kernel_is_extern_c(self, kernel_src):
        assert 'extern "C"' in kernel_src

    def test_kernel_has_no_dst_write(self, kernel_src):
        assert "dst[" not in kernel_src
        assert "dst =" not in kernel_src


class TestHbmDatasetSizing:
    """Verify dataset sizing arithmetic matches kernel expectations."""

    WORKGROUP_SIZE = 256
    UNROLL = 16
    ELEM_SIZE = 16  # sizeof(__uint128_t)
    DATASET_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB

    def _calc(self, cus):
        workgroups = 128 * cus
        elems_per_step = workgroups * self.WORKGROUP_SIZE * self.UNROLL
        total_elems = self.DATASET_BYTES // self.ELEM_SIZE
        num_steps = (total_elems + elems_per_step - 1) // elems_per_step
        total_elems = num_steps * elems_per_step
        alloc_bytes = total_elems * self.ELEM_SIZE
        return num_steps, total_elems, alloc_bytes, workgroups

    @pytest.mark.parametrize("cus", [64, 128, 256, 304])
    def test_alloc_at_least_4gb(self, cus):
        _, _, alloc_bytes, _ = self._calc(cus)
        assert alloc_bytes >= self.DATASET_BYTES

    @pytest.mark.parametrize("cus", [64, 128, 256, 304])
    def test_total_elems_divisible_by_step(self, cus):
        num_steps, total_elems, _, workgroups = self._calc(cus)
        elems_per_step = workgroups * self.WORKGROUP_SIZE * self.UNROLL
        assert total_elems == num_steps * elems_per_step

    @pytest.mark.parametrize("cus", [64, 128, 256, 304])
    def test_num_steps_positive(self, cus):
        num_steps, _, _, _ = self._calc(cus)
        assert num_steps > 0

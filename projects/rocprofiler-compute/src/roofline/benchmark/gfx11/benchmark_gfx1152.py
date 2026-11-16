# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1152.py
#
# Benchmarking class for all gfx1152 products
#
# -----------------------------------------------------------------------------

from . import benchmark_gfx11_base


# =============================================================================
# Bench_gfx1152 Class
# =============================================================================
class Bench_gfx1152(benchmark_gfx11_base.Bench_gfx11):
    def __init__(self, device_id: int, cache_sizes: dict, gpu_arch: str) -> None:
        super().__init__(device_id, cache_sizes)

        self.apply_roofline_spec(gpu_arch)

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1152.py
#
# Benchmarking class for all gfx1152 products
# AMD Ryzen AI 7 / AI 5
#
# -----------------------------------------------------------------------------

from . import benchmark_gfx11_base


# =============================================================================
# Bench_gfx1152 Class
# =============================================================================
class Bench_gfx1152(benchmark_gfx11_base.Bench_gfx11):
    def __init__(self, device_id: int, cache_sizes: dict) -> None:
        super().__init__(device_id, cache_sizes)

        self.unsupported_data_types.append("MALL")

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1150.py
#
# Benchmarking class for all gfx1150 products
# AMD Ryzen AI 9
#
# -----------------------------------------------------------------------------

from . import benchmark_gfx11_base


# =============================================================================
# Bench_gfx1150 Class
# =============================================================================
class Bench_gfx1150(benchmark_gfx11_base.Bench_gfx11):
    def __init__(self, device_id: int, cache_sizes: dict) -> None:
        super().__init__(device_id, cache_sizes)

        self.unsupported_data_types = [
            "HBM",
            "MALL",
            "I8",
            "I32",
            "I64",
            "WMMA-F4",
            "WMMA-F6",
            "WMMA-F6F4",
            "WMMA-F8",
            "WMMA-F16",
            "WMMA-BF16",
            "WMMA-F32",
            "WMMA-F64",
            "WMMA-I8",
        ]

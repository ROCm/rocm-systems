# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1201.py
#
# Benchmarking class for gfx1201 (Navi48 / Radeon RX 9070 XT).
#
# -----------------------------------------------------------------------------

from . import benchmark_gfx12_base


# =============================================================================
# Bench_gfx1201 Class
# =============================================================================
class Bench_gfx1201(benchmark_gfx12_base.Bench_gfx12):
    def __init__(self, device_id: int, cache_sizes: dict) -> None:
        super().__init__(device_id, cache_sizes)

        # VALU benchmarks only. WMMA counters are not in the analysis configs
        # yet, matching gfx115x. L1 and MALL benches are not used on Navi48.
        self.unsupported_data_types = [
            "L1",
            "MALL",
            "WMMA-F4",
            "WMMA-F6",
            "WMMA-F6F4",
            "WMMA-MXF8",
            "WMMA-F8",
            "WMMA-F16",
            "WMMA-BF16",
            "WMMA-F32",
            "WMMA-F64",
            "WMMA-I8",
        ]

    def set_kernel_source(self) -> None:
        """Use the shared gfx12 kernels; WMMA sources stay unused."""
        super().set_kernel_source()

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1250.py
#
# Benchmarking class for all gfx1250 products
#
# -----------------------------------------------------------------------------

from . import benchmark_gfx12_base


# =============================================================================
# Bench_gfx1250 Class
# =============================================================================
class Bench_gfx1250(benchmark_gfx12_base.Bench_gfx12):
    def __init__(self, device_id: int, cache_sizes: dict) -> None:
        super().__init__(device_id, cache_sizes)

        self.unsupported_data_types = ["L1", "MALL", "WMMA-F64"]

        self.matrix_ops = {
            "F4": 65536,
            "F6": 65536,
            "F6F4": 65536,  # Mixed precision F6 x F4
            "F8": 32768,
            "F16": 16384,
            "F32": 2048,
            "F64": 0,  # Unsupported
            "BF16": 16384,
            "I8": 32768,
        }

    # -----------------------------------------------------------------------------
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------

    def set_kernel_source(self) -> None:
        # Fill in the generic source kernels contained in the super
        super().set_kernel_source()

        # Cache bandwidth, FLOPs, and Matrix ops benchmarking
        # ----------------------------------------
        # Completed in the super

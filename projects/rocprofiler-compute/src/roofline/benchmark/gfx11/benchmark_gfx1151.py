# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1151.py
#
# Benchmarking class for all gfx1151 products
# Strix Halo
#
# -----------------------------------------------------------------------------

from . import benchmark_gfx11_base


# =============================================================================
# Bench_gfx1151 Class
# =============================================================================
class Bench_gfx1151(benchmark_gfx11_base.Bench_gfx11):
    def __init__(self, device_id: int) -> None:
        # APU does not have concept of MCD like dGPU,
        # mark value as 1 to signify unified IOD
        self.mcd_count = 1

        super().__init__(device_id)

        self.unsupported_data_types = [
            "HBM",
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

        self.matrix_ops = {}


    # -----------------------------------------------------------------------------
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------

    def set_kernel_source(self) -> None:
        # Fill in the generic source kernels contained in the super
        super().set_kernel_source()

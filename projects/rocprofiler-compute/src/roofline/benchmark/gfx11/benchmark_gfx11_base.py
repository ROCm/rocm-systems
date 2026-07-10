# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_base_gfx11.py
#
# Benchmarking base class for all gfx11-based products.
#
# -----------------------------------------------------------------------------

from .. import benchmark_base


# =============================================================================
# Bench_gfx11 Class (ABSTRACT)
# =============================================================================
class Bench_gfx11(benchmark_base.Bench_base):
    def __init__(self, device_id: int, cache_sizes: dict) -> None:
        super().__init__(device_id, cache_sizes)

        # TODO: there is potential wavefront size could be set to 64,
        # but default for gfx11 is 32
        self.WAVEFRONT_SIZE = 32
        self.MATRIX_OPS_TYPE = "WMMA"

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

        self.matrix_kernel_selector = {}
        self.matrix_ops = {}

        self.tests = {
            "HBM": super().hbm_bw_benchmark,
            "MALL": super().mall_bw_bench,
            "L2": super().l2_bw_bench,
            "L1": super().l1_bw_bench,
            "L0": super().l0_bw_bench,
            "LDS": super().lds_bw_benchmark,
            "F16": super().fp16_benchmark,
            "F32": super().fp32_benchmark,
            "F64": super().fp64_benchmark,
            "I8": super().int8_benchmark,
            "I32": super().int32_benchmark,
            "I64": super().int64_benchmark,
            "WMMA-F4": super().matrix_f4_bench,
            "WMMA-F6": super().matrix_f6_bench,
            "WMMA-F6F4": super().matrix_f6f4_bench,
            "WMMA-F8": super().matrix_f8_bench,
            "WMMA-F16": super().matrix_f16_bench,
            "WMMA-BF16": super().matrix_bf16_bench,
            "WMMA-F32": super().matrix_f32_bench,
            "WMMA-F64": super().matrix_f64_bench,
            "WMMA-I8": super().matrix_i8_bench,
        }

        self.csv_cols_map = {
            "HBM": "HBMBw",
            "MALL": "MALLBw",
            "L2": "L2Bw",
            "L1": "L1Bw",
            "L0": "L0Bw",
            "LDS": "LDSBw",
            "F16": "FP16Flops",
            "F32": "FP32Flops",
            "F64": "FP64Flops",
            "I8": "I8Ops",
            "I32": "I32Ops",
            "I64": "I64Ops",
            "WMMA-F4": "WMMAF4Flops",
            "WMMA-F6": "WMMAF6Flops",
            "WMMA-F6F4": "WMMAF6F4Flops",
            "WMMA-F8": "WMMAF8Flops",
            "WMMA-F16": "WMMAF16Flops",
            "WMMA-BF16": "WMMABF16Flops",
            "WMMA-F32": "WMMAF32Flops",
            "WMMA-F64": "WMMAF64Flops",
            "WMMA-I8": "WMMAI8Ops",
        }

    # -----------------------------------------------------------------------------
    # Helper Methods and Classes
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------
    def set_kernel_source(self) -> None:
        # Fill in the generic source kernels contained in the super
        super().set_kernel_source()

        # Cache bandwidth and FLOPs benchmarking
        # ----------------------------------------
        # All other cache and FLOPs definitions are completed in the Bench_base
        # class set_kernel_source()
        # HBM Bandwidth benchmark
        self.hbm_bw_src = """"""

        # Matrix operations
        # ----------------------------------------
        # Kernels need arch-specific definitions or are unsupported by the hardware

        self.matrix_f64_src = """"""
        self.matrix_f32_src = """"""
        self.matrix_f16_src = """"""
        self.matrix_bf16_src = """"""
        self.matrix_i8_src = """"""
        self.matrix_f8_src = """"""
        self.matrix_f8f6f4_src = """"""

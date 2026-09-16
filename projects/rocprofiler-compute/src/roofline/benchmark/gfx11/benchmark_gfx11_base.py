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
            "BF16": super().bf16_benchmark,
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
            "BF16": "BF16Flops",
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

        # BF16 uses a separate dot-product kernel for VALU flops measurement because:
        # -> __hip_bfloat16 is not a valid ext_vector_type element.
        # -> dot2 and dual-issue dot2 are the only supported VOP* operations on gfx115x,
        # therefore create a custom kernel using dot2 builtin.
        # BF16 precision dot2 VALU ops kernel source should only be used for gfx115x.
        self.bf16_flops_benchmark_src = f"""
        extern "C" __global__ void bf16_dot_flops(__hip_bfloat16 *buf, int count)
        {{
            const int tid = blockDim.x * blockIdx.x + threadIdx.x;

            // Each thread owns 4 BF16 values = 2 v2short pairs for fdot2 operands.
            // Memory layout matches dataset_size = 4 * sizeof(c_short) * threads.
            using v2short = short __attribute__((ext_vector_type(2)));
            v2short *sbuf = (v2short *)buf;
            v2short a = sbuf[tid * 2];
            v2short b = sbuf[tid * 2 + 1];

            float acc0 = 1.0f;
            float acc1 = 2.0f;
            float acc2 = 3.0f;
            float acc3 = 4.0f;

            for (int i = 0; i < count; i++) {{
                for (int j = 0; j < {benchmark_base.VALU_NFMA} / 8; j++) {{

                    // 4 independent fdot2 ops (4 FLOPs each = 16 FLOPs/iteration)
                    acc0 = __builtin_amdgcn_fdot2_f32_bf16(a, b, acc0, false);
                    acc1 = __builtin_amdgcn_fdot2_f32_bf16(a, b, acc1, false);
                    acc2 = __builtin_amdgcn_fdot2_f32_bf16(a, b, acc2, false);
                    acc3 = __builtin_amdgcn_fdot2_f32_bf16(a, b, acc3, false);
                }}
            }}

            if (acc0 != 2 * acc0) buf[tid * 4] = (__hip_bfloat16)acc0;
        }}
        """

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

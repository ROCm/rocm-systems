# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_base_gfx12.py
#
# Benchmarking base class for all gfx12-based products.
#
# -----------------------------------------------------------------------------

from .. import benchmark_base


# =============================================================================
# Bench_gfx12 Class (ABSTRACT)
# =============================================================================
class Bench_gfx12(benchmark_base.Bench_base):
    def __init__(self, device_id: int, cache_sizes: dict) -> None:
        super().__init__(device_id, cache_sizes)

        self.WAVEFRONT_SIZE = 32
        self.MATRIX_OPS_TYPE = "WMMA"

        # gfx1250: 384KB shared LDS/GL0, with max. LDS being 320KB
        # Confirm if the default is 128KB, and if the amd-smi tools can read
        # LDS and GL0 accurately if not the default.

        self.matrix_kernel_selector = {
            "F4": "wmma_f8f6f4<FP4_E2M1>",
            "F6": "wmma_f8f6f4<FP6_E2M3>",
            "F6F4": "wmma_f8f6f4<FP6_FP4_MIXED>",
            "F8": "wmma_f8",
            "F16": "wmma_f16",
            "BF16": "wmma_bf16",
            "F32": "wmma_f32",
            "F64": "wmma_f64",
            "I8": "wmma_i8",
        }

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
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------
    def set_kernel_source(self) -> None:
        # Fill in the generic source kernels contained in the super
        super().set_kernel_source()

        # HBM Bandwidth benchmark
        # ----------------------------------------
        self.hbm_bw_src = """
        extern "C" __global__ void HBM_bw(__uint128_t *src, long numSteps)
        {
            unsigned long offset = (unsigned long)blockIdx.x * blockDim.x
                                   + threadIdx.x;
            const unsigned long stride = (unsigned long)gridDim.x * blockDim.x;
            __uint128_t v = 0;

            #pragma unroll 1
            for (long step = 0; step < numSteps; step++)
            {
                #pragma unroll
                for (int i = 0; i < 16; i++)
                {
                    v |= __builtin_nontemporal_load(&src[offset]);
                    offset += stride;
                }
            }
            if (v == 0) src[0] = v;
        }
        """

        # Cache bandwidth and FLOPs benchmarking
        # ----------------------------------------
        # Completed in the Bench_base class set_kernel_source()

        # Matrix operations
        # ----------------------------------------
        # Kernels need arch-specific definitions or are unsupported by the hardware

        # self.matrix_f64_src = """"""

        self.matrix_f32_src = (
            self.vector_types_src
            + """
            extern "C" __global__ void wmma_f32(int iter, float *dummy)
            {
                // Input: 2 F32 registers
                vec2<float> a;
                a[1] = a[0] = threadIdx.x;

                // Input: negate 'bool' (const int) per matrix
                // 0 -- none, 1 -- neg
                const int A_mod = 0;
                const int result_mod = 0;

                // Input: matrix reuse 'bool' (const int)
                const int A_reuse = 0;

                // Output: 8 F32 registers
                vec8<float> result = {0};

                // gfx1250: V_WMMA_F32_16X16X4_F32 ops: 16x16x4x2 = 2048
                for(int i = 0; i < iter; ++i)
                {
                    result = __builtin_amdgcn_wmma_f32_16x16x4_f32 (A_mod, a, A_mod, a,\
                          result_mod, result, A_reuse, A_reuse);
                }

                if (result[0] != 2*result[0])
                {
                    dummy[0] = result[0];
                }
            }
            """
        )

        self.matrix_f16_src = (
            self.vector_types_src
            + """
            extern "C" __global__ void wmma_f16(int iter, float *dummy)
            {
                // Input: 8 F32 registers (16 x f16)
                vec8<float> a;
                a[0] = a[1] = a[2] = a[3] = a[4] = a[5] = a[6] = a[7] = threadIdx.x;

                // Input: negate 'bool' (const int) per matrix
                // 0 -- none, 1 -- neg
                const int A_mod = 0;
                const int result_mod = 0;

                // Input: matrix reuse 'bool' (const int)
                const int A_reuse = 0;

                // Output: 4 F32 registers (8 x f16)
                vec4<float> result = {0};

                // gfx1250: V_WMMA_F16_16X16X32_F16_w32 ops: 16x16x32x2 = 16384
                for(int i = 0; i < iter; ++i)
                {
                    result = __builtin_amdgcn_wmma_f16_16x16x32_f16 (A_mod, a, A_mod,\
                          a, result_mod, result, A_reuse, A_reuse);
                }

                if (result[0] != 2*result[0])
                {
                    dummy[0] = result[0];
                }
            }
            """
        )

        self.matrix_bf16_src = (
            self.vector_types_src
            + """
            extern "C" __global__ void wmma_bf16(int iter, float *dummy)
            {
                // Input: 8 i32 registers (16 x bf16)
                vec8<int> a;
                a[0] = a[1] = a[2] = a[3] = a[4] = a[5] = a[6] = a[7] =threadIdx.x;

                // Input: negate 'bool' (const int) per matrix
                // 0 -- none, 1 -- neg
                const int A_mod = 0;
                const int result_mod = 0;

                // Input: matrix reuse 'bool' (const int)
                const int A_reuse = 0;

                // Output: 4 i32 registers (8 x bf16)
                vec4<int> result = {0};

                // gfx1250: V_WMMA_BF16_16X16X32_BF16_w32 ops: 16x16x32x2 = 16384
                for(int i = 0; i < iter; ++i)
                {
                    result = __builtin_amdgcn_wmma_bf16_16x16x32_bf16 (A_mod, a, A_mod,\
                          a, result_mod, result, A_reuse, A_reuse);
                }

                if (result[0] != 2*result[0])
                {
                    dummy[0] = result[0];
                }
            }
            """
        )

        self.matrix_i8_src = (
            self.vector_types_src
            + """
            extern "C" __global__ void wmma_i8(int iter, float *dummy)
            {
                // Input: 8 i32 registers
                vec8<int> a;
                a[0] = a[1] = a[2] = a[3] = a[4] = a[5] = a[6] = a[7] =threadIdx.x;

                // Input: negate 'bool' (const int) per matrix
                // 0 -- none, 1 -- neg
                const int A_mod = 0;

                // Input: matrix reuse 'bool' (const int)
                const int A_reuse = 0;

                // Output: 8 i32 registers
                vec8<int> result = {0};

                // gfx1250: V_WMMA_I32_16X16X64_IU8_w32 ops: 16x16x64x2 = 32768
                for(int i = 0; i < iter; ++i)
                {
                    result = __builtin_amdgcn_wmma_i32_16x16x64_iu8 (A_mod, a, A_mod,\
                          a, result, A_reuse, A_reuse);
                }

                if (result[0] != 2*result[0])
                {
                    dummy[0] = result[0];
                }
            }
            """
        )

        self.matrix_f8_src = (
            self.vector_types_src
            + """
            extern "C" __global__ void wmma_f8(int iter, float *dummy)
            {
                // Input: 8 i32 registers
                vec8<int> a;
                a[0] = a[1] = a[2] = a[3] = a[4] = a[5] = a[6] = a[7] =threadIdx.x;

                // Input: matrix mod
                // 0 - none, 1 - neg, 2 - abs, 3 - neg(abs)
                const int result_mod = 0;

                // Input: matrix reuse 'bool' (const int)
                const int A_reuse = 0;

                // Output: 8 F32 registers
                vec8<float> result = {0};

                // gfx1250: V_WMMA_F32_16X16X64_FP8_FP8_w32 ops: 16x16x64x2 = 32768
                for(int i = 0; i < iter; ++i)
                {
                    result = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8 (a, a,\
                          result_mod, result, A_reuse, A_reuse);
                }

                if (result[0] != 2*result[0])
                {
                    dummy[0] = result[0];
                }
            }
            """
        )

        self.matrix_f8f6f4_src = (
            self.vector_types_src
            + """
            #define FP8_E4M3 0
            #define BF8_E5M2 1
            #define FP6_E2M3 2
            #define BF6_E3M2 3
            #define FP4_E2M1 4
            #define FP6_FP4_MIXED 5

            template<int datatype> __global__ void wmma_f8f6f4(int iter, float *dummy)
            {
                // Input: 16 i32 registers
                vec16<int> a;
                a[0] = a[1] = a[2] = a[3] = a[4] = a[5] = a[6] = a[7] = a[8] = \
                    a[9] = a[10] = a[11] = a[12] = a[13] = a[14] = a[15] = threadIdx.x;

                // Input: matrix mod
                // 0 - none, 1 - neg, 2 - abs, 3 - neg(abs)
                const int result_mod = 0;

                // Output: 8 F32 registers
                vec8<float> result = {0};

                // gfx1250: V_WMMA_F32_16X16X128_F8F6F4 ops: 16x16x128x2 = 65536
                switch (datatype)
                {
                    case FP8_E4M3: // fp8 x fp8
                        for(int i = 0; i < iter; ++i)
                        {
                            result = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                0,          //a_fmt
                                a,          //a
                                0,          //b_fmt
                                a,          //b
                                result_mod, //matrix mod
                                result      //result
                            );
                        }
                        break;
                    case BF8_E5M2: // bf8 x bf8
                        for(int i = 0; i < iter; ++i)
                        {
                            result = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                1,
                                a,
                                1,
                                a,
                                result_mod,
                                result
                            );
                        }
                        break;
                    case FP6_E2M3: // fp6 x fp6
                        for(int i = 0; i < iter; ++i)
                        {
                            result = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                2,
                                a,
                                2,
                                a,
                                result_mod,
                                result
                            );
                        }
                        break;
                    case BF6_E3M2: // bf6 x bf6
                        for(int i = 0; i < iter; ++i)
                        {
                            result = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                3,
                                a,
                                3,
                                a,
                                result_mod,
                                result
                            );
                        }
                        break;
                    case FP4_E2M1: // fp4 x fp4
                        for(int i = 0; i < iter; ++i)
                        {
                            result = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                4,
                                a,
                                4,
                                a,
                                result_mod,
                                result
                            );
                        }
                        break;
                    case FP6_FP4_MIXED: // fp6 x fp4 (mixed precision)
                        for(int i = 0; i < iter; ++i)
                        {
                            result = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                                2, // FP6_E2M3 for input A
                                a,
                                4, // FP4_E2M1 for input B
                                a,
                                result_mod,
                                result
                            );
                        }
                        break;
                }

                if (result[0] != 2*result[0])
                {
                    dummy[0] = result[0];
                }
            }
            """
        )

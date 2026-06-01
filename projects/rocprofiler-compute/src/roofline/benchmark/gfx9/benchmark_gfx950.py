# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx950.py
#
# Benchmarking class for all gfx950 products
# MI350X, MI355X
#
# -----------------------------------------------------------------------------

from . import benchmark_gfx9_base


# =============================================================================
# Bench_gfx950 Class
# =============================================================================
class Bench_gfx950(benchmark_gfx9_base.Bench_gfx9):
    def __init__(self, device_id: int, cache_sizes: dict) -> None:
        super().__init__(device_id, cache_sizes)

        self.unsupported_data_types = []

        self.matrix_ops = {
            "F4": 131072,
            "F6": 131072,
            "F6F4": 131072,  # Mixed precision F6 x F4
            "F8": 32768,
            "F16": 32768,
            "F32": 4096,
            "BF16": 32768,
            "I8": 65536,
            "F64": 2048,
        }

    # -----------------------------------------------------------------------------
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------

    def set_kernel_source(self) -> None:
        # Fill in the generic source kernels contained in the super
        super().set_kernel_source()

        # Cache bandwidth and FLOPs benchmarking
        # ----------------------------------------
        # Completed in the super

        # Matrix operations
        # ----------------------------------------
        self.matrix_f8_src = (
            self.vector_types_src
            + """
        extern "C" __global__ void mfma_f8(int iter, float *dummy)
        {
            // MI300 series only - note gfx940/gfx941/gfx942 only uses fnuz f8
            long a =  threadIdx.x;

            vec16<float> result = {0};

            for(int i = 0; i < iter; ++i)
            {
                result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(\
                    a, a, result, 0, 0, 0);
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
        extern "C" __global__ void mfma_f16(int iter, float *dummy)
        {
            vec16<float> result = {0};
            vec8<__fp16> a;
            a[7] = a[6] = a[5] = a[4] = a[3] = a[2] = a[1] = a[0] = threadIdx.x;

            for(int i = 0; i < iter; ++i)
            {
                result = __builtin_amdgcn_mfma_f32_32x32x16_f16(a, a, result, 0, 0, 0);
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
        extern "C" __global__ void mfma_bf16(int iter, float *dummy)
        {
            vec16<float> result = {0};
            vec8<short> a;
            a[7] = a[6] = a[5] = a[4] = a[3] = a[2] = a[1] = a[0] = threadIdx.x;

            for(int i = 0; i < iter; ++i)
            {
                result = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, a, result, 0, 0, 0);
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
        extern "C" __global__ void mfma_i8(int iter, float *dummy)
        {
            vec16<int> result = {0};
            vec2<long> a;
            a[1] = a[0] = threadIdx.x;

            for(int i = 0; i < iter; ++i)
            {
                result = __builtin_amdgcn_mfma_i32_32x32x32_i8(a, a, result, 0, 0, 0);
            }

            if (result[0] != 2*result[0])
            {
                dummy[0] = result[0];
            }
        }
        """
        )

        # F4F6 is only supported by MI350 series and above
        self.matrix_f8f6f4_src = (
            self.vector_types_src
            + """
        #define FP8_E4M3 0
        #define BF8_E5M2 1
        #define FP6_E2M3 2
        #define BF6_E3M2 3
        #define FP4_E2M1 4
        #define FP6_FP4_MIXED 5

        template<int datatype> __global__ void mfma_f8f6f4(int iter, float *dummy)
        {
            vec8<int> a;
            a[0] = a[1] = a[2] = a[3] = a[4] = a[5] = a[6] = a[7] = threadIdx.x;

            // Output: 16 F32 registers
            vec16<float> result = {0};

            switch (datatype)
            {
                case FP8_E4M3: // fp8 x fp8
                    for(int i = 0; i < iter; ++i)
                    {
                        result = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                            a,
                            a,
                            result,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0
                        );
                    }
                case BF8_E5M2: // bf8 x bf8
                    for(int i = 0; i < iter; ++i)
                    {
                        result = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                            a,
                            a,
                            result,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0
                        );
                    }
                    break;
                case FP6_E2M3: // fp6 x fp6
                    for(int i = 0; i < iter; ++i)
                    {
                        result = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                            a,
                            a,
                            result,
                            2,
                            2,
                            0,
                            0,
                            0,
                            0
                        );
                    }
                    break;
                case BF6_E3M2: // bf6 x bf6
                    for(int i = 0; i < iter; ++i)
                    {
                        result = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                            a,
                            a,
                            result,
                            3,
                            3,
                            0,
                            0,
                            0,
                            0
                        );
                    }
                    break;
                case FP4_E2M1: // fp4 x fp4
                    for(int i = 0; i < iter; ++i)
                    {
                        result = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                            a,
                            a,
                            result,
                            4,
                            4,
                            0,
                            0,
                            0,
                            0
                        );
                    }
                    break;
                case FP6_FP4_MIXED: // fp6 x fp4 (mixed precision)
                    for(int i = 0; i < iter; ++i)
                    {
                        result = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                            a,
                            a,
                            result,
                            2,  // FP6_E2M3 for input A
                            4,  // FP4_E2M1 for input B
                            0,
                            0,
                            0,
                            0
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

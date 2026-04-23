# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx90a.py
#
# Benchmarking class for all gfx90a products
# MI210, MI250, MI250X
#
# -----------------------------------------------------------------------------

from . import benchmark_gfx9_base


# =============================================================================
# Bench_gfx90a Class
# =============================================================================
class Bench_gfx90a(benchmark_gfx9_base.Bench_gfx9):
    def __init__(self, device_ids: list) -> None:
        super().__init__(device_ids)

        self.unsupported_data_types = [
            "MALL",
            "MFMA-F4",
            "MFMA-F6",
            "MFMA-F6F4",
            "MFMA-F8",
        ]

        self.cache_kernel_selector = {
            "L1": "Cache_bw<float, 16 * 1024, 256>",
            "L2": "Cache_bw<float, 8 * 1024 * 1024, 256>",
            "MALL": "",
        }

        self.matrix_ops = {
            "F4": 0,
            "F6": 0,
            "F6F4": 0,  # Mixed precision F6 x F4
            "F8": 0,
            "F16": 16384,
            "F32": 4096,
            "BF16": 8192,
            "I8": 16384,
            "F64": 2048,
        }

        self.cache_sizes = {"L1": 16 * 1024, "L2": 8 * 1024 * 1024, "MALL": 0}

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
        self.matrix_f16_src = (
            self.vector_types_src
            + """
        extern "C" __global__ void mfma_f16(int iter, float *dummy)
        {
            vec16<float> result = {0};
            vec4<__fp16> a;
            a[3] = a[2] = a[1] = a[0] = threadIdx.x;

            for(int i = 0; i < iter; ++i)
            {
                result = __builtin_amdgcn_mfma_f32_32x32x8f16(a, a, result, 0, 0, 0);
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
            vec2<short> a;
            a[1] = a[0]= threadIdx.x;

            for(int i = 0; i < iter; ++i)
            {
                result = __builtin_amdgcn_mfma_f32_32x32x4bf16(a, a, result, 0, 0, 0);
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
            int a = threadIdx.x;

            for(int i = 0; i < iter; ++i)
            {
                result = __builtin_amdgcn_mfma_i32_32x32x8i8(a, a, result, 0, 0, 0);
            }

            if (result[0] != 2*result[0])
            {
                dummy[0] = result[0];
            }
        }
        """
        )

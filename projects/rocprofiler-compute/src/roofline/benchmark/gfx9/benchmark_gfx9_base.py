# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_base_gfx9.py
#
# Benchmarking base class for all gfx9-based products.
#
# -----------------------------------------------------------------------------

from .. import benchmark_base


# =============================================================================
# Bench_gfx9 Class (ABSTRACT)
# =============================================================================
class Bench_gfx9(benchmark_base.Bench_base):
    def __init__(self, device_id: str) -> None:
        super().__init__(device_id)

        # Must define number of AID per arch
        self.aid_count: int

        self.WAVEFRONT_SIZE = 64
        self.MATRIX_OPS_TYPE = "MFMA"

        self.matrix_kernel_selector = {
            "F4": "mfma_f8f6f4<FP4_E2M1>",
            "F6": "mfma_f8f6f4<FP6_E2M3>",
            "F6F4": "mfma_f8f6f4<FP6_FP4_MIXED>",
            "F8": "mfma_f8",
            "F16": "mfma_f16",
            "BF16": "mfma_bf16",
            "F32": "mfma_f32",
            "F64": "mfma_f64",
            "I8": "mfma_i8",
        }

        self.tests = {
            "HBM": super().hbm_bw_benchmark,
            "MALL": super().mall_bw_bench,
            "L2": super().l2_bw_bench,
            "L1": super().l1_bw_bench,
            "LDS": super().lds_bw_benchmark,
            "F16": super().fp16_benchmark,
            "F32": super().fp32_benchmark,
            "F64": super().fp64_benchmark,
            "I8": super().int8_benchmark,
            "I32": super().int32_benchmark,
            "I64": super().int64_benchmark,
            "MFMA-F4": super().matrix_f4_bench,
            "MFMA-F6": super().matrix_f6_bench,
            "MFMA-F6F4": super().matrix_f6f4_bench,
            "MFMA-F8": super().matrix_f8_bench,
            "MFMA-F16": super().matrix_f16_bench,
            "MFMA-BF16": super().matrix_bf16_bench,
            "MFMA-F32": super().matrix_f32_bench,
            "MFMA-F64": super().matrix_f64_bench,
            "MFMA-I8": super().matrix_i8_bench,
        }

        self.csv_cols_map = {
            "HBM": "HBMBw",
            "MALL": "MALLBw",
            "L2": "L2Bw",
            "L1": "L1Bw",
            "LDS": "LDSBw",
            "F16": "FP16Flops",
            "F32": "FP32Flops",
            "F64": "FP64Flops",
            "I8": "I8Ops",
            "I32": "I32Ops",
            "I64": "I64Ops",
            "MFMA-F4": "MFMAF4Flops",
            "MFMA-F6": "MFMAF6Flops",
            "MFMA-F6F4": "MFMAF6F4Flops",
            "MFMA-F8": "MFMAF8Flops",
            "MFMA-F16": "MFMAF16Flops",
            "MFMA-BF16": "MFMABF16Flops",
            "MFMA-F32": "MFMAF32Flops",
            "MFMA-F64": "MFMAF64Flops",
            "MFMA-I8": "MFMAI8Ops",
        }

    # -----------------------------------------------------------------------------
    # Helper Methods and Classes
    # -----------------------------------------------------------------------------

    def set_cache_sizes(self) -> None:
        from utils.amdsmi_interface import (
            amdsmi_ctx,
            get_gpu_cache_info,
            get_gpu_num_compute_units,
        )

        with amdsmi_ctx():
            cu_count = get_gpu_num_compute_units()
            assert cu_count != -1
            cache_info = get_gpu_cache_info()
            assert cache_info  # cache info must be populated

        self.cache_sizes = {}

        for cache_values in cache_info["cache"]:
            # Cache level is L1 and we are looking for vL1d which means
            # there should be a cache instance per CU available on the GPU
            if (
                cache_values["cache_level"] == 1
                and cache_values["num_cache_instance"] == cu_count
            ):
                self.cache_sizes["L1"] = cache_values["cache_size"] * 1024
            # Cache levels L2 and L3/MALL are shared across all CUs
            # therefore only have one cache instance
            elif cache_values["cache_level"] == 2:
                self.cache_sizes["L2"] = cache_values["cache_size"] * 1024
            elif cache_values["cache_level"] == 3:
                self.cache_sizes["MALL"] = int(
                    cache_values["cache_size"] * 1024 / self.aid_count
                )

    # -----------------------------------------------------------------------------
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------
    def set_kernel_source(self) -> None:
        # Fill in the generic source kernels contained in the super
        super().set_kernel_source()

        # Cache bandwidth and FLOPs benchmarking
        # ----------------------------------------
        # Completed in the Bench_base class set_kernel_source()

        # Matrix operations
        # ----------------------------------------
        # Kernels need arch-specific definitions or are unsupported by the hardware
        self.matrix_f16_src = """"""
        self.matrix_bf16_src = """"""
        self.matrix_i8_src = """"""
        self.matrix_f8f6f4_src = """"""
        self.matrix_f8_src = """"""

        self.matrix_f32_src = (
            self.vector_types_src
            + """
            extern "C" __global__ void mfma_f32(int iter, float *dummy)
            {
                float a = threadIdx.x;
                vec16<float> result = {0};

                for(int i = 0; i < iter; ++i)
                {
                    result = __builtin_amdgcn_mfma_f32_32x32x2f32(\
                        a, a, result, 0, 0, 0);
                }

                if (result[0] != 2*result[0])
                {
                    dummy[0] = result[0];
                }
            }
            """
        )

        self.matrix_f64_src = (
            self.vector_types_src
            + """
        extern "C" __global__ void mfma_f64(int iter, float *dummy)
        {
            double a =  threadIdx.x;

            vec4<double> result = {0};

            for(int i = 0; i < iter; ++i)
            {
                result = __builtin_amdgcn_mfma_f64_16x16x4f64(a, a, result, 0, 0, 0);
            }

            if (result[0] != 2*result[0])
            {
                dummy[0] = result[0];
            }
        }
        """
        )

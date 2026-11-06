# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1201.py
#
# Benchmarking class for gfx1201 products: Navi 48 RDNA4 dGPUs such as the
# AMD Radeon RX 9070 XT and Radeon AI PRO R9700.
#
# Navi 48: 64 CUs, 8 MB L2, 64 MB Infinity Cache, DRAM (GDDR6, 640 GB/s).
#
# The gfx12 base class targets gfx1250, whose WMMA builtins (16x16x32,
# 16x16x64, 16x16x128) do not exist on RDNA4. This class therefore replaces the
# matrix kernel sources with the wave32 16x16x16 gfx12 builtins and narrows the
# supported precisions to what RDNA4 actually implements.
# -----------------------------------------------------------------------------

from ctypes import c_int64

import utils.hip_interface as hip

from .. import benchmark_base
from . import benchmark_gfx12_base

# One wmma.16x16x16 issues 16*16*16 = 4096 MACs = 8192 FLOPs (multiply + add).
# Each kernel below runs WMMA_ACC independent accumulator chains per iteration so
# the WMMA unit stays saturated and the measured ceiling is stable.
WMMA_ACC = 8
WMMA_FLOPS_PER_INSTRUCTION = 8192

# Navi 48 Infinity Cache capacity (public RDNA 4 docs), used when AMD-SMI reports
# no level-3 cache.
INFINITY_CACHE_BYTES = 64 * 1024 * 1024

# The Infinity Cache benchmark streams half the Infinity Cache capacity: four times
# the 8 MB L2, so repeated passes miss in L2 and hit in the Infinity Cache.
MALL_WORKING_SET_DIVISOR = 2
MALL_WORKGROUPS_PER_CU = 4
MALL_UNROLL = 16
MALL_ELEMENT_BYTES = 16  # sizeof(__uint128_t)
MALL_TRAFFIC_BYTES = 32 * 1024 * 1024 * 1024


# =============================================================================
# Bench_gfx1201 Class
# =============================================================================
class Bench_gfx1201(benchmark_gfx12_base.Bench_gfx12):
    def __init__(self, device_id: int, cache_sizes: dict, gpu_arch: str) -> None:
        super().__init__(device_id, cache_sizes)

        # DRAM is the device-memory roofline test (CSV column DRAMBw).
        # Infinity Cache uses the MALL column. mi_gpu_spec.yaml lists no
        # F4/F6/F32/F64/I8 matrix path, so those tests are marked unsupported
        # by apply_roofline_spec. DRAM keeps HBM's place in the CSV column order.
        ordered_tests = {}
        for name, func in self.tests.items():
            if name == "HBM":
                ordered_tests["DRAM"] = self.dram_bw_benchmark
            else:
                ordered_tests[name] = func
        self.tests = ordered_tests
        self.tests["MALL"] = self.mall_bw_bench
        self.apply_roofline_spec(gpu_arch)

        self.matrix_kernel_selector = {
            "F8": "wmma_fp8",
            "F16": "wmma_f16",
            "BF16": "wmma_bf16",
        }

        # FLOPs per wave per inner iteration, accounting for the WMMA_ACC chains.
        self.matrix_ops = {
            "F8": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "F16": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "BF16": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
        }

    def dram_bw_benchmark(self, device: int) -> float:
        """Measure sustained DRAM read bandwidth for Navi 48.

        The result is stored under the DRAM roofline test (CSV column DRAMBw).
        """
        return self.hbm_bw_benchmark(device, "DRAM_bw", "DRAM")

    # -----------------------------------------------------------------------------
    # Infinity Cache (MALL) bandwidth benchmark
    # -----------------------------------------------------------------------------
    def mall_bw_bench(self, device: int) -> float:
        """Measure Infinity Cache bandwidth over a working set it holds but L2 does
        not."""
        capacity = self.cache_sizes.get("MALL", INFINITY_CACHE_BYTES)
        return self.resident_stream_bw_bench(
            device, "MALL", capacity // MALL_WORKING_SET_DIVISOR
        )

    def resident_stream_bw_bench(
        self, device: int, label: str, working_set_bytes: int
    ) -> float:
        """Measure read bandwidth of repeated grid-stride passes over one buffer.

        Every pass reads each 16-byte element once, so the bandwidth reflects the
        cache level that holds the whole buffer.
        """
        hip.hipSetDevice(device)
        num_experiments = benchmark_base.DEFAULT_NUM_EXPERIMENTS
        workgroup_size = benchmark_base.DEFAULT_WORKGROUP_SIZE

        cus = hip.hipGetDeviceProperties(device).multiProcessorCount
        workgroups = MALL_WORKGROUPS_PER_CU * cus
        step_bytes = workgroups * workgroup_size * MALL_UNROLL * MALL_ELEMENT_BYTES
        num_steps = max(1, working_set_bytes // step_bytes)
        buffer_bytes = num_steps * step_bytes
        num_passes = max(1, MALL_TRAFFIC_BYTES // buffer_bytes)
        total_bytes = num_passes * buffer_bytes

        src = hip.hipMalloc(buffer_bytes)
        dummy = hip.hipMalloc(MALL_ELEMENT_BYTES)
        # Keep the program alive: a temporary Program unloads the module before launch.
        program = self.Program(self.mall_bw_src)
        func = program.get_kernel("MALL_bw")
        args = [src, c_int64(num_steps), num_passes, dummy]

        self.launch_kernel(
            func, [workgroups, 1, 1], [workgroup_size, 1, 1], 0, None, args
        )
        hip.hipDeviceSynchronize()

        samples = self.run_get_samples(
            num_experiments,
            total_bytes,
            func,
            [workgroups, 1, 1],
            [workgroup_size, 1, 1],
            0,
            None,
            args,
        )
        peak = max(samples)

        print(
            f"{label} BW, GPU ID: {device}, workgroupSize:{workgroup_size}, "
            f"workgroups:{workgroups}, experiments:{num_experiments}, "
            f"working set:{buffer_bytes} bytes, passes:{num_passes}, "
            f"traffic:{total_bytes} bytes, duration:{total_bytes / peak / 1e6:.1f} ms, "
            f"peak:{peak:.1f} GB/sec"
        )
        return peak

    # -----------------------------------------------------------------------------
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------
    def set_kernel_source(self) -> None:
        super().set_kernel_source()

        # DRAM bandwidth. Same streaming body as the gfx12 base (a dataset far
        # larger than the 64 MB Infinity Cache), with the entry point renamed.
        # The source stays on hbm_bw_src because hbm_bw_benchmark compiles that
        # attribute. The roofline test key is DRAM (CSV column DRAMBw). Keep
        # this kernel extern "C": the launcher looks the entry point up by the
        # unmangled name "DRAM_bw".
        self.hbm_bw_src = """
        extern "C" __global__ void DRAM_bw(__uint128_t *src, long numSteps)
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

        # Ordinary (temporal) loads, so the Infinity Cache keeps the buffer
        # between passes.
        self.mall_bw_src = """
            extern "C" __global__ void MALL_bw(const __uint128_t *src,
                                               long numSteps, int numPasses,
                                               __uint128_t *dummy)
            {
                const unsigned long start =
                    (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned long stride =
                    (unsigned long)gridDim.x * blockDim.x;
                __uint128_t v = 0;

                #pragma unroll 1
                for (int pass = 0; pass < numPasses; pass++)
                {
                    unsigned long offset = start;
                    #pragma unroll 1
                    for (long step = 0; step < numSteps; step++)
                    {
                        #pragma unroll
                        for (int i = 0; i < 16; i++)
                        {
                            v ^= src[offset];
                            offset += stride;
                        }
                    }
                }
                if (v == 0) dummy[0] = v;
            }
            """

        # Matrix paths absent on RDNA4; blanked so a stray call fails loudly at
        # compile time rather than silently benchmarking a gfx1250 kernel.
        self.matrix_f32_src = ""
        self.matrix_f64_src = ""
        self.matrix_i8_src = ""
        self.matrix_f8f6f4_src = ""

        self.matrix_f16_src = """
            typedef __fp16 half8 __attribute__((ext_vector_type(8)));
            typedef float  float8 __attribute__((ext_vector_type(8)));
            extern "C" __global__ void wmma_f16(int iter, float *dummy)
            {
                half8 a, b;
                for (int i = 0; i < 8; ++i) {
                    a[i] = (__fp16)(threadIdx.x + i);
                    b[i] = (__fp16)(threadIdx.x - i);
                }
                float8 c0={0},c1={0},c2={0},c3={0},c4={0},c5={0},c6={0},c7={0};
                #pragma unroll 1
                for (int i = 0; i < iter; ++i) {
                    c0=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c0);
                    c1=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c1);
                    c2=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c2);
                    c3=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c3);
                    c4=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c4);
                    c5=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c5);
                    c6=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c6);
                    c7=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c7);
                }
                float8 s = c0+c1+c2+c3+c4+c5+c6+c7;
                if (s[0] != 2 * s[0]) dummy[0] = s[0];
            }
            """

        self.matrix_bf16_src = """
            typedef short short8 __attribute__((ext_vector_type(8)));
            typedef float float8 __attribute__((ext_vector_type(8)));
            extern "C" __global__ void wmma_bf16(int iter, float *dummy)
            {
                short8 a, b;
                for (int i = 0; i < 8; ++i) {
                    a[i] = (short)(threadIdx.x + i);
                    b[i] = (short)(threadIdx.x - i);
                }
                float8 c0={0},c1={0},c2={0},c3={0},c4={0},c5={0},c6={0},c7={0};
                #pragma unroll 1
                for (int i = 0; i < iter; ++i) {
                    c0=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c0);
                    c1=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c1);
                    c2=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c2);
                    c3=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c3);
                    c4=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c4);
                    c5=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c5);
                    c6=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c6);
                    c7=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c7);
                }
                float8 s = c0+c1+c2+c3+c4+c5+c6+c7;
                if (s[0] != 2 * s[0]) dummy[0] = s[0];
            }
            """

        # FP8 (e4m3) inputs packed as 2x int32 per lane (8 fp8 bytes).
        self.matrix_f8_src = """
            typedef int   int2v  __attribute__((ext_vector_type(2)));
            typedef float float8 __attribute__((ext_vector_type(8)));
            extern "C" __global__ void wmma_fp8(int iter, float *dummy)
            {
                int2v a = {(int)threadIdx.x + 1, (int)threadIdx.x + 2};
                int2v b = {(int)threadIdx.x + 3, (int)threadIdx.x + 4};
                float8 c0={0},c1={0},c2={0},c3={0},c4={0},c5={0},c6={0},c7={0};
                #pragma unroll 1
                for (int i = 0; i < iter; ++i) {
                    c0=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c0);
                    c1=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c1);
                    c2=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c2);
                    c3=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c3);
                    c4=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c4);
                    c5=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c5);
                    c6=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c6);
                    c7=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c7);
                }
                float8 s = c0+c1+c2+c3+c4+c5+c6+c7;
                if (s[0] != 2 * s[0]) dummy[0] = s[0];
            }
            """

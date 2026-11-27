# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1201.py
#
# Benchmarking class for gfx1201 products: Navi 48 (RDNA4).
#
# The gfx1250 WMMA builtins are replaced here with the wave32 16x16x16
# builtins this class supplies.
# -----------------------------------------------------------------------------

import utils.hip_interface as hip

from .. import benchmark_base
from . import benchmark_gfx12_base

# The L0 test reads one GL0-sized buffer (32 KB) of 16-byte elements. The
# traffic is large enough that kernel launch is a small part of the timed
# region: 8 GB at the first reading lasted under 2 ms.
L0_ELEMENT_BYTES = 16
L0_WORKING_SET_BYTES = 32 * 1024
L0_TRAFFIC_BYTES = 64 * 1024 * 1024 * 1024
L0_WORKGROUPS_PER_CU = 128

# One 16x16x16 WMMA does 16**3 multiply-adds, which are 2 FLOPs each. Each
# kernel below runs WMMA_ACC independent accumulator chains per iteration so
# the WMMA unit stays saturated and the measured ceiling is stable.
WMMA_ACC = 8
WMMA_FLOPS_PER_INSTRUCTION = 2 * 16**3


# =============================================================================
# Bench_gfx1201 Class
# =============================================================================
class Bench_gfx1201(benchmark_gfx12_base.Bench_gfx12):
    DEVICE_MEMORY_LABEL = "VRAM"

    def __init__(self, device_id: int, cache_sizes: dict, gpu_arch: str) -> None:
        super().__init__(device_id, cache_sizes)

        # VRAM takes the place of the gfx12 base's HBM test, so VRAMBw stays
        # the first roofline.csv column.
        self.tests = {
            ("VRAM" if name == "HBM" else name): func
            for name, func in self.tests.items()
        }
        # The gfx12 base bound the generic cache test to L0.
        self.tests["L0"] = self.l0_bw_bench
        self.matrix_kernel_selector = {
            "F8": "wmma_fp8",
            "F16": "wmma_f16",
            "BF16": "wmma_bf16",
        }
        # Tests that mi_gpu_spec.yaml does not list for gpu_arch are skipped
        # and written as 0.
        self.apply_roofline_spec(gpu_arch)

        # FLOPs per wave per inner iteration, accounting for the WMMA_ACC chains.
        self.matrix_ops = {
            "F8": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "F16": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "BF16": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
        }

    # -----------------------------------------------------------------------------
    # Benchmarking kernel methods
    # -----------------------------------------------------------------------------
    def l0_bw_bench(self, device: int) -> float:
        """Measure GL0 read bandwidth with repeated reads of one GL0-sized buffer."""
        hip.hipSetDevice(device)
        num_experiments = benchmark_base.DEFAULT_NUM_EXPERIMENTS
        workgroup_size = benchmark_base.DEFAULT_WORKGROUP_SIZE
        cus = hip.hipGetDeviceProperties(device).multiProcessorCount
        workgroups = L0_WORKGROUPS_PER_CU * cus
        per_pass_bytes = workgroups * L0_WORKING_SET_BYTES
        iters = max(1, L0_TRAFFIC_BYTES // per_pass_bytes)
        total_bytes = iters * per_pass_bytes

        src = hip.hipMalloc(L0_WORKING_SET_BYTES)
        dummy = hip.hipMalloc(L0_ELEMENT_BYTES)
        program = self.Program(self.l0_bw_src)
        func = program.get_kernel("cache_l0_bw")
        args = [src, L0_WORKING_SET_BYTES // L0_ELEMENT_BYTES, iters, dummy]
        grid = [workgroups, 1, 1]
        block = [workgroup_size, 1, 1]

        self.launch_kernel(func, grid, block, 0, None, args)
        hip.hipDeviceSynchronize()

        samples = self.run_get_samples(
            num_experiments, total_bytes, func, grid, block, 0, None, args
        )
        peak = max(samples)

        print(
            f"L0 BW, GPU ID: {device}, workgroupSize:{workgroup_size}, "
            f"workgroups:{workgroups}, experiments:{num_experiments}, "
            f"working set:{L0_WORKING_SET_BYTES} bytes, iters:{iters}, "
            f"traffic:{total_bytes} bytes, "
            f"duration:{total_bytes / peak / 1e6:.1f} ms, "
            f"peak:{peak:.1f} GB/sec"
        )
        return peak

    # -----------------------------------------------------------------------------
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------
    def set_kernel_source(self) -> None:
        super().set_kernel_source()

        # 16-byte reads of one buffer. count is a power of two and a multiple
        # of 8 * blockDim.x. Adding iter * step changes which thread reads
        # which element, so each iteration still covers the buffer once and
        # the compiler cannot keep the buffer in registers. The eight
        # accumulators keep eight loads in flight per thread.
        self.l0_bw_src = """
            extern "C" __global__ void cache_l0_bw(
                const __uint128_t *src, int count, int iters, __uint128_t *dummy)
            {
                const int tid = (int)threadIdx.x;
                const int step = (int)blockDim.x;
                const int mask = count - 1;
                __uint128_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                __uint128_t s4 = 0, s5 = 0, s6 = 0, s7 = 0;

                #pragma unroll 1
                for (int iter = 0; iter < iters; ++iter) {
                    const int base = (iter * step) & mask;
                    #pragma unroll 1
                    for (int element = tid; element < count; element += step * 8) {
                        s0 ^= src[(base + element) & mask];
                        s1 ^= src[(base + element + step) & mask];
                        s2 ^= src[(base + element + step * 2) & mask];
                        s3 ^= src[(base + element + step * 3) & mask];
                        s4 ^= src[(base + element + step * 4) & mask];
                        s5 ^= src[(base + element + step * 5) & mask];
                        s6 ^= src[(base + element + step * 6) & mask];
                        s7 ^= src[(base + element + step * 7) & mask];
                    }
                }
                __uint128_t sink = s0 ^ s1 ^ s2 ^ s3 ^ s4 ^ s5 ^ s6 ^ s7;
                if (sink == 0) dummy[0] = sink;
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

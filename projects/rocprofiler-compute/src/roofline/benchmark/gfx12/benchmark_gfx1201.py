# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1201.py
#
# Benchmarking class for gfx1201 products: Navi 48 RDNA4 dGPUs such as the
# AMD Radeon RX 9070 XT and Radeon AI PRO R9700.
#
# Navi 48: 64 CUs, 8 MB L2, 64 MB Infinity Cache, 640 GB/s of VRAM.
#
# The gfx12 base class targets gfx1250, whose WMMA builtins (16x16x32,
# 16x16x64, 16x16x128) do not exist on RDNA4. This class therefore replaces the
# matrix kernel sources with the wave32 16x16x16 gfx12 builtins and narrows the
# supported precisions to what RDNA4 actually implements.
# -----------------------------------------------------------------------------

from functools import partial

from . import benchmark_gfx12_base
from .benchmark_gfx12_base import ResidentReadRoof

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

# Navi 48 cache pipes from the gfx1201 cache-hierarchy spec.
# GL0 is 32 KB per TCP, 64 TCPs, and the read return is 128 B per clock.
# GL2 is 8 MB across 32 banks, and the GL2-to-GL1 read return is 128 B per
# bank per clock. GL1 is a buffer with no tags and no capacity, so a read
# stream has no GL1-resident roof. The 8 x 128 B GL1 port is narrower than
# the traffic a GL2 hit actually returns, and scoring a GL2-resident kernel
# against it reports more than 100 percent.
GL0_INSTANCES = 64
GL0_READ_BYTES_PER_CLOCK = 128
GL0_CAPACITY_BYTES = 32 * 1024
GL2_BANKS = 32
GL2_HIT_RETURN_BYTES_PER_CLOCK = 128
GL2_CAPACITY_BYTES = 8 * 1024 * 1024

# L0 is one whole TCP. L1 and L2 both miss GL0 and still fit in GL2, so both
# are GL2-hit roofs. L2 is the larger of the two resident sets. The full 8 MB
# L2 does not fit: it misses into the Infinity Cache and measures the fill
# path instead of the GL2 return.
L0_WORKING_SET_BYTES = 32 * 1024
L1_WORKING_SET_BYTES = 1 * 1024 * 1024
L2_WORKING_SET_BYTES = 4 * 1024 * 1024
# On the RX 9070 XT, 8 workgroups per multiprocessor and 4 loads in flight
# reached 95-97 percent of the GL2 return. The same launch only filled about
# 60 percent of the GL0 return. GL0 holds the whole working set, so extra
# waves hide load latency instead of thrashing: 128 workgroups and 8 loads
# in flight reached about 75 percent, and stayed under the 128 B return.
CACHE_WORKGROUPS_PER_CU = 8
CACHE_LOADS_IN_FLIGHT = 4
L0_WORKGROUPS_PER_CU = 128
L0_LOADS_IN_FLIGHT = 8

# GL1 has no cache, so L1 is not an 8 x 128 B GL1 roof: the bytes come back on
# the 32-bank GL2 read return, as they do for L2 over a larger set.
L0_ROOF = ResidentReadRoof(
    L0_WORKING_SET_BYTES,
    GL0_INSTANCES,
    GL0_READ_BYTES_PER_CLOCK,
    "GL0 read return",
    L0_WORKGROUPS_PER_CU,
    L0_LOADS_IN_FLIGHT,
    "cache_l0_bw",
)
L1_ROOF = ResidentReadRoof(
    L1_WORKING_SET_BYTES,
    GL2_BANKS,
    GL2_HIT_RETURN_BYTES_PER_CLOCK,
    "GL2 hit return",
    CACHE_WORKGROUPS_PER_CU,
    CACHE_LOADS_IN_FLIGHT,
    "cache_resident_bw",
)
L2_ROOF = L1_ROOF._replace(working_set_bytes=L2_WORKING_SET_BYTES)

# Navi 48 device memory: 256-bit VRAM interface at 20 Gbps per pin (AMD RX 9070 XT
# spec). A read-only stream cannot exceed the pin rate.
DRAM_BUS_BITS = 256
DRAM_GBPS_PER_PIN = 20


# =============================================================================
# Bench_gfx1201 Class
# =============================================================================
class Bench_gfx1201(benchmark_gfx12_base.Bench_gfx12):
    def __init__(self, device_id: int, cache_sizes: dict, gpu_arch: str) -> None:
        super().__init__(device_id, cache_sizes)

        # VRAM is the device-memory roofline test (CSV column VRAMBw). It runs
        # the DRAM_bw kernel and prints the share of the pin rate it reached.
        # Infinity Cache uses the MALL column. mi_gpu_spec.yaml lists no
        # F4/F6/F32/F64/I8 matrix path, so those tests are marked unsupported
        # by apply_roofline_spec. VRAM keeps HBM's place in the CSV column order.
        dram_bw_benchmark = partial(
            self.dram_pin_rated_bw_benchmark,
            kernel_name="DRAM_bw",
            label="VRAM",
            bus_bits=DRAM_BUS_BITS,
            gbps_per_pin=DRAM_GBPS_PER_PIN,
        )
        ordered_tests = {}
        for name, func in self.tests.items():
            if name == "HBM":
                ordered_tests["VRAM"] = dram_bw_benchmark
            else:
                ordered_tests[name] = func
        self.tests = ordered_tests
        # The gfx12 base stored the shared scalar cache benchmarks. Rebind
        # them: those kernels use the whole reported cache and a 4-byte load,
        # so L2 misses out of the 8 MB GL2 and L0 cannot reach the 128 B return.
        self.tests["L0"] = partial(self.cache_resident_bw, label="L0", roof=L0_ROOF)
        self.tests["L1"] = partial(self.cache_resident_bw, label="L1", roof=L1_ROOF)
        self.tests["L2"] = partial(self.cache_resident_bw, label="L2", roof=L2_ROOF)
        # The Infinity Cache holds a set that L2 does not.
        self.tests["MALL"] = partial(
            self.cache_stream_bw_bench,
            level="MALL",
            default_capacity_bytes=INFINITY_CACHE_BYTES,
            working_set_divisor=MALL_WORKING_SET_DIVISOR,
        )
        # V_WMMA_F32_16X16X16_BF8_BF8 is the same 16x16x16 shape as FP8.
        self.tests["WMMA-BF8"] = self.matrix_bf8_bench
        self.matrix_kernel_selector = {
            "F8": "wmma_fp8",
            "BF8": "wmma_bf8",
            "F16": "wmma_f16",
            "BF16": "wmma_bf16",
        }
        self.apply_roofline_spec(gpu_arch)

        # FLOPs per wave per inner iteration, accounting for the WMMA_ACC chains.
        # BF8 is V_WMMA_F32_16X16X16_BF8_BF8: the same 16x16x16 MAC count as FP8.
        self.matrix_ops = {
            "F8": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "BF8": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "F16": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "BF16": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
        }

    # -----------------------------------------------------------------------------
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------
    def set_kernel_source(self) -> None:
        super().set_kernel_source()

        # 16-byte reads of one buffer. count is a power of two and a multiple
        # of 4 * blockDim.x. Adding iter * step rotates which thread reads
        # which element, so each iteration still covers the buffer once and
        # the address is not invariant. A fixed address lets the compiler
        # keep a small buffer in registers after the first pass. The four
        # accumulators keep four loads in flight per thread. Thread 0 of each
        # workgroup records its clock64 and wall_clock64 spans; clocks keeps
        # the longest of each.
        self.cache_resident_src = """
            extern "C" __global__ void cache_resident_bw(
                const __uint128_t *src, int count, int iters, __uint128_t *dummy,
                unsigned long long *clocks)
            {
                const int tid = (int)threadIdx.x;
                const int step = (int)blockDim.x;
                const int mask = count - 1;
                unsigned long long cycles_start = 0, wall_start = 0;
                if (tid == 0) {
                    cycles_start = clock64();
                    wall_start = wall_clock64();
                }
                __uint128_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

                #pragma unroll 1
                for (int iter = 0; iter < iters; ++iter) {
                    const int base = (iter * step) & mask;
                    #pragma unroll 1
                    for (int element = tid; element < count; element += step * 4) {
                        s0 ^= src[(base + element) & mask];
                        s1 ^= src[(base + element + step) & mask];
                        s2 ^= src[(base + element + step * 2) & mask];
                        s3 ^= src[(base + element + step * 3) & mask];
                    }
                }
                __uint128_t sink = s0 ^ s1 ^ s2 ^ s3;
                if (sink == 0) dummy[0] = sink;
                if (tid == 0) {
                    atomicMax(&clocks[0], clock64() - cycles_start);
                    atomicMax(&clocks[1], wall_clock64() - wall_start);
                }
            }

            // Eight independent loads. The 16-32 KB GL0 set is one inner
            // pass at this width, so a 4-load chain leaves the TCP return idle.
            extern "C" __global__ void cache_l0_bw(
                const __uint128_t *src, int count, int iters, __uint128_t *dummy,
                unsigned long long *clocks)
            {
                const int tid = (int)threadIdx.x;
                const int step = (int)blockDim.x;
                const int mask = count - 1;
                unsigned long long cycles_start = 0, wall_start = 0;
                if (tid == 0) {
                    cycles_start = clock64();
                    wall_start = wall_clock64();
                }
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
                if (tid == 0) {
                    atomicMax(&clocks[0], clock64() - cycles_start);
                    atomicMax(&clocks[1], wall_clock64() - wall_start);
                }
            }
            """

        # DRAM bandwidth. Same streaming body as the gfx12 base (a dataset far
        # larger than the 64 MB Infinity Cache), with the entry point renamed.
        # The source stays on hbm_bw_src because hbm_bw_benchmark compiles that
        # attribute. The roofline test key is VRAM (CSV column VRAMBw). Keep
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

        # BF8 (e5m2) uses the same packing as FP8: 2x int32 per lane.
        # V_WMMA_F32_16X16X16_BF8_BF8, RDNA4 ISA section 7.12.
        self.matrix_bf8_src = """
            typedef int   int2v  __attribute__((ext_vector_type(2)));
            typedef float float8 __attribute__((ext_vector_type(8)));
            extern "C" __global__ void wmma_bf8(int iter, float *dummy)
            {
                int2v a = {(int)threadIdx.x + 1, (int)threadIdx.x + 2};
                int2v b = {(int)threadIdx.x + 3, (int)threadIdx.x + 4};
                float8 c0={0},c1={0},c2={0},c3={0},c4={0},c5={0},c6={0},c7={0};
                #pragma unroll 1
                for (int i = 0; i < iter; ++i) {
                    c0=__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a,b,c0);
                    c1=__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a,b,c1);
                    c2=__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a,b,c2);
                    c3=__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a,b,c3);
                    c4=__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a,b,c4);
                    c5=__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a,b,c5);
                    c6=__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a,b,c6);
                    c7=__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a,b,c7);
                }
                float8 s = c0+c1+c2+c3+c4+c5+c6+c7;
                if (s[0] != 2 * s[0]) dummy[0] = s[0];
            }
            """

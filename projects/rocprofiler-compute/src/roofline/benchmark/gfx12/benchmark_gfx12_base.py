# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_base_gfx12.py
#
# Benchmarking base class for all gfx12-based products.
#
# -----------------------------------------------------------------------------

from ctypes import c_int64, c_uint64, sizeof
from typing import NamedTuple, Tuple

import utils.hip_interface as hip

from .. import benchmark_base

# wall_clock64() on gfx12 counts a constant 100 MHz reference.
WALL_CLOCK_HZ = 100e6

# Cache-resident reads. A child class's cache_resident_src loads 16-byte
# elements. Long enough that kernel launch is a small part of the timed
# region: 8 GB at the first reading lasted under 2 ms.
CACHE_ELEMENT_BYTES = 16
CACHE_TRAFFIC_BYTES = 64 * 1024 * 1024 * 1024

# Resident stream. A child class's mall_bw_src runs grid-stride passes of
# MALL_UNROLL 16-byte loads per step.
MALL_WORKGROUPS_PER_CU = 4
MALL_UNROLL = 16
MALL_ELEMENT_BYTES = 16  # sizeof(__uint128_t)
MALL_TRAFFIC_BYTES = 32 * 1024 * 1024 * 1024


class ResidentReadRoof(NamedTuple):
    """A cache-resident read benchmark and the pipe it is scored against."""

    working_set_bytes: int
    instances: int
    bytes_per_clock: int
    pipe_name: str
    workgroups_per_cu: int
    loads_in_flight: int
    kernel_name: str


def dram_pin_bandwidth_gbps(bus_bits: int, gbps_per_pin: float) -> float:
    """Peak DRAM bandwidth in decimal GB/s from bus width and pin data rate."""
    return bus_bits * gbps_per_pin / 8


def pipe_bandwidth_gbps(instances: int, bytes_per_clock: int, clock_hz: float) -> float:
    """Aggregate pipe bandwidth in decimal GB/s. Matches the benchmark, which
    reports bytes/s divided by 1e9."""
    return instances * bytes_per_clock * clock_hz / 1e9


def percent_of_ceiling(peak: float, ceiling: float) -> str:
    """Format peak as a percentage of ceiling. The ceiling is a diagnostic only,
    so an unusable one, such as a zero clock reading, prints n/a and never stops
    the benchmark."""
    if ceiling <= 0:
        return "n/a"
    return f"{100.0 * peak / ceiling:.1f}%"


def resident_stream_plan(
    compute_units: int, workgroup_size: int, working_set_bytes: int
) -> Tuple[int, int, int, int, int]:
    """Size the grid-stride resident stream and return (workgroups, num_steps,
    buffer_bytes, num_passes, total_bytes). compute_units is HIP's
    multiProcessorCount. The buffer is a whole number of grid steps of at least
    one, and the pass count brings the traffic up to MALL_TRAFFIC_BYTES."""
    workgroups = MALL_WORKGROUPS_PER_CU * compute_units
    step_bytes = workgroups * workgroup_size * MALL_UNROLL * MALL_ELEMENT_BYTES
    num_steps = max(1, working_set_bytes // step_bytes)
    buffer_bytes = num_steps * step_bytes
    num_passes = max(1, MALL_TRAFFIC_BYTES // buffer_bytes)
    return workgroups, num_steps, buffer_bytes, num_passes, num_passes * buffer_bytes


def shader_clock_hz(shader_cycles: int, wall_ticks: int) -> float:
    """Shader clock measured in one workgroup: clock64 cycles per wall second."""
    if wall_ticks <= 0:
        return 0.0
    return shader_cycles * WALL_CLOCK_HZ / wall_ticks


# =============================================================================
# Bench_gfx12 Class (ABSTRACT)
# =============================================================================
class Bench_gfx12(benchmark_base.Bench_base):
    def __init__(self, device_id: int, cache_sizes: dict) -> None:
        super().__init__(device_id, cache_sizes)

        self.WAVEFRONT_SIZE = 32

        # gfx1250: 384KB shared LDS/GL0, with max. LDS being 320KB
        # Confirm if the default is 128KB, and if the amd-smi tools can read
        # LDS and GL0 accurately if not the default.

        self.matrix_kernel_selector = {
            "F4": "wmma_f8f6f4<FP4_E2M1>",
            "F6": "wmma_f8f6f4<FP6_E2M3>",
            "F6F4": "wmma_f8f6f4<FP6_FP4_MIXED>",
            "MXF8": "wmma_f8f6f4<FP8_E4M3>",
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
            "WMMA-MXF8": super().matrix_mxf8_bench,
            "WMMA-F8": super().matrix_f8_bench,
            "WMMA-F16": super().matrix_f16_bench,
            "WMMA-BF16": super().matrix_bf16_bench,
            "WMMA-F32": super().matrix_f32_bench,
            "WMMA-F64": super().matrix_f64_bench,
            "WMMA-I8": super().matrix_i8_bench,
        }

    # -----------------------------------------------------------------------------
    # Device-memory, cache-resident, and resident-stream bandwidth benchmarks
    # -----------------------------------------------------------------------------
    def dram_pin_rated_bw_benchmark(
        self,
        device: int,
        kernel_name: str,
        label: str,
        bus_bits: int,
        gbps_per_pin: float,
    ) -> float:
        """Measure sustained device-memory read bandwidth and print it as a share
        of the DRAM pin rate, which a read-only stream cannot exceed."""
        peak = self.hbm_bw_benchmark(device, kernel_name, label)
        ceiling = dram_pin_bandwidth_gbps(bus_bits, gbps_per_pin)
        print(
            f"DRAM pin-rate reference: {bus_bits}-bit x {gbps_per_pin} Gbps "
            f"= {ceiling:.1f} GB/sec ({percent_of_ceiling(peak, ceiling)})"
        )
        return peak

    def cache_resident_bw(
        self, device: int, label: str, roof: ResidentReadRoof
    ) -> float:
        """Repeated 16-byte reads of one buffer from cache_resident_src, scored
        against one pipe."""
        workgroup_size = benchmark_base.DEFAULT_WORKGROUP_SIZE
        working_set_bytes = roof.working_set_bytes
        element_count = working_set_bytes // CACHE_ELEMENT_BYTES
        loads_per_step = workgroup_size * roof.loads_in_flight
        if (
            working_set_bytes % CACHE_ELEMENT_BYTES != 0
            or element_count < loads_per_step
            or element_count & (element_count - 1) != 0
        ):
            raise ValueError(
                "cache working set must be a power of two of at least "
                f"{loads_per_step * CACHE_ELEMENT_BYTES} bytes, "
                f"got {working_set_bytes}"
            )

        hip.hipSetDevice(device)
        num_experiments = benchmark_base.DEFAULT_NUM_EXPERIMENTS
        cus = hip.hipGetDeviceProperties(device).multiProcessorCount
        workgroups = roof.workgroups_per_cu * cus
        per_pass_bytes = workgroups * working_set_bytes
        iters = max(1, CACHE_TRAFFIC_BYTES // per_pass_bytes)
        total_bytes = iters * per_pass_bytes

        src = hip.hipMalloc(working_set_bytes)
        dummy = hip.hipMalloc(CACHE_ELEMENT_BYTES)
        clocks = hip.hipMalloc(2 * sizeof(c_uint64))
        program = self.Program(self.cache_resident_src)
        func = program.get_kernel(roof.kernel_name)
        args = [src, element_count, iters, dummy, clocks]
        grid = [workgroups, 1, 1]
        block = [workgroup_size, 1, 1]

        self.launch_kernel(func, grid, block, 0, None, args)
        hip.hipDeviceSynchronize()

        samples = []
        for experiment in range(num_experiments):
            samples.append(
                self.timed_clocked_launch(func, grid, block, args, clocks, total_bytes)
            )
            self.show_progress(float(experiment + 1) / num_experiments)
        print()

        peak, clock_hz = max(samples)
        ceiling = pipe_bandwidth_gbps(roof.instances, roof.bytes_per_clock, clock_hz)
        print(
            f"{label} BW, GPU ID: {device}, workgroupSize:{workgroup_size}, "
            f"workgroups:{workgroups}, experiments:{num_experiments}, "
            f"working set:{working_set_bytes} bytes, iters:{iters}, "
            f"traffic:{total_bytes} bytes, "
            f"duration:{total_bytes / peak / 1e6:.1f} ms, "
            f"peak:{peak:.1f} GB/sec, {roof.pipe_name} ceiling:{ceiling:.1f} GB/sec "
            f"({percent_of_ceiling(peak, ceiling)} at {clock_hz / 1e6:.0f} MHz)"
        )
        return peak

    def timed_clocked_launch(
        self,
        func: object,
        grid: list,
        block: list,
        args: list,
        clocks: hip.HIPDeviceMemory,
        total_bytes: int,
    ) -> tuple:
        """One launch. Returns (GB/s, shader clock in Hz measured in-kernel).

        The kernel records the longest workgroup's clock64 and wall_clock64
        spans, so each sample is scored at the clock it actually ran at.
        """
        zeros = (c_uint64 * 2)(0, 0)
        spans = (c_uint64 * 2)()
        hip.hipMemcpyHtoD(clocks, zeros, sizeof(zeros))
        event_start = hip.hipEventCreate()
        event_stop = hip.hipEventCreate()
        hip.hipEventRecord(event_start)
        self.launch_kernel(func, grid, block, 0, None, args)
        hip.hipEventRecord(event_stop)
        hip.hipDeviceSynchronize()
        event_ms = hip.hipEventElapsedTime(event_start, event_stop)
        hip.hipMemcpyDtoH(spans, clocks, sizeof(spans))
        return (
            float(total_bytes) / event_ms / 1e6,
            shader_clock_hz(spans[0], spans[1]),
        )

    def cache_stream_bw_bench(
        self,
        device: int,
        level: str,
        default_capacity_bytes: int,
        working_set_divisor: int,
    ) -> float:
        """Stream a fraction of one cache level's capacity. The capacity is the
        one AMD-SMI reports, or default_capacity_bytes when it reports none."""
        capacity = self.cache_sizes.get(level, default_capacity_bytes)
        return self.resident_stream_bw_bench(
            device, level, capacity // working_set_divisor
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
        workgroups, num_steps, buffer_bytes, num_passes, total_bytes = (
            resident_stream_plan(cus, workgroup_size, working_set_bytes)
        )

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

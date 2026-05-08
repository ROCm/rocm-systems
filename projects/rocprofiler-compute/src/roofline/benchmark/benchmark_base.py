# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_base.py
#
# Benchmarking base class for all architectures, on all accelerators.
#
# -----------------------------------------------------------------------------

import csv
import fcntl
import math
from abc import ABC
from collections import namedtuple
from collections.abc import Generator
from contextlib import contextmanager
from ctypes import (
    POINTER,
    byref,
    c_double,
    c_float,
    c_int,
    c_int8,
    c_int32,
    c_int64,
    c_short,
    c_void_p,
    cast,
    sizeof,
)
from pathlib import Path
from typing import Any

import utils.hip_interface as hip
import utils.hiprtc_interface as hiprtc

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

Stats = namedtuple("Stats", ["mean", "stdev", "confidence"])
PerfMetrics = namedtuple("PerfMetrics", ["mean", "low", "high"])

DEFAULT_WORKGROUP_SIZE = 256
DEFAULT_WORKGROUPS = 8192
DEFAULT_THREADS = DEFAULT_WORKGROUP_SIZE * DEFAULT_WORKGROUPS
DEFAULT_NUM_EXPERIMENTS = 100
DEFAULT_NUM_ITERS = 10

# Number of FMA operations per thread iteration in VALU benchmark.
# This controls the compute intensity - higher values stress compute throughput.
VALU_NFMA = 1024


# =============================================================================
# Bench_base Class (ABSTRACT)
# =============================================================================
class Bench_base(ABC):
    def __init__(self, device_id: int, cache_sizes: dict) -> None:
        self.device_id = device_id

        # Arch or hardware-specific variables must be set in child classes
        # self.lds_sizes: dict[str, int]
        self.unsupported_data_types: list[str]
        self.cache_kernel_selector: dict[str, str]
        self.matrix_kernel_selector: dict[str, str]
        self.matrix_ops: dict[str, int]
        self.cache_sizes: dict[str, int]
        self.tests: dict[str, str]
        self.csv_cols_map: dict[str, str]
        self.WAVEFRONT_SIZE: int
        self.MATRIX_OPS_TYPE: str

        self.cache_sizes = cache_sizes

        # Some data types have different rates. Set the number of iterations
        # to keep running time under control.
        self.flops_kernel_iterations = {
            "FP16": 256,
            "FP32": 256,
            "FP64": 128,
            "INT8": 128,
            "INT32": 128,
            "INT64": 64,
        }

        self.flops_kernel_selector = {
            "FP16": [f"flops_benchmark<_Float16, {VALU_NFMA}>", sizeof(c_short)],
            "FP32": [f"flops_benchmark<float, {VALU_NFMA}>", sizeof(c_float)],
            "FP64": [f"flops_benchmark<double, {VALU_NFMA}>", sizeof(c_double)],
            "INT8": [f"flops_benchmark<char, {VALU_NFMA}>", sizeof(c_int8)],
            "INT32": [f"flops_benchmark<int, {VALU_NFMA}>", sizeof(c_int32)],
            "INT64": [f"flops_benchmark<long, {VALU_NFMA}>", sizeof(c_int64)],
        }

        self.vector_types_src = """
            template<typename T, int Rank>
            using vecT = T __attribute__((ext_vector_type(Rank)));

            template<typename T> using vec2 = vecT<T, 2>;
            template<typename T> using vec4 = vecT<T, 4>;
            template<typename T> using vec8 = vecT<T, 8>;
            template<typename T> using vec16 = vecT<T, 16>;
            """

        # Kernel source must be set in child classes
        self.hbm_bw_src: str
        self.mall_bw_src: str
        self.l2_bw_src: str
        self.l1_bw_src: str
        self.lds_bw_src: str
        self.fp16_src: str
        self.fp32_src: str
        self.fp64_src: str
        self.int8_src: str
        self.int32_src: str
        self.int64_src: str
        self.matrix_f4_src: str
        self.matrix_f6_src: str
        self.matrix_f6f4_src: str
        self.matrix_f8_src: str
        self.matrix_f16_src: str
        self.matrix_bf16_src: str
        self.matrix_f32_src: str
        self.matrix_f64_src: str
        self.matrix_i8_src: str
        self.set_kernel_source()
        self.set_cache_kernel_selector()

    # -----------------------------------------------------------------------------
    # Helper Methods and Classes
    # -----------------------------------------------------------------------------

    @contextmanager
    def gpu_benchmark_lock(self, device: int) -> Generator[None, None, None]:
        """Acquire exclusive lock for benchmarking a specific GPU."""
        gpu_uuid = bytes(hip.hipGetDeviceProperties(device).uuid.uuid).hex()

        # Get/create lock directory with sticky bit for multi-user safety
        lock_dir = Path("/tmp/rocprof-compute-benchmark")
        lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            lock_dir.chmod(0o1777)  # rwx for all + sticky bit
        except PermissionError:
            pass  # Already created by another user with correct permissions

        lock_file = lock_dir / f"rocprof-compute-benchmark-{gpu_uuid}.lock"

        with open(lock_file, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                msg = (
                    f"Waiting for GPU {device} (UUID: {gpu_uuid[:8]}...) - "
                    "another rocprof-compute benchmark is in progress..."
                )
                print(msg, flush=True)
                fcntl.flock(f, fcntl.LOCK_EX)  # Blocking wait
                msg = f"Acquired lock for GPU {device}, proceeding with benchmark."
                print(msg, flush=True)
            yield

    def show_progress(self, pct: float) -> None:
        """Displays progress bar in log for the current benchmark."""
        bar_char = "|"
        bar_size = 60

        count = int(bar_size * pct)
        bar = "[" + bar_char * count + " " * (bar_size - count) + "]"

        print(f"\r{int(pct * 100):3d}% {bar}", end="", flush=True)

    def calc_stats(self, samples: list) -> Stats:
        """Returns a named tuple with the mean, std deviation and confidence."""
        mean = sum(samples) / len(samples)

        stdev = 0.0

        for i in range(len(samples)):
            stdev += math.pow(samples[i] - mean, 2)

        stdev = math.sqrt(stdev / len(samples))

        return Stats(mean, stdev, 1.96 * stdev / math.sqrt(len(samples)))

    class Program:
        """Helper class for loading and compiling kernels."""

        def __init__(self, src: str, templates: list[str] = []) -> None:
            self.prog = hiprtc.hiprtcCreateProgram(src, "prog")

            for t in templates:
                hiprtc.hiprtcAddNameExpression(self.prog, t)
            try:
                hiprtc.hiprtcCompileProgram(self.prog)
            except hiprtc.HIPRTCError as e:
                log = hiprtc.hiprtcGetProgramLog(self.prog)
                print(f"Program log: {log}")
                raise e

            self.code = hiprtc.hiprtcGetCode(self.prog)
            self.module = hip.hipModuleLoadData(self.code)

        def get_kernel(self, kernel_name: str) -> POINTER:
            # TODO: Why doesn't hiprtcGetLoweredName work with non-template functions?
            if "<" in kernel_name:
                kernel_name = hiprtc.hiprtcGetLoweredName(self.prog, kernel_name)

            return hip.hipModuleGetFunction(self.module, kernel_name)

    def launch_kernel(
        self,
        func: POINTER,
        grid_size: list[int],
        block_size: list[int],
        shared_mem_size: int,
        stream: POINTER,
        args: list[Any] = [],
    ) -> None:
        """Helper method for launching kernel."""
        # Convert to native types
        args_converted = []
        for arg in args:
            if isinstance(arg, int):
                args_converted.append(c_int(arg))
            elif isinstance(arg, hip.HIPDeviceMemory):
                args_converted.append(arg.ptr)
            else:
                args_converted.append(arg)

        # Convert to void pointers
        normalized = [cast(byref(arg), c_void_p) for arg in args_converted]

        args_ptr = (c_void_p * len(args))(*normalized)

        hip.hipModuleLaunchKernel(
            func,
            grid_size[0],
            grid_size[1],
            grid_size[2],
            block_size[0],
            block_size[1],
            block_size[2],
            shared_mem_size,
            stream,
            args_ptr,
        )

    def get_gfx_arch(self, device: int) -> str:
        """Retrieve the gfx architecture."""
        arch_str = hip.hipGetDeviceProperties(device).gcnArchName

        # Parse out only gfx
        return arch_str.split(":", 1)[0]

    def set_cache_kernel_selector(self) -> None:
        self.cache_kernel_selector = {}

        for level in ["L1", "L2", "MALL"]:
            if level in self.cache_sizes.keys():
                self.cache_kernel_selector[level] = (
                    f"Cache_bw<float, {self.cache_sizes[level]}, 256>"
                )

    def run_get_samples(
        self,
        count: int,
        work_per_kernel: int,
        func: POINTER,
        grid_size: list[int],
        block_size: list[int],
        shared_mem_size: int,
        stream: POINTER,
        args: list[Any] = [],
    ) -> list[float]:
        """Helper method to run a kernel and collect samples."""
        event_start = hip.hipEventCreate()
        event_stop = hip.hipEventCreate()

        samples = []
        for i in range(count):
            hip.hipEventRecord(event_start)
            self.launch_kernel(
                func,
                grid_size,
                block_size,
                shared_mem_size,
                stream,
                args,
            )
            hip.hipEventRecord(event_stop)
            hip.hipDeviceSynchronize()
            self.show_progress(float(i + 1) / count)
            event_ms = hip.hipEventElapsedTime(event_start, event_stop)

            samples.append(float(work_per_kernel) / event_ms / 1e6)

        print()
        return samples

    # -----------------------------------------------------------------------------
    # Benchmarking kernel source
    # -----------------------------------------------------------------------------

    def set_kernel_source(self) -> None:
        # HBM Bandwidth benchmark
        self.hbm_bw_src = """
        template<typename T>
        __global__ void HBM_bw(T *dst, const T *src)
        {
            const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned int tid = threadIdx.x;

            dst[gid] = src[gid];
        }
        """

        # Cache Bandwidth benchmark
        self.cache_bw_src = """
        template <typename T, int cacheSize, int workgroup_size>
        __global__ void Cache_bw(const T *memBlock, T *dummy, int numIter)
        {
        const int thread_id = threadIdx.x;
        constexpr int cache_count = cacheSize / sizeof(T);

        T sink;

        sink = 0;
        for (int iter = 0; iter < numIter; ++iter)
        {
        #pragma unroll 32
            for (int i = 0; i < cache_count; i += workgroup_size)
            {
            // if the size of the memory block is small (e.g., the size
            // of L1), then we need a slightly more complicated index
            // calculation. Otherwise, the compiler holds all the loads
            // in the inner loop in registers upon the first pass of the
            // outer loop, and it doesn't do the loads upon subsequent
            // passes of the outer loop.
            // OTOH, if the size of the memory block is larger (such as L2
            // size), experimentation showed that the overhead of the more
            // complicated index calculation has a noticeable effect on BW,
            // so we use a simpler index expression instead. This works since
            // for larger memory blocks, the compiler cannot hold the loads
            // of the inner loop in registers anymore, as it can with L1-sized
            // buffers.
            if constexpr (cache_count / workgroup_size <= 32)
            {
                sink += memBlock[(thread_id + i + iter) % cache_count];
            }
            else
            {
                sink += memBlock[thread_id + i];
            }
            }
        }

        dummy[thread_id] = sink;
        }
        """

        # LDS Bandwidth benchmark
        self.lds_benchmark_src = """
        extern "C" __global__ void LDS_bw(int numIter, float *dummy)
        {
            const int tid = threadIdx.x;
            __shared__ unsigned char shmem[64];


            if (tid == 0)
            {
                #pragma unroll
                for (int i=0;i<63;i++)
                    shmem[i] = i+1;

                shmem[63] = 0;
            }

            __syncthreads();

            int index = tid;
            #pragma unroll 64
            for(int iter = 0; iter < numIter; iter++)
                index = shmem[index];

            dummy[tid] = (float )index;
        }

        """

        # FLOPs benchmark
        self.flops_benchmark_src = (
            self.vector_types_src
            + """

        template<typename T, int nFMA>
        __global__ void flops_benchmark(T *buf, int count)
        {
            static_assert(nFMA % 4 == 0,\
                  "nFMA must be divisible by 4 for vec4 operations");

            const T k = (T)1.1;

            const int grid_size = gridDim.x * blockDim.x;
            const int tid = blockDim.x * blockIdx.x + threadIdx.x;

            vec4<T>* ptr = (vec4<T>*)buf;

            vec4<T> value0 = ptr[0 * grid_size + tid];

            vec4<T> x0 = {(T)1,(T)2,(T)3,(T)4};

            for(int i = 0; i < count; i++) {
                for(int j = 0; j < nFMA / 4; j++) {

                    // 4 FMA ops
                    x0 = x0 * value0 + k;
                }
            }

            ptr[tid] = x0;
        }
        """
        )

    # -----------------------------------------------------------------------------
    # Benchmarking kernel methods
    # -----------------------------------------------------------------------------

    # HBM bandwidth benchmark
    def hbm_bw_benchmark(self, device: int) -> PerfMetrics:
        num_experiments = DEFAULT_NUM_EXPERIMENTS
        hip.hipSetDevice(device)

        cus = hip.hipGetDeviceProperties(device).multiProcessorCount

        prog = self.Program(self.hbm_bw_src, ["HBM_bw<double>"])
        func = prog.get_kernel("HBM_bw<double>")

        workgroup_size = DEFAULT_WORKGROUP_SIZE
        workgroups_per_cu = 20 * 1024
        workgroups = cus * workgroups_per_cu
        dataset_entries = workgroups * workgroup_size

        d_src = hip.hipMalloc(dataset_entries * sizeof(c_double))
        d_dst = hip.hipMalloc(dataset_entries * sizeof(c_double))

        total_bytes = dataset_entries * sizeof(c_double) * 2

        self.launch_kernel(
            func, [workgroups, 1, 1], [workgroup_size, 1, 1], 0, None, [d_dst, d_src]
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
            [d_dst, d_src],
        )

        stats = self.calc_stats(samples)

        mean = stats.mean
        stdev = stats.stdev

        perf_metrics = PerfMetrics(
            mean, mean - stats.confidence, mean + stats.confidence
        )

        event_ms = total_bytes / mean / 1e6

        print(
            f"HBM BW, GPU ID: {device}, workgroupSize:{workgroup_size}, "
            f"workgroups:{workgroups}, experiments:{num_experiments}, "
            f"traffic:{total_bytes} bytes, duration:{event_ms:.1f} ms, "
            f"mean:{mean:.1f} GB/sec, stdev:{stdev:.1f} GB/sec"
        )

        return perf_metrics

    # Generic cache bandwidth benchmark
    def cache_bw_bench(self, device: int, type: str, iters: int) -> PerfMetrics:
        hip.hipSetDevice(device)

        num_experiments = DEFAULT_NUM_EXPERIMENTS
        workgroup_size = DEFAULT_WORKGROUP_SIZE

        cus = hip.hipGetDeviceProperties(device).multiProcessorCount

        cache_size = self.cache_sizes[type]

        mem_block = hip.hipMalloc(cache_size)
        dummy = hip.hipMalloc(workgroup_size * sizeof(c_float))

        kernel_name = self.cache_kernel_selector[type]
        prog = self.Program(self.cache_bw_src, [kernel_name])
        func = prog.get_kernel(kernel_name)

        workgroups = 128 * cus
        total_bytes = workgroups * iters * cache_size

        self.launch_kernel(
            func,
            [workgroups, 1, 1],
            [workgroup_size, 1, 1],
            0,
            None,
            [mem_block, dummy, iters],
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
            [mem_block, dummy, iters],
        )

        stats = self.calc_stats(samples)
        mean = stats.mean
        stdev = stats.stdev

        perf_metrics = PerfMetrics(
            mean, mean - stats.confidence, mean + stats.confidence
        )

        event_ms = total_bytes / mean / 1e6

        print(
            f"{type} BW, GPU ID: {device}, workgroupSize:{workgroup_size}, "
            f"workgroups:{workgroups}, experiments:{num_experiments}, "
            f"traffic:{total_bytes} bytes, duration:{event_ms:.1f} ms, "
            f"mean:{mean:.1f} GB/sec, stdev:{stdev:1f} GB/sec"
        )

        return perf_metrics

    # MALL cache bandwidth benchmark
    def mall_bw_bench(self, device: int) -> PerfMetrics:
        return self.cache_bw_bench(device, "MALL", 1)

    # L1 cache bandwidth benchmark
    def l1_bw_bench(self, device: int) -> PerfMetrics:
        return self.cache_bw_bench(device, "L1", 100)

    # L2 cache bandwidth benchmark
    def l2_bw_bench(self, device: int) -> PerfMetrics:
        return self.cache_bw_bench(device, "L2", 10)

    # LDS cache bandwidth benchmark
    def lds_bw_benchmark(self, device: int) -> PerfMetrics:
        num_experiments = DEFAULT_NUM_EXPERIMENTS
        workgroup_size = DEFAULT_WORKGROUP_SIZE

        cus = hip.hipGetDeviceProperties(device).multiProcessorCount

        iters = 2000

        workgroups = 128 * cus
        total_bytes = workgroups * workgroup_size * iters * sizeof(c_float)

        dummy = hip.hipMalloc(workgroup_size * sizeof(c_float))

        prog = self.Program(self.lds_benchmark_src)
        func = prog.get_kernel("LDS_bw")

        # Warmup
        self.launch_kernel(
            func, [workgroups, 1, 1], [workgroup_size, 1, 1], 0, None, [iters, dummy]
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
            [iters, dummy],
        )

        stats = self.calc_stats(samples)
        mean = stats.mean
        stdev = stats.stdev

        perf_metrics = PerfMetrics(
            mean, mean - stats.confidence, mean + stats.confidence
        )

        event_ms = total_bytes / mean / 1e6

        print(
            f"LDS BW, GPU ID: {device}, workgroupSize:{workgroup_size}, "
            f"workgroups:{workgroups}, experiments:{num_experiments}, "
            f"traffic:{total_bytes} bytes, duration:{event_ms:.1f} ms, "
            f"mean:{mean:.1f} GB/sec, stdev:{stdev:1f} GB/sec"
        )

        return perf_metrics

    # Generic FLOPs benchmark
    def flops_bench(self, device: int, type: str, unit: str, rate: int) -> PerfMetrics:
        num_experiments = DEFAULT_NUM_EXPERIMENTS
        workgroup_size = DEFAULT_WORKGROUP_SIZE
        cus = hip.hipGetDeviceProperties(device).multiProcessorCount

        workgroups = 128 * cus
        threads = workgroups * workgroup_size

        kernel_name = self.flops_kernel_selector[type][0]
        type_size = self.flops_kernel_selector[type][1]

        # Each thread reads a vec4
        dataset_size = 4 * type_size * threads
        memblock = hip.hipMalloc(dataset_size)

        iterations = self.flops_kernel_iterations[type]
        total_flops = threads * iterations * VALU_NFMA * 2

        prog = self.Program(self.flops_benchmark_src, [kernel_name])

        func = prog.get_kernel(kernel_name)

        # Warmup
        self.launch_kernel(
            func,
            [workgroups, 1, 1],
            [workgroup_size, 1, 1],
            0,
            None,
            [memblock, iterations],
        )
        hip.hipDeviceSynchronize()

        samples = self.run_get_samples(
            num_experiments,
            total_flops,
            func,
            [workgroups, 1, 1],
            [workgroup_size, 1, 1],
            0,
            None,
            [memblock, iterations],
        )

        stats = self.calc_stats(samples)
        mean = stats.mean
        stdev = stats.stdev

        perf_metrics = PerfMetrics(
            mean, mean - stats.confidence, mean + stats.confidence
        )

        event_ms = total_flops / mean / 1e6

        print(
            f"Peak VALU {unit}s ({type}), GPU ID: {device}, "
            f"workgroupSize:{workgroup_size}, "
            f"workgroups:{workgroups}, experiments:{num_experiments}, "
            f"{unit}:{total_flops}, duration:{event_ms:.1f} ms, "
            f"mean:{mean:.1f} {rate}, stdev={stdev:.1f} GFLOPS"
        )

        return perf_metrics

    # Generic matrix operations benchmark
    def matrix_bench(self, device: int, type: str, unit: str, rate: int) -> PerfMetrics:
        experiments = DEFAULT_NUM_EXPERIMENTS
        iters = 2000

        cus = hip.hipGetDeviceProperties(device).multiProcessorCount

        workgroups = 128 * cus
        workgroup_size = DEFAULT_WORKGROUP_SIZE

        total_flops = (
            workgroups
            * workgroup_size
            // self.WAVEFRONT_SIZE
            * iters
            * self.matrix_ops[type]
        )

        dummy = hip.hipMalloc(64 * sizeof(c_float))

        kernel_name = self.matrix_kernel_selector[type]

        if type == "F32":
            src = self.matrix_f32_src
        elif type == "F8":
            src = self.matrix_f8_src
        elif type == "F16":
            src = self.matrix_f16_src
        elif type == "BF16":
            src = self.matrix_bf16_src
        elif type == "F64":
            src = self.matrix_f64_src
        elif type == "I8":
            src = self.matrix_i8_src
        else:
            src = self.matrix_f8f6f4_src

        prog = self.Program(src, [kernel_name])
        func = prog.get_kernel(kernel_name)

        samples = self.run_get_samples(
            experiments,
            total_flops,
            func,
            [workgroups, 1, 1],
            [workgroup_size, 1, 1],
            0,
            None,
            [iters, dummy],
        )

        stats = self.calc_stats(samples)
        mean = stats.mean
        stdev = stats.stdev

        perf_metrics = PerfMetrics(
            mean, mean - stats.confidence, mean + stats.confidence
        )

        event_ms = total_flops / mean / 1e6

        print(
            f"Peak {self.MATRIX_OPS_TYPE} {unit}s ({type}), GPU ID: {device}, "
            f"workgroupSize:{workgroup_size}, workgroups:{workgroups}, "
            f"experiments:{experiments}, {unit}:{total_flops}, "
            f"duration:{event_ms:.2f} ms, mean:{mean:.1f} {rate}, "
            f"stdev:{stdev:.1f} GFLOPS"
        )

        return perf_metrics

    def matrix_f32_bench(self, device: int) -> PerfMetrics:
        return self.matrix_bench(device, "F32", "FLOP", "GFLOPS")

    def matrix_f16_bench(self, device: int) -> PerfMetrics:
        return self.matrix_bench(device, "F16", "FLOP", "GFLOPS")

    def matrix_bf16_bench(self, device: int) -> PerfMetrics:
        return self.matrix_bench(device, "BF16", "FLOP", "GFLOPS")

    def matrix_f64_bench(self, device: int) -> PerfMetrics:
        return self.matrix_bench(device, "F64", "FLOP", "GFLOPS")

    def matrix_f8_bench(self, device: int) -> PerfMetrics:
        return self.matrix_bench(device, "F8", "FLOP", "GFLOPS")

    def matrix_i8_bench(self, device: int) -> PerfMetrics:
        return self.matrix_bench(device, "I8", "IOP", "GOPS")

    def matrix_f4_bench(self, device: int) -> PerfMetrics:
        return self.matrix_bench(device, "F4", "FLOP", "GFLOPS")

    def matrix_f6_bench(self, device: int) -> PerfMetrics:
        return self.matrix_bench(device, "F6", "FLOP", "GFLOPS")

    def matrix_f6f4_bench(self, device: int) -> PerfMetrics:
        return self.matrix_bench(device, "F6F4", "FLOP", "GFLOPS")

    def fp16_benchmark(self, device: int) -> PerfMetrics:
        return self.flops_bench(device, "FP16", "FLOP", "GFLOPS")

    def fp32_benchmark(self, device: int) -> PerfMetrics:
        return self.flops_bench(device, "FP32", "FLOP", "GFLOPS")

    def fp64_benchmark(self, device: int) -> PerfMetrics:
        return self.flops_bench(device, "FP64", "FLOP", "GFLOPS")

    def int8_benchmark(self, device: int) -> PerfMetrics:
        return self.flops_bench(device, "INT8", "IOP", "GOPS")

    def int32_benchmark(self, device: int) -> PerfMetrics:
        return self.flops_bench(device, "INT32", "IOP", "GOPS")

    def int64_benchmark(self, device: int) -> PerfMetrics:
        return self.flops_bench(device, "INT64", "IOP", "GOPS")

    def run_benchmark(self, device: int) -> dict[PerfMetrics]:
        """
        Run the roofline tests on the specified device.
        Returns a dictionary of PerfMetrics.
        """
        with self.gpu_benchmark_lock(device):
            metrics_dict = {}

            arch = self.get_gfx_arch(device)
            cus = hip.hipGetDeviceProperties(device).multiProcessorCount

            print(f"GPU Device {device} ({arch}) with {cus} CUs: Profiling...")

            for name, func in self.tests.items():
                if name in self.unsupported_data_types:
                    print(f"Skipping {name}")
                    metrics = PerfMetrics(0, 0, 0)
                else:
                    metrics = func(device)

                metrics_dict[name] = metrics

            print("GPU Benchmarking completed")
            return metrics_dict

    def dump_csv(self, metrics: dict[PerfMetrics], file_path: str) -> None:
        """Generate a csv file containing the collected benchmark metrics."""
        # TODO: Better way to map CSV column names?
        with open(file_path, "w") as f:
            writer = csv.writer(f)

            types = self.csv_cols_map.keys()

            # Write the first row (col names)
            row = ["device"]
            for t in types:
                row.append(self.csv_cols_map[t])
                row.append(self.csv_cols_map[t] + "Low")
                row.append(self.csv_cols_map[t] + "High")

            writer.writerow(row)

            row = [self.device_id]
            for t in types:
                row.append(metrics[t].mean)
                row.append(metrics[t].low)
                row.append(metrics[t].high)

            writer.writerow(row)

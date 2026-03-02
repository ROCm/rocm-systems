##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import csv
import fcntl
import math
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

import hip.hip as hip
import hip.hiprtc as hiprtc

lds_sizes = {
    "gfx908": 64 * 1024,
    "gfx90a": 64 * 1024,
    "gfx940": 64 * 1024,
    "gfx941": 64 * 1024,
    "gfx942": 64 * 1024,
    "gfx950": 64 * 1024,
}

unsupported_data_types = {
    "gfx908": [
        "MALL",
        "MFMA-F4",
        "MFMA-F6",
        "MFMA-F6F4",
        "MFMA-F8",
        "MFMA-F16",
        "MFMA-BF16",
        "MFMA-F64",
        "MFMA-I8",
    ],  # MI100 series
    "gfx90a": ["MALL", "MFMA-F4", "MFMA-F6", "MFMA-F6F4", "MFMA-F8"],  # MI200 series
    "gfx940": ["MFMA-F4", "MFMA-F6", "MFMA-F6F4"],  # MI300A_A0
    "gfx941": ["MFMA-F4", "MFMA-F6", "MFMA-F6F4"],  # MI300X_A0
    "gfx942": ["MFMA-F4", "MFMA-F6", "MFMA-F6F4"],  # MI300A_A1, MI300X_A1, MI308
    "gfx950": [],  # MI350, MI355
}

cache_kernel_selector = {
    "L1": {
        "gfx908": "Cache_bw<float, 16 * 1024, 256>",
        "gfx90a": "Cache_bw<float, 16 * 1024, 256>",
        "gfx940": "Cache_bw<float, 32 * 1024, 256>",
        "gfx941": "Cache_bw<float, 32 * 1024, 256>",
        "gfx942": "Cache_bw<float, 32 * 1024, 256>",
        "gfx950": "Cache_bw<float, 32 * 1024, 256>",
    },
    "L2": {
        "gfx908": "Cache_bw<float, 8 * 1024 * 1024, 256>",
        "gfx90a": "Cache_bw<float, 8 * 1024 * 1024, 256>",
        "gfx940": "Cache_bw<float, 4 * 1024 * 1024, 256>",
        "gfx941": "Cache_bw<float, 4 * 1024 * 1024, 256>",
        "gfx942": "Cache_bw<float, 4 * 1024 * 1024, 256>",
        "gfx950": "Cache_bw<float, 4 * 1024 * 1024, 256>",
    },
    "MALL": {
        "gfx940": "Cache_bw<float, 64 * 1024 * 1024, 256>",
        "gfx941": "Cache_bw<float, 64 * 1024 * 1024, 256>",
        "gfx942": "Cache_bw<float, 64 * 1024 * 1024, 256>",
        "gfx950": "Cache_bw<float, 64 * 1024 * 1024, 256>",
    },
}

mfma_kernel_selector = {
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

# Number of FMA operations per thread iteration in VALU benchmark.
# This controls the compute intensity - higher values stress compute throughput.
VALU_NFMA = 1024

# Some data types have different rates. Set the number of iterations
# to keep running time under control.
flops_kernel_iterations = {
    "FP16": 256,
    "FP32": 256,
    "FP64": 128,
    "INT8": 128,
    "INT32": 128,
    "INT64": 64,
}

flops_kernel_selector = {
    "FP16": [f"flops_benchmark<_Float16, {VALU_NFMA}>", sizeof(c_short)],
    "FP32": [f"flops_benchmark<float, {VALU_NFMA}>", sizeof(c_float)],
    "FP64": [f"flops_benchmark<double, {VALU_NFMA}>", sizeof(c_double)],
    "INT8": [f"flops_benchmark<char, {VALU_NFMA}>", sizeof(c_int8)],
    "INT32": [f"flops_benchmark<int, {VALU_NFMA}>", sizeof(c_int32)],
    "INT64": [f"flops_benchmark<long, {VALU_NFMA}>", sizeof(c_int64)],
}

mfma_ops = {
    "F4": {"gfx950": 131072},
    "F6": {"gfx950": 131072},
    "F6F4": {"gfx950": 131072},  # Mixed precision F6 x F4
    "F8": dict.fromkeys(["gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"], 32768),
    "F16": dict.fromkeys(["gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"], 16384),
    "F32": dict.fromkeys(
        ["gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"], 4096
    ),
    "BF16": dict.fromkeys(["gfx940", "gfx941", "gfx942", "gfx950"], 16384)
    | dict.fromkeys(["gfx90a"], 8192),
    "I8": dict.fromkeys(["gfx940", "gfx941", "gfx942", "gfx950"], 32768)
    | dict.fromkeys(["gfx90a"], 16384),
    "F64": dict.fromkeys(["gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"], 2048),
}

cache_sizes = {
    "L1": {
        "gfx908": 16 * 1024,
        "gfx90a": 16 * 1024,
        "gfx940": 32 * 1024,
        "gfx941": 32 * 1024,
        "gfx942": 32 * 1024,
        "gfx950": 32 * 1024,
    },
    "L2": {
        "gfx908": 8 * 1024 * 1024,
        "gfx90a": 8 * 1024 * 1024,
        "gfx940": 4 * 1024 * 1024,
        "gfx941": 4 * 1024 * 1024,
        "gfx942": 4 * 1024 * 1024,
        "gfx950": 4 * 1024 * 1024,
    },
    "MALL": {
        "gfx940": 64 * 1024 * 1024,
        "gfx941": 64 * 1024 * 1024,
        "gfx942": 64 * 1024 * 1024,
        "gfx950": 64 * 1024 * 1024,
    },
}


Stats = namedtuple("Stats", ["mean", "stdev", "confidence"])
PerfMetrics = namedtuple("PerfMetrics", ["mean", "low", "high"])

DEFAULT_WORKGROUP_SIZE = 256
DEFAULT_WORKGROUPS = 8192
DEFAULT_THREADS = DEFAULT_WORKGROUP_SIZE * DEFAULT_WORKGROUPS
DEFAULT_NUM_EXPERIMENTS = 100
DEFAULT_NUM_ITERS = 10


@contextmanager
def gpu_benchmark_lock(device: int) -> Generator[None, None, None]:
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


def show_progress(pct: float) -> None:
    bar_char = "|"
    bar_size = 60

    count = int(bar_size * pct)
    bar = "[" + bar_char * count + " " * (bar_size - count) + "]"

    print(f"\r{int(pct * 100):3d}% {bar}", end="", flush=True)


# Returns a named tuple with the mean, std deviation and confidence
def calc_stats(samples: list) -> Stats:
    mean = sum(samples) / len(samples)

    stdev = 0.0

    for i in range(len(samples)):
        stdev += math.pow(samples[i] - mean, 2)

    stdev = math.sqrt(stdev / len(samples))

    return Stats(mean, stdev, 1.96 * stdev / math.sqrt(len(samples)))


# Helper class for loading and compiling kerels
class Program:
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


# Helper method for launching kernel
def launch_kernel(
    func: POINTER,
    grid_size: list[int],
    block_size: list[int],
    shared_mem_size: int,
    stream: POINTER,
    args: list[Any] = [],
) -> None:
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


# Retrieve the gfx architecture
def get_gfx_arch(device: int) -> str:
    arch_str = hip.hipGetDeviceProperties(device).gcnArchName

    # Parse out only gfx
    return arch_str.split(":", 1)[0]


# Helper method to run a kernel and collect samples
def run_get_samples(
    count: int,
    work_per_kernel: int,
    func: POINTER,
    grid_size: list[int],
    block_size: list[int],
    shared_mem_size: int,
    stream: POINTER,
    args: list[Any] = [],
) -> list[float]:
    event_start = hip.hipEventCreate()
    event_stop = hip.hipEventCreate()

    samples = []
    for i in range(count):
        hip.hipEventRecord(event_start)
        launch_kernel(
            func,
            grid_size,
            block_size,
            shared_mem_size,
            stream,
            args,
        )
        hip.hipEventRecord(event_stop)
        hip.hipDeviceSynchronize()
        show_progress(float(i + 1) / count)
        event_ms = hip.hipEventElapsedTime(event_start, event_stop)

        samples.append(float(work_per_kernel) / event_ms / 1e6)

    print()

    return samples


cache_bw_src = """
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

hbm_bw_src = """
template<typename T>
__global__ void HBM_bw(T *dst, const T *src)
{
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;

    dst[gid] = src[gid];
}
"""


def hbm_bw_benchmark(device: int) -> PerfMetrics:
    num_experiments = DEFAULT_NUM_EXPERIMENTS
    hip.hipSetDevice(device)

    cus = hip.hipGetDeviceProperties(device).multiProcessorCount

    prog = Program(hbm_bw_src, ["HBM_bw<double>"])
    func = prog.get_kernel("HBM_bw<double>")

    workgroup_size = DEFAULT_WORKGROUP_SIZE
    workgroups_per_cu = 20 * 1024
    workgroups = cus * workgroups_per_cu
    dataset_entries = workgroups * workgroup_size

    d_src = hip.hipMalloc(dataset_entries * sizeof(c_double))
    d_dst = hip.hipMalloc(dataset_entries * sizeof(c_double))

    total_bytes = dataset_entries * sizeof(c_double) * 2

    launch_kernel(
        func, [workgroups, 1, 1], [workgroup_size, 1, 1], 0, None, [d_dst, d_src]
    )
    hip.hipDeviceSynchronize()

    samples = run_get_samples(
        num_experiments,
        total_bytes,
        func,
        [workgroups, 1, 1],
        [workgroup_size, 1, 1],
        0,
        None,
        [d_dst, d_src],
    )

    stats = calc_stats(samples)

    mean = stats.mean
    stdev = stats.stdev

    perf_metrics = PerfMetrics(mean, mean - stats.confidence, mean + stats.confidence)

    event_ms = total_bytes / mean / 1e6

    print(
        f"HBM BW, GPU ID: {device}, workgroupSize:{workgroup_size}, "
        f"workgroups:{workgroups}, experiments:{num_experiments}, "
        f"traffic:{total_bytes} bytes, duration:{event_ms:.1f} ms, "
        f"mean:{mean:.1f} GB/sec, stdev:{stdev:.1f} GB/sec"
    )

    return perf_metrics


def cache_bw_bench(device: int, type: str, iters: int) -> PerfMetrics:
    hip.hipSetDevice(device)

    num_experiments = DEFAULT_NUM_EXPERIMENTS
    workgroup_size = DEFAULT_WORKGROUP_SIZE

    cus = hip.hipGetDeviceProperties(device).multiProcessorCount

    arch = get_gfx_arch(device)
    cache_size = cache_sizes[type][arch]

    mem_block = hip.hipMalloc(cache_size)
    dummy = hip.hipMalloc(workgroup_size * sizeof(c_float))

    kernel_name = cache_kernel_selector[type][arch]
    prog = Program(cache_bw_src, [kernel_name])
    func = prog.get_kernel(kernel_name)

    workgroups = 128 * cus
    total_bytes = workgroups * iters * cache_size

    launch_kernel(
        func,
        [workgroups, 1, 1],
        [workgroup_size, 1, 1],
        0,
        None,
        [mem_block, dummy, iters],
    )
    hip.hipDeviceSynchronize()

    samples = run_get_samples(
        num_experiments,
        total_bytes,
        func,
        [workgroups, 1, 1],
        [workgroup_size, 1, 1],
        0,
        None,
        [mem_block, dummy, iters],
    )

    stats = calc_stats(samples)
    mean = stats.mean
    stdev = stats.stdev

    perf_metrics = PerfMetrics(mean, mean - stats.confidence, mean + stats.confidence)

    event_ms = total_bytes / mean / 1e6

    print(
        f"{type} BW, GPU ID: {device}, workgroupSize:{workgroup_size}, "
        f"workgroups:{workgroups}, experiments:{num_experiments}, "
        f"traffic:{total_bytes} bytes, duration:{event_ms:.1f} ms, "
        f"mean:{mean:.1f} GB/sec, stdev:{stdev:1f} GB/sec"
    )

    return perf_metrics


def mall_bw_bench(device: int) -> PerfMetrics:
    return cache_bw_bench(device, "MALL", 1)


def l1_bw_bench(device: int) -> PerfMetrics:
    return cache_bw_bench(device, "L1", 100)


def l2_bw_bench(device: int) -> PerfMetrics:
    return cache_bw_bench(device, "L2", 10)


lds_benchmark_src = """
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


def lds_bw_benchmark(device: int) -> PerfMetrics:
    num_experiments = DEFAULT_NUM_EXPERIMENTS
    workgroup_size = DEFAULT_WORKGROUP_SIZE

    cus = hip.hipGetDeviceProperties(device).multiProcessorCount

    iters = 2000

    workgroups = 128 * cus
    total_bytes = workgroups * workgroup_size * iters * sizeof(c_float)

    dummy = hip.hipMalloc(workgroup_size * sizeof(c_float))

    prog = Program(lds_benchmark_src)
    func = prog.get_kernel("LDS_bw")

    # Warmup
    launch_kernel(
        func, [workgroups, 1, 1], [workgroup_size, 1, 1], 0, None, [iters, dummy]
    )
    hip.hipDeviceSynchronize()

    samples = run_get_samples(
        num_experiments,
        total_bytes,
        func,
        [workgroups, 1, 1],
        [workgroup_size, 1, 1],
        0,
        None,
        [iters, dummy],
    )

    stats = calc_stats(samples)
    mean = stats.mean
    stdev = stats.stdev

    perf_metrics = PerfMetrics(mean, mean - stats.confidence, mean + stats.confidence)

    event_ms = total_bytes / mean / 1e6

    print(
        f"LDS BW, GPU ID: {device}, workgroupSize:{workgroup_size}, "
        f"workgroups:{workgroups}, experiments:{num_experiments}, "
        f"traffic:{total_bytes} bytes, duration:{event_ms:.1f} ms, "
        f"mean:{mean:.1f} GB/sec, stdev:{stdev:1f} GB/sec"
    )

    return perf_metrics


flops_benchmark_src = """
template<typename T, int Rank>
using vecT = T __attribute__((ext_vector_type(Rank)));

template<typename T> using vec4 = vecT<T, 4>;

template<typename T, int nFMA>
__global__ void flops_benchmark(T *buf, int count)
{
    static_assert(nFMA % 4 == 0, "nFMA must be divisible by 4 for vec4 operations");

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


def flops_bench(device: int, type: str, unit: str, rate: int) -> PerfMetrics:
    num_experiments = DEFAULT_NUM_EXPERIMENTS
    workgroup_size = DEFAULT_WORKGROUP_SIZE
    cus = hip.hipGetDeviceProperties(device).multiProcessorCount

    workgroups = 128 * cus
    threads = workgroups * workgroup_size

    kernel_name = flops_kernel_selector[type][0]
    type_size = flops_kernel_selector[type][1]

    # Each thread reads a vec4
    dataset_size = 4 * type_size * threads
    memblock = hip.hipMalloc(dataset_size)

    iterations = flops_kernel_iterations[type]
    total_flops = threads * iterations * VALU_NFMA * 2

    prog = Program(flops_benchmark_src, [kernel_name])

    func = prog.get_kernel(kernel_name)

    # Warmup
    launch_kernel(
        func,
        [workgroups, 1, 1],
        [workgroup_size, 1, 1],
        0,
        None,
        [memblock, iterations],
    )
    hip.hipDeviceSynchronize()

    samples = run_get_samples(
        num_experiments,
        total_flops,
        func,
        [workgroups, 1, 1],
        [workgroup_size, 1, 1],
        0,
        None,
        [memblock, iterations],
    )

    stats = calc_stats(samples)
    mean = stats.mean
    stdev = stats.stdev

    perf_metrics = PerfMetrics(mean, mean - stats.confidence, mean + stats.confidence)

    event_ms = total_flops / mean / 1e6

    print(
        f"Peak VALU {unit}s ({type}), GPU ID: {device}, "
        f"workgroupSize:{workgroup_size}, "
        f"workgroups:{workgroups}, experiments:{num_experiments}, "
        f"{unit}:{total_flops}, duration:{event_ms:.1f} ms, "
        f"mean:{mean:.1f} {rate}, stdev={stdev:.1f} GFLOPS"
    )

    return perf_metrics


mfma_f32_src = """
using f32_16vec = __attribute__((__vector_size__(16 * sizeof(float)))) float;

extern "C" __global__ void mfma_f32(int iter, float *dummy)
{
    // Input: 1 F32 register
    float a =  threadIdx.x;

    // Output: 16 F32 registers
    f32_16vec result = {0};

    // CDNA2: v_mfma_f32_32x32x2f32 ops: 32x32x2x2 = 4096
    // CDNA3: v_mfma_f32_32x32x2_f32
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x2f32(a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}
"""

mfma_f16_src = """

using f32_16vec = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using f16_2vec = __attribute__((__vector_size__(2 * sizeof(__2f16))))  float;

extern "C" __global__ void mfma_f16(int iter, float *dummy)
{
    // Input: 2 F32 registers
    f16_2vec a;
    a[1] = a[0] = threadIdx.x;

    //Output: 16 F32 registers
    f32_16vec result = {0};

    // CDNA2: v_mfma_f32_32x32x8f16 ops: 32x32x8x2 = 16384
    // CDNA3: v_mfma_f32_32x32x8_f16
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

mfma_bf16_src = """

using f32_16vec = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using bf16_4vec = __attribute__((__vector_size__(2 * sizeof(__2i16))))  short;
using bf16_2vec = __attribute__((__vector_size__(1 * sizeof(__2i16))))  short;

extern "C" __global__ void mfma_bf16(int iter, float *dummy)
{
    // Output: 16 F32 registers
    f32_16vec result = {0};

// MI100/MI200
#if defined(__gfx908__) or defined(__gfx90a__)
    // Input: 1 F32 register
    // builtin mfma expects 2 short registers
    bf16_2vec a;
    a[1] = a[0]= threadIdx.x;

    // CDNA1/2: v_mfma_f32_32x32x4bf16 ops: 32x32x4x2 = 8192
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x4bf16(a, a, result, 0, 0, 0);
    }
//MI300 series
#else
    // Input: 2 F32 registers
    // builting mfma expects 4 short registers
    bf16_4vec a;
    a[3] = a[2] = a[1] = a[0]= threadIdx.x;

    // CDNA3: v_mfma_f32_32x32x8_bf16 ops: 32x32x8x2 = 16384
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, a, result, 0, 0, 0);
    }
#endif

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}
"""

mfma_f64_src = """

using f64_4vec = __attribute__((__vector_size__(4 * sizeof(double)))) double;

extern "C" __global__ void mfma_f64(int iter, float *dummy)
{
    // MI200 and above
    // Input: 1 F64 register
    double a =  threadIdx.x;

    // Output: 4 F64 registers
    f64_4vec result = {0};

    // CDNA2: v_mfma_f64_16x16x4f64 ops: 16x16x4x2 = 2048
    // CDNA3: v_mfma_f64_16x16x4_f64
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

mfma_i8_src = """
using int32_8vec = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using int32_16vec = __attribute__((__vector_size__(16 * sizeof(int)))) int;

extern "C" __global__ void mfma_i8(int iter, float *dummy)
{
    // Output: 16 I32 registers
    int32_16vec result = {0};

// MI100/MI200
#if defined(__gfx908__) or defined(__gfx90a__)
    // Input: 1 I32 register
    int a = threadIdx.x;

    // CDNA1/2: v_mfma_i32_32x32x8i8 ops: 32x32x8x2 = 16384
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_i32_32x32x8i8(a, a, result, 0, 0, 0);
    }
// MI300 series
#else
    // Input: 2 I32 registers
    // builting mfma expects I64 input
    long a =  threadIdx.x;

    // CDNA3: v_mfma_i32_32x32x16_i8 ops: 32x32x16x2 = 32768
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_i32_32x32x16_i8(a, a, result, 0, 0, 0);
    }
#endif

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}
"""

mfma_f8_src = """

using f32_16vec = __attribute__((__vector_size__(16 * sizeof(float)))) float;

extern "C" __global__ void mfma_f8(int iter, float *dummy)
{
    // MI300 series only - note gfx940/gfx941/gfx942 only uses fnuz f8
    // Input: 2 F32 registers
    // builtin mfma expects double input
    double a =  threadIdx.x;

    // Output: 16 F32 registers
    f32_16vec result = {0};

    // CDNA3: v_mfma_f32_32x32x16_fp8_fp8 ops: 32x32x16x2 = 32768
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}
"""

mfma_f8f6f4_src = """

using int32_16vec = __attribute__((__vector_size__(16 * sizeof(int)))) int;
using int32_8vec = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using bf16_2vec = __attribute__((__vector_size__(1 * sizeof(__2i16))))  short;
using bf16_4vec = __attribute__((__vector_size__(2 * sizeof(__2i16))))  short;
using f32_16vec = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using f16_2vec = __attribute__((__vector_size__(2 * sizeof(__2f16))))  float;

#define FP8_E4M3 0
#define BF8_E5M2 1
#define FP6_E2M3 2
#define BF6_E3M2 3
#define FP4_E2M1 4
#define FP6_FP4_MIXED 5

template<int datatype> __global__ void mfma_f8f6f4(int iter, float *dummy)
{
    // MI350 series only
    // Input: 8 i32 registers
    int32_8vec a;
    a[0] = a[1] = a[2] = a[3] = a[4] = a[5] = a[6] = a[7] = threadIdx.x;

    // Output: 16 F32 registers
    f32_16vec result = {0};

    // CDNA4: v_mfma_f32_32x32x64_f8f6f4    ops: 32x32x64x2 = 131072
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


def mfma_bench(device: int, type: str, unit: str, rate: int) -> PerfMetrics:
    SIMDS_PER_CU = 4
    experiments = DEFAULT_NUM_EXPERIMENTS
    iters = 2000

    cus = hip.hipGetDeviceProperties(device).multiProcessorCount

    workgroups = 128 * cus
    workgroup_size = DEFAULT_WORKGROUP_SIZE

    arch = get_gfx_arch(device)
    total_flops = workgroups * SIMDS_PER_CU * iters * mfma_ops[type][arch]

    dummy = hip.hipMalloc(64 * sizeof(c_float))

    kernel_name = mfma_kernel_selector[type]

    if type == "F32":
        src = mfma_f32_src
    elif type == "F8":
        src = mfma_f8_src
    elif type == "F16":
        src = mfma_f16_src
    elif type == "BF16":
        src = mfma_bf16_src
    elif type == "F64":
        src = mfma_f64_src
    elif type == "I8":
        src = mfma_i8_src
    else:
        src = mfma_f8f6f4_src

    prog = Program(src, [kernel_name])
    func = prog.get_kernel(kernel_name)

    samples = run_get_samples(
        experiments,
        total_flops,
        func,
        [workgroups, 1, 1],
        [workgroup_size, 1, 1],
        0,
        None,
        [iters, dummy],
    )

    stats = calc_stats(samples)
    mean = stats.mean
    stdev = stats.stdev

    perf_metrics = PerfMetrics(mean, mean - stats.confidence, mean + stats.confidence)

    event_ms = total_flops / mean / 1e6

    print(
        f"Peak MFMA {unit}s ({type}), GPU ID: {device}, "
        f"workgroupSize:{workgroup_size}, workgroups:{workgroups}, "
        f"experiments:{experiments}, {unit}:{total_flops}, "
        f"duration:{event_ms:.2f} ms, mean:{mean:.1f} {rate}, "
        f"stdev:{stdev:.1f} GFLOPS"
    )

    return perf_metrics


def mfma_f32_bench(device: int) -> PerfMetrics:
    return mfma_bench(device, "F32", "FLOP", "GFLOPS")


def mfma_f16_bench(device: int) -> PerfMetrics:
    return mfma_bench(device, "F16", "FLOP", "GFLOPS")


def mfma_bf16_bench(device: int) -> PerfMetrics:
    return mfma_bench(device, "BF16", "FLOP", "GFLOPS")


def mfma_f64_bench(device: int) -> PerfMetrics:
    return mfma_bench(device, "F64", "FLOP", "GFLOPS")


def mfma_f8_bench(device: int) -> PerfMetrics:
    return mfma_bench(device, "F8", "FLOP", "GFLOPS")


def mfma_i8_bench(device: int) -> PerfMetrics:
    return mfma_bench(device, "I8", "IOP", "GOPS")


def mfma_f4_bench(device: int) -> PerfMetrics:
    return mfma_bench(device, "F4", "FLOP", "GFLOPS")


def mfma_f6_bench(device: int) -> PerfMetrics:
    return mfma_bench(device, "F6", "FLOP", "GFLOPS")


def mfma_f6f4_bench(device: int) -> PerfMetrics:
    return mfma_bench(device, "F6F4", "FLOP", "GFLOPS")


def fp16_benchmark(device: int) -> PerfMetrics:
    return flops_bench(device, "FP16", "FLOP", "GFLOPS")


def fp32_benchmark(device: int) -> PerfMetrics:
    return flops_bench(device, "FP32", "FLOP", "GFLOPS")


def fp64_benchmark(device: int) -> PerfMetrics:
    return flops_bench(device, "FP64", "FLOP", "GFLOPS")


def int8_benchmark(device: int) -> PerfMetrics:
    return flops_bench(device, "INT8", "IOP", "GOPS")


def int32_benchmark(device: int) -> PerfMetrics:
    return flops_bench(device, "INT32", "IOP", "GOPS")


def int64_benchmark(device: int) -> PerfMetrics:
    return flops_bench(device, "INT64", "IOP", "GOPS")


tests = {
    "HBM": hbm_bw_benchmark,
    "MALL": mall_bw_bench,
    "L2": l2_bw_bench,
    "L1": l1_bw_bench,
    "LDS": lds_bw_benchmark,
    "F16": fp16_benchmark,
    "F32": fp32_benchmark,
    "F64": fp64_benchmark,
    "I8": int8_benchmark,
    "I32": int32_benchmark,
    "I64": int64_benchmark,
    "MFMA-F4": mfma_f4_bench,
    "MFMA-F6": mfma_f6_bench,
    "MFMA-F6F4": mfma_f6f4_bench,
    "MFMA-F8": mfma_f8_bench,
    "MFMA-F16": mfma_f16_bench,
    "MFMA-BF16": mfma_bf16_bench,
    "MFMA-F32": mfma_f32_bench,
    "MFMA-F64": mfma_f64_bench,
    "MFMA-I8": mfma_i8_bench,
}


# Run the roofline tests on the specified device
def run_benchmark(device: int) -> dict[PerfMetrics]:
    with gpu_benchmark_lock(device):
        metrics_dict = {}

        arch = get_gfx_arch(device)
        cus = hip.hipGetDeviceProperties(device).multiProcessorCount

        print(f"GPU Device {device} ({arch}) with {cus} CUs: Profiling...")

        for name, func in tests.items():
            if arch in unsupported_data_types and name in unsupported_data_types[arch]:
                print(f"Skipping {name}")
                metrics = PerfMetrics(0, 0, 0)
            else:
                metrics = func(device)

            metrics_dict[name] = metrics

        return metrics_dict


# Run the benchmark test on the specified devices
# Returns a dictionary mapping device ID to dictionary of
# metrics
def run_on_devices(devices: list[int]) -> dict[dict[PerfMetrics]]:
    metrics = {}
    for d in devices:
        metrics[d] = run_benchmark(d)

    return metrics


def dump_csv(metrics: dict[dict[PerfMetrics]], file_path: str) -> None:
    # TODO: Better way to map CSV column names?
    csv_cols_map = {
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

    with open(file_path, "w") as f:
        writer = csv.writer(f)

        types = csv_cols_map.keys()

        # Write the first row (col names)
        row = ["device"]
        for t in types:
            row.append(csv_cols_map[t])
            row.append(csv_cols_map[t] + "Low")
            row.append(csv_cols_map[t] + "High")

        writer.writerow(row)

        for d in metrics:
            row = [d]
            for t in types:
                row.append(metrics[d][t].mean)
                row.append(metrics[d][t].low)
                row.append(metrics[d][t].high)

            writer.writerow(row)


if __name__ == "__main__":
    import sys

    device_id = 0

    if len(sys.argv) >= 3:
        if sys.argv[1] == "-d":
            device_id = int(sys.argv[2])

    metrics = run_on_devices([device_id])
    dump_csv(metrics, "roofline.csv")

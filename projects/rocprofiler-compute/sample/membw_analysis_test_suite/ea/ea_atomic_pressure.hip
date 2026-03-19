// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// EA HBM Atomic Pressure Workload
//
// Target metrics: EA HBM Atomic Pressure (30.18.7),
//                 EA atomic fraction of writes (30.13.6),
//                 EA HBM write vs atomic split (30.13.7),
//                 EA atomic latency (30.14.12)
//
// Baseline: Histogram-like pattern with atomicAdd to a 64K-bin output array.
//           All threads issue atomics to global memory, creating high atomic
//           traffic at HBM.
//           Expected: HBM Atomic Pressure > 50%, atomic fraction > 50%.
//
// Optimized: LDS-local partial histogram per workgroup, with a single
//            atomicAdd per bin per workgroup to global memory.
//            Expected: HBM Atomic Pressure < 10%, atomic fraction < 10%.

#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            fflush(stderr); \
            exit(1); \
        } \
    } while(0)

constexpr int NUM_BINS = 65536; // 64K bins

// Baseline: Every thread atomicAdd directly to global histogram — heavy atomic traffic
__global__ void ea_atomic_pressure_baseline(const int* __restrict__ input,
                                             int* __restrict__ histogram,
                                             int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        int bin = input[i] & (NUM_BINS - 1); // Mask to bin range
        atomicAdd(&histogram[bin], 1);
    }
}

// Optimized: LDS-local histogram, then merge to global with far fewer atomics
__global__ void ea_atomic_pressure_optimized(const int* __restrict__ input,
                                              int* __restrict__ histogram,
                                              int N) {
    // Use a subset of bins in LDS to fit in shared memory
    // 256 bins * 4 bytes = 1KB per workgroup (fits easily)
    constexpr int LDS_BINS = 256;
    __shared__ int local_hist[LDS_BINS];

    // Initialize local histogram
    for (int i = threadIdx.x; i < LDS_BINS; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Accumulate into LDS-local histogram (atomics to LDS are fast)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        int bin = input[i] & (LDS_BINS - 1);
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    // Merge local histogram to global — only LDS_BINS atomics per workgroup
    for (int i = threadIdx.x; i < LDS_BINS; i += blockDim.x) {
        if (local_hist[i] > 0) {
            atomicAdd(&histogram[i], local_hist[i]);
        }
    }
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    // 1GB of ints = 256M elements
    int N = 256 * 1024 * 1024;
    size_t input_bytes = (size_t)N * sizeof(int);
    size_t hist_bytes = NUM_BINS * sizeof(int);

    int* d_input;
    CHECK_HIP(hipMalloc(&d_input, input_bytes));

    int* d_histogram;
    CHECK_HIP(hipMalloc(&d_histogram, hist_bytes));

    // Host init + copy (no hipMemset)
    int* h_input = new int[N];
    for (int i = 0; i < N; i++) {
        // Pseudo-random bin assignment to spread atomics
        h_input[i] = (i * 2654435761u) & (NUM_BINS - 1);
    }
    CHECK_HIP(hipMemcpy(d_input, h_input, input_bytes, hipMemcpyHostToDevice));
    delete[] h_input;

    // Zero-init histogram via host copy
    int* h_hist = new int[NUM_BINS]();
    CHECK_HIP(hipMemcpy(d_histogram, h_hist, hist_bytes, hipMemcpyHostToDevice));
    delete[] h_hist;

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    int num_blocks = prop.multiProcessorCount * 8;

    dim3 block(256);
    dim3 grid(num_blocks);

    // Single dispatch for clean profiling
    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        hipLaunchKernelGGL(ea_atomic_pressure_optimized, grid, block, 0, 0,
                           d_input, d_histogram, N);
    } else {
        hipLaunchKernelGGL(ea_atomic_pressure_baseline, grid, block, 0, 0,
                           d_input, d_histogram, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_histogram));
    return 0;
}

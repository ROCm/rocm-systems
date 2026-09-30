// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// EA HBM Write Credit Pressure Workload
//
// Target metrics: EA HBM BW Bound - Write Credit Pressure (30.18.1),
//                 EA to HBM write BW (30.14.1), EA Write Backpressure (30.18.5)
//
// Baseline: Streaming writes to a 4GB buffer using grid-stride loop.
//           Each thread writes sequentially, maximizing write request rate.
//           Expected: Write Credit Pressure > 10%, high EA to HBM write BW.
//
// Optimized: Accumulate partial results in registers across multiple iterations.
//            Write only the final accumulated value, reducing total write
//            requests by a large factor.
//            Expected: Write Credit Pressure < 5%, lower write BW.

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

// Baseline: Streaming writes across entire buffer — saturates HBM write BW
__global__ void ea_hbm_write_bw_baseline(double* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = tid; i < N; i += stride) {
        data[i] = (double)i * 0.001;
    }
}

// Optimized: Accumulate in registers, write only final result
__global__ void ea_hbm_write_bw_optimized(double* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    // Accumulate across all iterations in registers
    double acc = 0.0;
    for (long long i = tid; i < N; i += stride) {
        acc += (double)i * 0.001;
    }

    // Single write per thread
    if (tid < N) {
        data[tid] = acc;
    }
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    // 4GB of doubles = 512M elements
    long long N = 512LL * 1024 * 1024;
    size_t bytes = (size_t)N * sizeof(double);

    double* d_data;
    hipError_t alloc_err = hipMalloc(&d_data, bytes);
    if (alloc_err != hipSuccess) {
        // Fallback to 2GB
        N = 256LL * 1024 * 1024;
        bytes = (size_t)N * sizeof(double);
        CHECK_HIP(hipMalloc(&d_data, bytes));
    }

    // Host init + copy (no hipMemset)
    double* h_data = new double[N];
    for (long long i = 0; i < N; i++) h_data[i] = 0.0;
    CHECK_HIP(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    int num_blocks = prop.multiProcessorCount * 8;

    dim3 block(256);
    dim3 grid(num_blocks);

    // Single dispatch for clean profiling
    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        ea_hbm_write_bw_optimized<<<grid, block, 0, 0>>>(d_data, N);
    } else {
        ea_hbm_write_bw_baseline<<<grid, block, 0, 0>>>(d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

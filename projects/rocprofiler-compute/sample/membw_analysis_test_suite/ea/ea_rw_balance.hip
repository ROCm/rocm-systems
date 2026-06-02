// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// EA Read/Write Balance Workload
//
// Target metrics: EA Read/Write Balance (30.18.6),
//                 EA read request fraction - HBM (30.13.1),
//                 EA write request fraction - HBM (30.13.5)
//
// Baseline: Read-only streaming — sum a large array, write a single scalar.
//           Expected: R/W Balance > 95% (nearly all reads).
//
// Optimized: Balanced read-modify-write — vector copy from source to dest (1:1 R/W).
//            Expected: R/W Balance ~ 33-50%.

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

// Baseline: Read-only streaming — extreme read imbalance
__global__ void ea_rw_balance_baseline(const double* __restrict__ src,
                                        double* __restrict__ result,
                                        long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    double acc = 0.0;
    for (long long i = tid; i < N; i += stride) {
        acc += src[i];
    }

    // Single scalar write per thread (negligible write traffic)
    if (tid == 0) {
        result[0] = acc;
    }
}

// Optimized: 1:1 vector copy — balanced read/write
__global__ void ea_rw_balance_optimized(const double* __restrict__ src,
                                         double* __restrict__ dst,
                                         long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = tid; i < N; i += stride) {
        dst[i] = src[i];
    }
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    // 2GB of doubles = 256M elements
    long long N = 256LL * 1024 * 1024;
    size_t bytes = (size_t)N * sizeof(double);

    double* d_src;
    hipError_t alloc_err = hipMalloc(&d_src, bytes);
    if (alloc_err != hipSuccess) {
        // Fallback to 1GB
        N = 128LL * 1024 * 1024;
        bytes = (size_t)N * sizeof(double);
        CHECK_HIP(hipMalloc(&d_src, bytes));
    }

    double* d_dst;
    CHECK_HIP(hipMalloc(&d_dst, bytes));

    // Host init + copy (no hipMemset)
    double* h_data = new double[N];
    for (long long i = 0; i < N; i++) h_data[i] = i * 0.001;
    CHECK_HIP(hipMemcpy(d_src, h_data, bytes, hipMemcpyHostToDevice));

    // Zero-init destination via host copy
    for (long long i = 0; i < N; i++) h_data[i] = 0.0;
    CHECK_HIP(hipMemcpy(d_dst, h_data, bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    int num_blocks = prop.multiProcessorCount * 8;

    dim3 block(256);
    dim3 grid(num_blocks);

    // Single dispatch for clean profiling
    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        ea_rw_balance_optimized<<<grid, block, 0, 0>>>(d_src, d_dst, N);
    } else {
        ea_rw_balance_baseline<<<grid, block, 0, 0>>>(d_src, d_dst, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_src));
    CHECK_HIP(hipFree(d_dst));
    return 0;
}

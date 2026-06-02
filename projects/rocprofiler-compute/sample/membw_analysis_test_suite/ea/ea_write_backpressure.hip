// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// EA Write Path Backpressure Workload
//
// Target metrics: EA Write Backpressure (30.18.5),
//                 Normalized EA - Write stall (30.16.6),
//                 Normalized EA - Too many writes stall (30.16.7)
//
// Baseline: Coalesced streaming writes at maximum rate. Adjacent threads
//           write adjacent addresses across the full buffer in a tight loop
//           with no intervening compute. This floods TCC→EA with write
//           requests faster than EA can drain credits, triggering
//           TCC_EA0_WRREQ_STALL and TCC_EA0_WRREQ_DRAM_CREDIT_STALL.
//           Expected: Write Backpressure > 15%.
//
// Optimized: Same total writes but interleaved with heavy ALU work.
//            Compute between writes gives EA time to retire credits,
//            preventing write credit exhaustion.
//            Expected: Write Backpressure < 5%.

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

// Baseline: Coalesced streaming writes at maximum rate — floods EA with write requests
__global__ void ea_write_backpressure_baseline(float* __restrict__ data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Tight write loop with no intervening compute.
    // Adjacent threads write adjacent addresses (fully coalesced),
    // maximizing write request rate through TCC to EA.
    for (int iter = 0; iter < 64; iter++) {
        for (int i = tid; i < N; i += stride) {
            data[i] = (float)(i + iter);
        }
    }
}

// Optimized: Same writes but interleaved with heavy ALU work.
// Compute between writes gives EA time to drain write credits.
__global__ void ea_write_backpressure_optimized(float* __restrict__ data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        // Heavy ALU work before each write — throttles write request rate
        float val = (float)i;
        for (int j = 0; j < 256; j++) {
            val = val * 0.9999f + 0.0001f;
        }
        data[i] = val;
    }
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    // 2GB of floats = 512M elements
    int N = 512 * 1024 * 1024;
    size_t bytes = (size_t)N * sizeof(float);

    float* d_data;
    hipError_t alloc_err = hipMalloc(&d_data, bytes);
    if (alloc_err != hipSuccess) {
        // Fallback to 1GB
        N = 256 * 1024 * 1024;
        bytes = (size_t)N * sizeof(float);
        CHECK_HIP(hipMalloc(&d_data, bytes));
    }

    // Host init + copy (no hipMemset)
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 0.0f;
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
        ea_write_backpressure_optimized<<<grid, block, 0, 0>>>(d_data, N);
    } else {
        ea_write_backpressure_baseline<<<grid, block, 0, 0>>>(d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

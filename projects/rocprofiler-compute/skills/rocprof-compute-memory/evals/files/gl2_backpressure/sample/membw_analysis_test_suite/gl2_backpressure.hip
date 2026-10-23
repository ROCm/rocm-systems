// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

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

constexpr long long L2_STRIDE = 67 * 1024 * 1024LL;  // 67MB - doesn't divide N

// Baseline: Uncoalesced + large stride -> L2 misses + high request rate -> fill LFIFO -> TCR stall
__global__ void gl2_backpressure_baseline(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int lane = threadIdx.x & 63;  // Wave64

    // UNCOALESCED
    // LARGE STRIDE
    float acc = 0.0f;
    for (int i = 0; i < 128; i++) {
        long long offset = (long long)idx * 64 + (long long)lane * 32 + (long long)i * L2_STRIDE;
        int access_idx = (int)(offset % (long long)N);
        acc += data[access_idx];
    }
    data[idx] = acc;
}

// Optimized: Shared memory, minimal L2 traffic, no write pressure
__global__ void gl2_backpressure_optimized(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    __shared__ float sdata[256];
    sdata[threadIdx.x] = data[idx];
    __syncthreads();

    float sum = sdata[threadIdx.x];
    for (int i = 0; i < 64; i++) {
        sum = sum * 0.99f + 1.0f;
    }

    data[idx] = sum;  // Single coalesced write
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    // 2GB
    int N = 512 * 1024 * 1024;
    size_t bytes = (size_t)N * sizeof(float);

    float* d_data;
    hipError_t alloc_err = hipMalloc(&d_data, bytes);
    if (alloc_err != hipSuccess) {
        // Fallback to 1GB if 2GB fails (still exceeds 64MB L2)
        N = 256 * 1024 * 1024;
        bytes = (size_t)N * sizeof(float);
        CHECK_HIP(hipMalloc(&d_data, bytes));
    }

    // Host init + copy (no hipMemset - avoids __amd_rocclr_fillBufferAligned)
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = i * 0.001f;
    CHECK_HIP(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    // Block 64 -> more blocks -> more concurrent waves
    dim3 block(64);
    dim3 grid((N + block.x - 1) / block.x);

    // Single dispatch for clean profiling (no warmup to avoid double-dispatch in results)
    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        hipLaunchKernelGGL(gl2_backpressure_optimized, grid, block, 0, 0, d_data, N);
    } else {
        hipLaunchKernelGGL(gl2_backpressure_baseline, grid, block, 0, 0, d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

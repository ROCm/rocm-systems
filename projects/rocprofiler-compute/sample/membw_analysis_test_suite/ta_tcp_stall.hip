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

// Baseline: Random access, TA waits for TCP
__global__ void ta_tcp_stall_baseline(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float sum = 0;
    for (int i = 0; i < 32; i++) {
        long long rand_idx = ((long long)idx * 7919LL + (long long)i * 6547LL) % N;
        sum += data[rand_idx];
    }
    data[idx] = sum;
}

// Optimized: Sequential access, TCP serves TA quickly
__global__ void ta_tcp_stall_optimized(float* data, int N) {
    // Each thread handles 32 elements
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
    if (base + 31 >= N) return;

    float sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += data[base + i];
    }
    data[base] = sum;
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    const int N = 128 * 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    float* d_data;
    CHECK_HIP(hipMalloc(&d_data, bytes));

    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = i * 0.001f;
    CHECK_HIP(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    dim3 block(256);

    if (use_optimized) {
        int num_blocks = (N / 32 + block.x - 1) / block.x;
        dim3 grid(num_blocks);
        hipLaunchKernelGGL(ta_tcp_stall_optimized, grid, block, 0, 0, d_data, N);
    } else {
        dim3 grid((N + block.x - 1) / block.x);
        hipLaunchKernelGGL(ta_tcp_stall_baseline, grid, block, 0, 0, d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        int num_blocks = (N / 32 + block.x - 1) / block.x;
        dim3 grid(num_blocks);
        hipLaunchKernelGGL(ta_tcp_stall_optimized, grid, block, 0, 0, d_data, N);
    } else {
        dim3 grid((N + block.x - 1) / block.x);
        hipLaunchKernelGGL(ta_tcp_stall_baseline, grid, block, 0, 0, d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

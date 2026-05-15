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

/*
 * UTCL1 Stall Workload - Targets TCP_UTCL1_STALL_INFLIGHT_MAX / tcp_busy >= 10%
 *
 */

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

constexpr int PAGE_FLOATS = 4096 / sizeof(float);  // 1024 floats per 4KB page

// Baseline: Rapid page hopping - exceed UTCL1 entries, trigger INFLIGHT stall
__global__ void utcl1_stall_baseline(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int lane = threadIdx.x & 63;  // Wave64

    float acc = 0.0f;
    for (int i = 0; i < 512; i++) {
        unsigned long long page_offset = (unsigned long long)lane * PAGE_FLOATS
                                       + (unsigned long long)i * 64 * PAGE_FLOATS;
        int access_idx = (int)(((unsigned long long)idx + page_offset) % (unsigned long long)N);
        acc += data[access_idx];
    }
    data[idx] = acc;
}

// Optimized: All accesses within same page
__global__ void utcl1_stall_optimized(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int page_base = (idx / PAGE_FLOATS) * PAGE_FLOATS;

    float sum = 0.0f;
    for (int i = 0; i < 512; i++) {
        sum += data[page_base + (i % PAGE_FLOATS)];
    }
    data[idx] = sum;
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    const int N = 256 * 1024 * 1024;  // 1GB - 256K pages
    const size_t bytes = N * sizeof(float);

    float* d_data;
    CHECK_HIP(hipMalloc(&d_data, bytes));

    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = i * 0.001f;
    CHECK_HIP(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    dim3 block(64);
    dim3 grid((N + block.x - 1) / block.x);

    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        hipLaunchKernelGGL(utcl1_stall_optimized, grid, block, 0, 0, d_data, N);
    } else {
        hipLaunchKernelGGL(utcl1_stall_baseline, grid, block, 0, 0, d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

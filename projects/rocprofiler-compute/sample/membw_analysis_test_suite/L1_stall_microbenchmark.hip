// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// L1 stall micro-benchmarks
// Usage:  ./a.out       = baseline (uncoalesced) -> TA/UTCL1/L2 stall
//         ./a.out opt  = optimized (shared mem) -> no stall
//         ./a.out sq1  = SQ-L1 back-pressure     -> try to get "VMEM stalled by L1" > 0

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
    } while (0)

// -----------------------------------------------------------------------------
// SQ-L1 back-pressure kernel: maximize VMEM issue rate so SQ->TA FIFOs fill.
// - Many independent loads in a row (no data dependency) so SQ can issue as fast
//   as the pipeline allows; then stores. Goal: fill VMEM command/addr/data FIFO.
// - Uncoalesced so TA->TCP is slow and FIFOs back up to SQ.
// - Actual behaviour: uncoalesced heavy L1 -> L2 traffic, but not filling up VMEM FIFO
// Run with: ./a sq1
// -----------------------------------------------------------------------------
__global__ void vmem_sq_l1_backpressure(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N / 64) return;
    int base = tid * 64;

    // Independent loads
    // Uncoalesced
    float v[32];
#pragma unroll
    for (int i = 0; i < 32; i++) {
        int load_idx = (base + threadIdx.x * 32 + i * 4096) % N;
        v[i] = data[load_idx];
    }

    // 32 stores. Uncoalesced, different stride.
#pragma unroll
    for (int i = 0; i < 32; i++) {
        int store_idx = (base + threadIdx.x * 32 + i * 8192) % N;
        data[store_idx] = v[i] * 2.0f;
    }
}

// Fixed baseline: UNCOALESCED accesses
__global__ void vmem_stall_baseline(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float vals[16];
    // Uncoalesced loads: threadIdx.x * 32 puts each of 32 threads in a warp on a
    // different cache line (32 floats = 128B). i * 1024 spreads across pages.
    for (int i = 0; i < 16; i++) {
        int load_idx = (idx + threadIdx.x * 32 + i * 1024) % N;
        vals[i] = data[load_idx];
    }

    // Uncoalesced stores: same idea, different stride to avoid overlap with loads.
    for (int i = 0; i < 16; i++) {
        int store_idx = (idx + threadIdx.x * 32 + i * 2048) % N;
        data[store_idx] = vals[i] * 2.0f;
    }
}

// Optimized: Shared memory reduces VMEM FIFO pressure
__global__ void vmem_stall_optimized(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    __shared__ float sdata[256];

    sdata[threadIdx.x] = data[idx];
    __syncthreads();

    float result = sdata[threadIdx.x];
    for (int i = 0; i < 16; i++) {
        result = result * 1.01f + 0.1f;
    }

    data[idx] = result;
}

int main(int argc, char** argv) {
    // opt = optimized (shared mem); sq1 = SQ-L1 back-pressure (try to get "VMEM stalled by L1" > 0)
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");
    bool use_sq1 = (argc > 1 && std::string(argv[1]) == "sq1");

    const int N = 256 * 1024 * 1024;  // 1GB
    const size_t bytes = N * sizeof(float);

    float* d_data;
    CHECK_HIP(hipMalloc(&d_data, bytes));

    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    CHECK_HIP(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    dim3 block(256);
    dim3 grid_baseline((N + block.x - 1) / block.x);
    dim3 grid_sq1((N / 64 + block.x - 1) / block.x);  // tid = N/64 threads total

    auto run = [&]() {
        if (use_optimized)
            hipLaunchKernelGGL(vmem_stall_optimized, grid_baseline, block, 0, 0, d_data, N);
        else if (use_sq1)
            hipLaunchKernelGGL(vmem_sq_l1_backpressure, grid_sq1, block, 0, 0, d_data, N);
        else
            hipLaunchKernelGGL(vmem_stall_baseline, grid_baseline, block, 0, 0, d_data, N);
    };

    run();
    CHECK_HIP(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    run();
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    const char* name = use_optimized ? "Optimized" : (use_sq1 ? "SQ-L1 backpressure" : "Baseline");
    std::cout << name << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

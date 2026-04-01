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

// L2 Atomic Stress
//
// Target metrics (tables 3008, 3009):
//   - L2-EA atomic latency   [TCC_EA0_ATOMIC_LEVEL / TCC_EA0_ATOMIC]
//   - L2 total request rate   [TCC_REQ / TCC_BUSY]
//   - TCC_EA0_ATOMIC (raw)
//   - TCC_ATOMIC (raw)
//   - L2-EA read/write latency (atomics contend with both paths)
//
// Mechanism:
//   Baseline: Many threads perform atomicAdd on a small set of addresses
//   scattered across a buffer larger than L2. With thousands of threads
//   targeting few cachelines, each atomic must serialize, and those that
//   miss L2 go to EA as atomic requests.
//   The TCC_EA0_ATOMIC counter tracks atomics sent to the EA interface.
//   High contention -> high per-atomic latency (TCC_EA0_ATOMIC_LEVEL).
//
//   Optimized: Uses regular read-modify-write (no atomics) on a small
//   buffer that fits in L2. This eliminates EA atomic traffic entirely,
//   showing the contrast between atomic-heavy and atomic-free paths.
//
// Build:  hipcc -g l2_atomic_stress.hip -o l2_atomic_stress
// Run:    ./l2_atomic_stress        (baseline)
//         ./l2_atomic_stress opt    (optimized)

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

size_t getL2CacheSize() {
    int val = 0;
    CHECK_HIP(hipDeviceGetAttribute(&val, hipDeviceAttributeL2CacheSize, 0));
    if (val > 0) return (size_t)val;
    return 256ULL * 1024 * 1024;
}

// Baseline: High-contention atomics on scattered addresses > L2 size.
// num_targets is small, so many threads compete
// on the same addresses. Buffer > L2 means atomics often go to EA.
__global__ void l2_stress_with_atomic(float* __restrict__ data,
                                        long long N, int num_targets) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Map thread to one of num_targets cachelines, spread across the buffer.
    // This creates high contention (many threads -> same cacheline) AND
    // L2 misses (targets spread over buffer > L2).
    long long target_spacing = N / (long long)num_targets;
    // Ensure spacing is cacheline-aligned
    target_spacing = (target_spacing / 16) * 16;
    if (target_spacing < 16) target_spacing = 16;

    int target_idx = (int)(tid % (long long)num_targets);
    long long addr = (long long)target_idx * target_spacing;
    if (addr >= N) addr = addr % N;

    // Repeated atomics on the same target -> high contention + EA atomics
    for (int iter = 0; iter < 512; iter++) {
        atomicAdd(&data[addr], 1.0f);
    }
}

// Optimized: Regular read-modify-write (no atomics) on a buffer that
// fits in L2. Each thread works on its own cacheline. No EA atomic
// traffic is generated. This contrasts with the baseline which sends
// all operations through the EA atomic path.
__global__ void l2_stress_without_atomic(float* __restrict__ data,
                                         long long N, long long tile_elems) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread gets a unique cacheline within the L2-fitting tile
    long long addr = (tid * 16) % tile_elems;

    // Regular RMW instead of atomics: hits in L2, no EA atomic traffic
    for (int iter = 0; iter < 512; iter++) {
        data[addr] = data[addr] + 1.0f;
    }
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    size_t l2_size = getL2CacheSize();
    std::cout << "L2 cache size: " << (l2_size / (1024 * 1024)) << " MB\n";

    // Buffer 2x L2 for baseline (atomics scattered beyond L2)
    size_t buffer_bytes = l2_size * 2;
    long long N = (long long)(buffer_bytes / sizeof(float));
    long long tile_elems = (long long)(l2_size / (4 * sizeof(float)));

    // Number of distinct target cachelines for contention.
    int num_targets = 1024;

    float* d_data;
    CHECK_HIP(hipMalloc(&d_data, buffer_bytes));

    float* h_data = new float[N];
    for (long long i = 0; i < N; i++) h_data[i] = 0.0f;
    CHECK_HIP(hipMemcpy(d_data, h_data, buffer_bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    dim3 block(64);
    dim3 grid(4096);

    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        l2_stress_without_atomic<<<grid, block>>>(d_data, N, tile_elems);
    } else {
        l2_stress_with_atomic<<<grid, block>>>(d_data, N, num_targets);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

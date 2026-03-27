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

// L2 Cache Thrash — Internal Stall Stress
//
// Target metrics (tables 3007, 3008, 3009, 3010):
//   - L2 tag stall rate              [TCC_TAG_STALL]
//   - L2 input buffer stall rate     [TCC_IB_STALL]  (back pressure indicator)
//   - L2 latency FIFO full rate      [TCC_LATENCY_FIFO_FULL]
//   - L2 utilization                 [TCC_BUSY / total_l2_chan * GRBM]
//   - L2 eviction rate               [TCC_NORMAL_EVICT]
//   - L2 Back Pressure Indicator     (table 3009)
//   - L2 Internal Resource Pressure  (table 3009)
//   - Normalized L2 stalls           (table 3010)
//
// Mechanism:
//   Baseline: Read-modify-write on scattered cachelines across a buffer
//   1.5x L2 size. The scatter pattern (prime-stride) ensures all TCC channels
//   are hit simultaneously. Each thread does repeated RMW iterations, creating:
//     1. High miss rate -> outstanding EA reads fill the latency FIFO
//     2. Dirty cachelines -> writeback on eviction -> writeback FIFO pressure
//     3. Continuous tag lookups with misses -> TAG_STALL
//     4. L2 can't accept new L1 requests -> IB_STALL -> backpressure
//     5. TAG_MISS_NOTHING_REPLACEABLE_STALL when all lines have pending fills
//
//   Optimized: Coalesced RMW on a small working set that fits in L2.
//   High hit rate, no stalls.
//
// Build:  hipcc -g l2_cache_thrash.hip -o l2_cache_thrash
// Run:    ./l2_cache_thrash        (baseline)
//         ./l2_cache_thrash opt    (optimized)

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

// Baseline: Scattered read-modify-write over buffer > L2.
// Prime-number stride ensures accesses are spread across TCC channels
// and consecutive iterations hit different cachelines, preventing reuse.
// The RMW pattern makes cachelines dirty, so eviction triggers writebacks.
__global__ void cache_thrash_baseline(float* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Use a large prime stride to scatter across TCC channels.
    // 7919 * 16 floats = ~500KB stride per step -> wraps around buffer,
    // hitting different cachelines each iteration.
    const long long prime_stride = 7919LL * 16;  // in float elements

    long long idx = (tid * 16) % N;  // Start at unique cacheline

    float acc = 0.0f;
    for (int iter = 0; iter < 256; iter++) {
        // Read-modify-write: makes the cacheline dirty
        float val = data[idx];
        acc += val;
        data[idx] = val + 1.0f;

        // Jump to a different cacheline (scattered pattern)
        idx = (idx + prime_stride) % N;
    }

    // Prevent optimization
    if (acc == -999.0f) data[0] = acc;
}

// Optimized: Sequential coalesced RMW on a small working set.
// All accesses hit in L2 after first pass.
__global__ void cache_thrash_optimized(float* __restrict__ data, long long N,
                                       long long tile_elems) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (int rep = 0; rep < 64; rep++) {
        for (long long i = tid; i < tile_elems; i += stride) {
            data[i] = data[i] + 1.0f;
        }
    }
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    size_t l2_size = getL2CacheSize();
    std::cout << "L2 cache size: " << (l2_size / (1024 * 1024)) << " MB\n";

    // 1.5x L2 for baseline: large enough to thrash but not so large
    // that we spend all time on memory transfers
    size_t buffer_bytes = (l2_size * 3) / 2;
    long long N = (long long)(buffer_bytes / sizeof(float));
    long long tile_elems = (long long)(l2_size / (4 * sizeof(float)));

    float* d_data;
    CHECK_HIP(hipMalloc(&d_data, buffer_bytes));

    float* h_data = new float[N];
    for (long long i = 0; i < N; i++) h_data[i] = (float)(i % 1000) * 0.001f;
    CHECK_HIP(hipMemcpy(d_data, h_data, buffer_bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    // More blocks = more concurrent waves = more L2 pressure
    dim3 block(64);
    dim3 grid(8192);

    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        cache_thrash_optimized<<<grid, block>>>(d_data, N, tile_elems);
    } else {
        cache_thrash_baseline<<<grid, block>>>(d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

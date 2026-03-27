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

// L2 HBM Read Bandwidth Stress
//
// Target metrics (tables 3007, 3008, 3009):
//   - L2-EA read credit stall (HBM)  [TCC_EA0_RDREQ_DRAM_CREDIT_STALL]
//   - L2-EA read BW
//   - HBM read fraction
//   - L2 Memory BW Bound - Read Credit Pressure
//   - L2 hit rate (expect low on baseline)
//   - L2 to Memory Traffic Ratio (expect high on baseline)
//   - L2 Cache Efficiency (expect low on baseline)
//
// Mechanism:
//   Baseline: Streaming reads over a buffer 4x L2 size. Each thread reads
//   from sequential cachelines with no reuse. Working set far exceeds L2,
//   so nearly every access is an L2 miss -> EA read -> HBM. This saturates
//   the L2-to-HBM read path and exhausts DRAM read credits.
//
//   Optimized: Reads only a small tile that fits in L2. High hit rate,
//   zero credit stalls.
//
// Build:  hipcc -g l2_hbm_read_bw_stress.hip -o l2_hbm_read_bw_stress
// Run:    ./l2_hbm_read_bw_stress        (baseline)
//         ./l2_hbm_read_bw_stress opt    (optimized)

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
    // Fallback: 256MB (MI350 default per XCD)
    return 256ULL * 1024 * 1024;
}

// Baseline: Streaming read over buffer >> L2 size.
// Each thread reads multiple cachelines spread across the full buffer,
// ensuring no temporal reuse in L2. The read-only pattern maximizes
// EA read requests to HBM and starves DRAM read credits.
__global__ void hbm_read_stress_baseline(const float* __restrict__ data,
                                          float* __restrict__ out,
                                          long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float acc = 0.0f;
    // Stride through entire buffer. Each iteration touches a new cacheline.
    // 16 floats = 64 bytes = 1 cacheline
    for (long long i = tid * 16; i < N; i += stride * 16) {
        acc += data[i];
    }

    // Prevent compiler from optimizing away reads
    if (tid == 0) out[0] = acc;
}

// Optimized: Read only a small region that fits comfortably in L2.
// Repeatedly access the same data -> high L2 hit rate, no EA traffic.
__global__ void hbm_read_stress_optimized(const float* __restrict__ data,
                                           float* __restrict__ out,
                                           long long N, long long tile_elems) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float acc = 0.0f;
    // Read within the tile repeatedly to get L2 hits
    for (int rep = 0; rep < 64; rep++) {
        for (long long i = tid * 16; i < tile_elems; i += stride * 16) {
            acc += data[i];
        }
    }

    if (tid == 0) out[0] = acc;
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    size_t l2_size = getL2CacheSize();
    std::cout << "L2 cache size: " << (l2_size / (1024 * 1024)) << " MB\n";

    // Buffer = 4x L2 size to guarantee streaming misses
    size_t buffer_bytes = l2_size * 4;
    long long N = (long long)(buffer_bytes / sizeof(float));

    // Tile for optimized = 1/4 of L2 (fits comfortably)
    long long tile_elems = (long long)(l2_size / (4 * sizeof(float)));

    float* d_data;
    float* d_out;
    hipError_t alloc_err = hipMalloc(&d_data, buffer_bytes);
    if (alloc_err != hipSuccess) {
        // Fallback to 2x L2
        buffer_bytes = l2_size * 2;
        N = (long long)(buffer_bytes / sizeof(float));
        CHECK_HIP(hipMalloc(&d_data, buffer_bytes));
    }
    CHECK_HIP(hipMalloc(&d_out, sizeof(float)));

    // Initialize with host data (avoid hipMemset which uses fill kernels)
    float* h_data = new float[N];
    for (long long i = 0; i < N; i++) h_data[i] = (float)(i % 1000) * 0.001f;
    CHECK_HIP(hipMemcpy(d_data, h_data, buffer_bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    // Launch config: block=64 (Wave64), many blocks for maximum L2 pressure
    dim3 block(64);
    dim3 grid(4096);

    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        hbm_read_stress_optimized<<<grid, block>>>(d_data, d_out, N, tile_elems);
    } else {
        hbm_read_stress_baseline<<<grid, block>>>(d_data, d_out, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_data));
    CHECK_HIP(hipFree(d_out));
    return 0;
}

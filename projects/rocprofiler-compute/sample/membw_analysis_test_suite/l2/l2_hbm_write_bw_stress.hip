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

// L2 HBM Write Bandwidth Stress
//
// Target metrics (tables 3007, 3008, 3009):
//   - L2-EA write credit stall (HBM)  [TCC_EA0_WRREQ_DRAM_CREDIT_STALL]
//   - L2-EA write stall rate          [TCC_EA0_WRREQ_STALL]
//   - L2-EA too many writes stall     [TCC_TOO_MANY_EA_WRREQS_STALL]
//   - L2 write data FIFO full rate    [TCC_SRC_FIFO_FULL]
//   - L2-EA write and atomic BW
//   - HBM write fraction
//   - L2 Memory BW Bound - Write Credit Pressure
//   - L2 writeback rate (expect high on baseline)
//
// Mechanism:
//   Baseline: Streaming writes over a buffer 4x L2 size. Every thread writes
//   to a unique cacheline (no reuse). This forces:
//     1. L2 miss on write -> allocate cacheline
//     2. Dirty cacheline evicted when replaced -> writeback to HBM
//     3. HBM write request rate at maximum -> write credits exhausted
//     4. Source data FIFO fills up (write data waiting to be sent to EA)
//     5. TOO_MANY_EA_WRREQS_STALL when max pending EA writes reached
//
//   Optimized: Writes to a small buffer that fits in L2. Overwrites same
//   cachelines repeatedly -> minimal EA write traffic.
//
// Build:  hipcc -g l2_hbm_write_bw_stress.hip -o l2_hbm_write_bw_stress
// Run:    ./l2_hbm_write_bw_stress        (baseline)
//         ./l2_hbm_write_bw_stress opt    (optimized)

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

// Baseline: Streaming write over buffer >> L2 size.
// Each thread writes to unique cachelines across the full buffer.
// Write-only pattern: every write allocates a new cacheline, and
// evicted dirty lines are written back to HBM, maximizing EA write traffic.
__global__ void hbm_write_stress_baseline(float* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    // Write 1 float per cacheline (16 floats = 64B cacheline).
    // This ensures each write touches a distinct cacheline.
    for (long long i = tid * 16; i < N; i += stride * 16) {
        data[i] = (float)i;
    }
}

// Optimized: Write to a small region that fits in L2.
// Same cachelines are overwritten repeatedly -> dirty lines stay in L2,
// minimal writeback to HBM.
__global__ void hbm_write_stress_optimized(float* __restrict__ data, long long N,
                                            long long tile_elems) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (int rep = 0; rep < 64; rep++) {
        for (long long i = tid * 16; i < tile_elems; i += stride * 16) {
            data[i] = (float)(i + rep);
        }
    }
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    size_t l2_size = getL2CacheSize();
    std::cout << "L2 cache size: " << (l2_size / (1024 * 1024)) << " MB\n";

    size_t buffer_bytes = l2_size * 4;
    long long N = (long long)(buffer_bytes / sizeof(float));
    long long tile_elems = (long long)(l2_size / (4 * sizeof(float)));

    float* d_data;
    hipError_t alloc_err = hipMalloc(&d_data, buffer_bytes);
    if (alloc_err != hipSuccess) {
        buffer_bytes = l2_size * 2;
        N = (long long)(buffer_bytes / sizeof(float));
        CHECK_HIP(hipMalloc(&d_data, buffer_bytes));
    }

    // Initialize (avoid hipMemset)
    float* h_data = new float[N];
    for (long long i = 0; i < N; i++) h_data[i] = 0.0f;
    CHECK_HIP(hipMemcpy(d_data, h_data, buffer_bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    dim3 block(64);
    dim3 grid(4096);

    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        hbm_write_stress_optimized<<<grid, block>>>(d_data, N, tile_elems);
    } else {
        hbm_write_stress_baseline<<<grid, block>>>(d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

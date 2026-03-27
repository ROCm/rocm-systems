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

// L2 Normalized Throughput & Combined Stall Stress
//
// Target metrics (tables 3008, 3009, 3010):
//   - All Normalized L2 Stall Metrics (% of GRBM_GUI_ACTIVE)
//   - All Normalized L2 Throughput Metrics (requests/s)
//   - L2 Memory BW Bound - Combined Credit Pressure
//   - L2-EA read BW, L2-EA write and atomic BW
//
// Mechanism:
//   Baseline: Mixed read+write streaming kernel over buffer 4x L2.
//   Purely memory-bound (minimal compute), so GRBM_GUI_ACTIVE closely
//   tracks memory stall time. Alternating reads and writes at cacheline
//   granularity maximizes both EA read and write traffic simultaneously,
//   which is the worst case for combined DRAM credit pressure.
//
//   Optimized: Compute-heavy kernel with minimal memory traffic.
//   Lots of FMA operations on register values -> high GRBM_GUI_ACTIVE
//   but near-zero L2 traffic. Normalized stalls will be very low
//   because the denominator (kernel time) is large while stall
//   numerators are near zero.
//
// Build:  hipcc -g l2_normalized_throughput.hip -o l2_normalized_throughput
// Run:    ./l2_normalized_throughput        (baseline: memory-bound)
//         ./l2_normalized_throughput opt    (optimized: compute-bound)

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

// Baseline: Mixed read+write streaming, purely memory-bound.
// Alternates reading one cacheline and writing the next, across the
// entire buffer. No compute to hide latency -> memory-bound.
// This maximizes both TCC_READ and TCC_WRITE simultaneously,
// creating combined DRAM credit pressure on both read and write paths.
__global__ void normalized_stress_baseline(float* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float acc = 0.0f;
    // Each iteration: read from cacheline i, write to cacheline i+1
    // Stride by 32 floats (2 cachelines) per step
    for (long long i = tid * 32; i < N - 16; i += stride * 32) {
        // Read cacheline
        acc += data[i];
        // Write next cacheline
        data[i + 16] = acc;
    }

    // Prevent optimization
    if (acc == -999.0f) data[0] = acc;
}

// Optimized: Compute-heavy, minimal memory.
// Reads a single float, does hundreds of FMAs in registers, writes back.
// GRBM_GUI_ACTIVE is high (kernel runs long), but L2 traffic is negligible.
// Result: normalized stall metrics are near zero.
__global__ void normalized_stress_optimized(float* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Single memory read
    float val = data[tid % N];

    // Pure compute: 1024 FMA iterations
    float a = val, b = 1.0001f, c = 0.9999f;
    for (int i = 0; i < 1024; i++) {
        a = a * b + c;
        b = b * c + a;
        c = c * a + b;
    }

    // Single memory write
    if (tid < N) data[tid] = a + b + c;
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    size_t l2_size = getL2CacheSize();
    std::cout << "L2 cache size: " << (l2_size / (1024 * 1024)) << " MB\n";

    size_t buffer_bytes = l2_size * 4;
    long long N = (long long)(buffer_bytes / sizeof(float));

    float* d_data;
    hipError_t alloc_err = hipMalloc(&d_data, buffer_bytes);
    if (alloc_err != hipSuccess) {
        buffer_bytes = l2_size * 2;
        N = (long long)(buffer_bytes / sizeof(float));
        CHECK_HIP(hipMalloc(&d_data, buffer_bytes));
    }

    // Initialize
    float* h_data = new float[N];
    for (long long i = 0; i < N; i++) h_data[i] = (float)(i % 1000) * 0.001f;
    CHECK_HIP(hipMemcpy(d_data, h_data, buffer_bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    dim3 block(64);
    dim3 grid(4096);

    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        normalized_stress_optimized<<<grid, block>>>(d_data, N);
    } else {
        normalized_stress_baseline<<<grid, block>>>(d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized (compute-bound)" : "Baseline (memory-bound)") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}

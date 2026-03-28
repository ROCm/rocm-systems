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

// EA IO/PCIe Bandwidth Workload
//
// Target metrics: EA IO BW Bound - Combined (30.18.4),
//                 EA read BW fraction - IO (30.13.4),
//                 EA Stall Destination Dominance - IO (30.18.10),
//                 EA to IO read BW (30.14.7)
//
// Baseline: Streaming reads from host-pinned (fine-grained coherent) memory.
//           All traffic traverses PCIe/IO path.
//           Expected: IO BW Bound > 5%, read BW fraction IO > 90%.
//
// Optimized: Copy host data to device memory first, then read from local HBM.
//            Expected: IO BW Bound ~ 0%, Stall Dominance HBM ~ 100%.
//
// Note: IO credit stalls may be lower in absolute terms than HBM stalls
// because PCIe BW is inherently lower. The key validation is that the
// Stall Destination Dominance metric correctly identifies IO as the
// dominant stall source in the baseline.

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

// Streaming read kernel — same kernel for both cases, just different pointers
__global__ void ea_io_bw_read(const double* __restrict__ data,
                               double* __restrict__ out,
                               long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    double acc = 0.0;
    for (long long i = tid; i < N; i += stride) {
        acc += data[i];
    }

    if (tid < N) {
        out[tid] = acc;
    }
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    // 1GB of doubles = 128M elements
    long long N = 128LL * 1024 * 1024;
    size_t bytes = (size_t)N * sizeof(double);

    // Allocate host-pinned coherent memory (fine-grained, visible via PCIe)
    double* h_pinned;
    CHECK_HIP(hipHostMalloc(&h_pinned, bytes, hipHostMallocCoherent));

    // Initialize host data
    for (long long i = 0; i < N; i++) h_pinned[i] = i * 0.001;

    // Allocate device output buffer
    double* d_out;
    CHECK_HIP(hipMalloc(&d_out, bytes));

    // For optimized: copy to device memory first
    double* d_local = nullptr;
    if (use_optimized) {
        CHECK_HIP(hipMalloc(&d_local, bytes));
        CHECK_HIP(hipMemcpy(d_local, h_pinned, bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipDeviceSynchronize());
    }

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    int num_blocks = prop.multiProcessorCount * 8;

    dim3 block(256);
    dim3 grid(num_blocks);

    // Pointer to read from: host-pinned (baseline) or device-local (optimized)
    const double* read_ptr = use_optimized ? d_local : h_pinned;

    // Single dispatch for clean profiling
    auto start = std::chrono::high_resolution_clock::now();
    ea_io_bw_read<<<grid, block, 0, 0>>>(read_ptr, d_out, N);
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_out));
    if (d_local) CHECK_HIP(hipFree(d_local));
    CHECK_HIP(hipHostFree(h_pinned));
    return 0;
}

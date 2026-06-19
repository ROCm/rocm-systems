// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// EA HBM Read Credit Pressure Workload
//
// Target metrics: EA HBM BW Bound - Read Credit Pressure (30.18.0),
//                 EA to HBM read BW (30.14.0), EA read latency (30.14.10)
//
// Baseline: Streaming reads from a 4GB buffer (>> 4MB L2) using grid-stride loop.
//           Coalesced sequential access maximizes read request rate, flooding EA
//           with outstanding read requests and exhausting HBM read credits.
//           Expected: Read Credit Pressure > 10%, high EA to HBM read BW.
//
// Optimized: Tiled access pattern where each workgroup reads a tile that fits
//            partially in L2, then reuses it across multiple passes.
//            L2 hits reduce EA read requests.
//            Expected: Read Credit Pressure < 5%, lower EA to HBM read BW.

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

// Baseline: Streaming reads across entire 4GB buffer — saturates HBM read BW
__global__ void ea_hbm_read_bw_baseline(const double* __restrict__ data,
                                         double* __restrict__ out,
                                         long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    double acc = 0.0;
    for (long long i = tid; i < N; i += stride) {
        acc += data[i];
    }
    // One write per workgroup to avoid polluting write metrics
    if (threadIdx.x == 0) {
        out[blockIdx.x] = acc;
    }
}

// Optimized: All workgroups re-read a small region that fits in L2 (~2MB)
// After the first pass cold-misses, subsequent passes hit in L2 cache.
// This dramatically reduces EA read requests compared to streaming the full buffer.
__global__ void ea_hbm_read_bw_optimized(const double* __restrict__ data,
                                          double* __restrict__ out,
                                          long long N) {
    // Region that fits in L2: 2MB = 256K doubles
    // MI350 has 4MB L2 per XCD; 2MB leaves room for other traffic
    constexpr long long REGION_ELEMS = 256LL * 1024; // 2MB
    constexpr int PASSES = 256; // Re-read the region many times

    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    double acc = 0.0;
    for (int pass = 0; pass < PASSES; pass++) {
        for (long long i = tid; i < REGION_ELEMS; i += stride) {
            acc += data[i];
        }
    }

    // One write per workgroup — negligible write traffic
    if (threadIdx.x == 0) {
        out[blockIdx.x] = acc;
    }
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    // 4GB of doubles = 512M elements
    long long N = 512LL * 1024 * 1024;
    size_t bytes = (size_t)N * sizeof(double);

    double* d_data;
    hipError_t alloc_err = hipMalloc(&d_data, bytes);
    if (alloc_err != hipSuccess) {
        // Fallback to 2GB if 4GB fails
        N = 256LL * 1024 * 1024;
        bytes = (size_t)N * sizeof(double);
        CHECK_HIP(hipMalloc(&d_data, bytes));
    }

    // Host init + copy (no hipMemset — avoids __amd_rocclr_fillBufferAligned)
    double* h_data = new double[N];
    for (long long i = 0; i < N; i++) h_data[i] = i * 0.001;
    CHECK_HIP(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    int num_blocks = prop.multiProcessorCount * 8;

    // Output: one element per workgroup (negligible write traffic)
    double* d_out;
    CHECK_HIP(hipMalloc(&d_out, (size_t)num_blocks * sizeof(double)));

    dim3 block(256);
    dim3 grid(num_blocks);

    // Single dispatch for clean profiling
    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        ea_hbm_read_bw_optimized<<<grid, block, 0, 0>>>(d_data, d_out, N);
    } else {
        ea_hbm_read_bw_baseline<<<grid, block, 0, 0>>>(d_data, d_out, N);
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

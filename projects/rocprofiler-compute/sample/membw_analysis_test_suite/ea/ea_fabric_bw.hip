// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// EA GMI Bandwidth Workload (multi-GPU)
//
// Target metrics: EA GMI BW Bound - Combined (30.18.3),
//                 EA read BW fraction - GMI (30.13.3),
//                 EA Stall Destination Dominance - GMI (30.18.9),
//                 EA to GMI read BW (30.14.4)
//
// Baseline: Streaming reads from remote GPU memory via peer access.
//           All traffic traverses GMI.
//           Expected: GMI BW Bound > 10%, read BW fraction GMI > 90%.
//
// Optimized: Copy remote data to local GPU memory first, then read locally.
//            All traffic via HBM.
//            Expected: GMI BW Bound ~ 0%, Stall Dominance HBM ~ 100%.
//
// Requires >= 2 GPUs. Skips gracefully on single-GPU systems.

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

// Streaming read kernel — same kernel used for both baseline (remote ptr) and optimized (local ptr)
__global__ void ea_fabric_bw_read(const double* __restrict__ data,
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

    int device_count;
    CHECK_HIP(hipGetDeviceCount(&device_count));

    if (device_count < 2) {
        std::cout << "Skipping: requires >= 2 GPUs (found " << device_count << ")\n";
        return 0;
    }

    // Verify peer access is possible
    int can_access;
    CHECK_HIP(hipDeviceCanAccessPeer(&can_access, 0, 1));
    if (!can_access) {
        std::cout << "Skipping: GPU 0 cannot access GPU 1 peer memory\n";
        return 0;
    }

    // 2GB of doubles = 256M elements
    long long N = 256LL * 1024 * 1024;
    size_t bytes = (size_t)N * sizeof(double);

    // Allocate source buffer on GPU 1
    CHECK_HIP(hipSetDevice(1));
    double* d_remote;
    hipError_t alloc_err = hipMalloc(&d_remote, bytes);
    if (alloc_err != hipSuccess) {
        // Fallback to 1GB
        N = 128LL * 1024 * 1024;
        bytes = (size_t)N * sizeof(double);
        CHECK_HIP(hipMalloc(&d_remote, bytes));
    }

    // Host init + copy to GPU 1
    double* h_data = new double[N];
    for (long long i = 0; i < N; i++) h_data[i] = i * 0.001;
    CHECK_HIP(hipMemcpy(d_remote, h_data, bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    // Switch to GPU 0 and enable peer access to GPU 1
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipDeviceEnablePeerAccess(1, 0));

    // Allocate output buffer on GPU 0
    double* d_out;
    CHECK_HIP(hipMalloc(&d_out, bytes));

    // For optimized: allocate local copy on GPU 0
    double* d_local = nullptr;
    if (use_optimized) {
        CHECK_HIP(hipMalloc(&d_local, bytes));
        CHECK_HIP(hipMemcpyPeer(d_local, 0, d_remote, 1, bytes));
        CHECK_HIP(hipDeviceSynchronize());
    }

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    int num_blocks = prop.multiProcessorCount * 8;

    dim3 block(256);
    dim3 grid(num_blocks);

    // Pointer to read from: remote (baseline) or local (optimized)
    const double* read_ptr = use_optimized ? d_local : d_remote;

    // Single dispatch for clean profiling
    auto start = std::chrono::high_resolution_clock::now();
    ea_fabric_bw_read<<<grid, block, 0, 0>>>(read_ptr, d_out, N);
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_out));
    if (d_local) CHECK_HIP(hipFree(d_local));
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipFree(d_remote));
    return 0;
}

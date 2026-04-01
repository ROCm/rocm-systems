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

// L2 Coherence Traffic
//
// Target metrics (tables 3008, 3009):
//   - Non-coherent request rate      [TCC_NC_REQ / TCC_REQ]
//   - Uncached request rate           [TCC_UC_REQ / TCC_REQ]
//   - Coherent cached request rate    [TCC_CC_REQ / TCC_REQ]
//   - Probe request rate              [TCC_PROBE / TCC_BUSY]
//   - Probe eviction rate             [TCC_PROBE_EVICT / TCC_PROBE]
//
// Mechanism:
//   Baseline (mode=fg): Uses hipHostMalloc with hipHostMallocCoherent flag
//   to allocate system memory visible to the GPU. GPU accesses to this
//   coherent host memory go through the IO/IF path and should generate
//   UC (uncached) or CC (coherent cached) requests, elevating TCC_UC_REQ
//   and/or TCC_CC_REQ. The kernel does read-modify-write on this memory.
//
//   Baseline (mode=nc): Attempts hipExtMallocWithFlags(hipDeviceMallocFinegrained),
//   falling back to hipMalloc if unavailable. The kernel unconditionally uses
//   __builtin_nontemporal_store to generate non-coherent streaming traffic (TCC_NC_REQ).
//
//   Optimized: Standard coarse-grained hipMalloc with normal cached accesses.
//   Most requests will be normal cached (not NC/UC/CC).
//
// Build:  hipcc -g l2_coherence_traffic.hip -o l2_coherence_traffic
// Run:    ./l2_coherence_traffic           (fine-grained CC traffic)
//         ./l2_coherence_traffic nc        (non-temporal/streaming traffic)
//         ./l2_coherence_traffic opt       (optimized, coarse-grained)

#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>

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

// Fine-grained RMW: CC memory type -> coherence protocol active.
// Each thread does read-modify-write which triggers internal probes
// (TCC self-probes for CC writes) and coherence tracking.
__global__ void coherence_fg_baseline(float* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = tid; i < N; i += stride) {
        // RMW on fine-grained memory: triggers CC protocol
        data[i] = data[i] + 1.0f;
    }
}

// Non-temporal streaming write: generates NC (non-coherent) traffic.
// Uses __builtin_nontemporal_store to bypass cache coherence.
__global__ void coherence_nc_baseline(float* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = tid; i < N; i += stride) {
        __builtin_nontemporal_store((float)i, &data[i]);
    }
}

// Optimized: Standard coarse-grained memory with normal cached accesses.
// No coherence protocol overhead, no non-temporal hints.
__global__ void coherence_optimized(float* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = tid; i < N; i += stride) {
        data[i] = data[i] + 1.0f;
    }
}

int main(int argc, char** argv) {
    enum Mode { FG, NC, OPT } mode = FG;
    if (argc > 1) {
        if (std::string(argv[1]) == "nc") mode = NC;
        else if (std::string(argv[1]) == "opt") mode = OPT;
    }

    size_t l2_size = getL2CacheSize();
    std::cout << "L2 cache size: " << (l2_size / (1024 * 1024)) << " MB\n";

    // Use 1/2 L2 size for fine-grained (smaller to keep coherence overhead dominant)
    // Use 2x L2 for NC (larger to generate streaming traffic)
    size_t buffer_bytes;
    if (mode == NC) {
        buffer_bytes = l2_size * 2;
    } else {
        buffer_bytes = l2_size / 2;
    }
    long long N = (long long)(buffer_bytes / sizeof(float));

    float* d_data = nullptr;

    if (mode == FG) {
        // Coherent host memory: GPU accesses go through IO/IF path
        // generating UC or CC requests in TCC
        CHECK_HIP(hipHostMalloc(&d_data, buffer_bytes, hipHostMallocCoherent));
    } else if (mode == NC) {
        // Try fine-grained device allocation first for NC traffic
        hipError_t err = hipExtMallocWithFlags((void**)&d_data, buffer_bytes,
                                                hipDeviceMallocFinegrained);
        if (err != hipSuccess) {
            std::cerr << "Warning: hipExtMallocWithFlags(Finegrained) failed: "
                      << hipGetErrorString(err) << "\n"
                      << "Falling back to hipMalloc with nontemporal stores\n";
            CHECK_HIP(hipMalloc(&d_data, buffer_bytes));
        }
    } else {
        // Coarse-grained for OPT mode
        CHECK_HIP(hipMalloc(&d_data, buffer_bytes));
    }

    // Initialize
    float* h_data = new float[N];
    for (long long i = 0; i < N; i++) h_data[i] = (float)(i % 1000) * 0.001f;
    CHECK_HIP(hipMemcpy(d_data, h_data, buffer_bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    dim3 block(64);
    dim3 grid(4096);

    const char* mode_str = (mode == FG) ? "Fine-grained (CC)" :
                           (mode == NC) ? "Non-temporal (NC)" : "Optimized (coarse)";
    std::cout << "Mode: " << mode_str << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    if (mode == FG) {
        coherence_fg_baseline<<<grid, block>>>(d_data, N);
    } else if (mode == NC) {
        coherence_nc_baseline<<<grid, block>>>(d_data, N);
    } else {
        coherence_optimized<<<grid, block>>>(d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    if (mode == FG) {
        CHECK_HIP(hipHostFree(d_data));
    } else {
        CHECK_HIP(hipFree(d_data));
    }
    return 0;
}

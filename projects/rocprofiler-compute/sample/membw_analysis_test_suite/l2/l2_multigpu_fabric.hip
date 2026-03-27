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

// L2 Multi-GPU Infinity Fabric Stress
//
// ** REQUIRES 2 GPUs ** — see instructions below.
//
// Target metrics (tables 3007, 3009, 3010):
//   - L2-EA read credit stall (IF)    [TCC_EA0_RDREQ_GMI_CREDIT_STALL]
//   - L2-EA write credit stall (IF)   [TCC_EA0_WRREQ_GMI_CREDIT_STALL]
//   - L2 Remote Access Pressure (IF)  (table 3009)
//   - Normalized L2-EA stalls (IF)    (table 3010)
//
// Mechanism:
//   GPU 0 accesses memory allocated on GPU 1 via P2P. The L2 on GPU 0
//   misses, and the EA request goes through the GMI (Infinity Fabric)
//   interface to reach GPU 1's HBM. The IF has limited credits, and
//   saturating this path exhausts GMI credits.
//
//   mode=read:  Streaming reads from remote GPU memory
//   mode=write: Streaming writes to remote GPU memory
//   mode=rw:    Interleaved read+write (default)
//
// Build:  hipcc -g l2_multigpu_fabric.hip -o l2_multigpu_fabric
//
// Run (profile GPU 0 while it accesses GPU 1's memory):
//   # Read from remote:
//   rocprof-compute profile -n fabric_read --membw-analysis --experimental --no-roof -- ./l2_multigpu_fabric read
//   # Write to remote:
//   rocprof-compute profile -n fabric_write --membw-analysis --experimental --no-roof -- ./l2_multigpu_fabric write
//   # Read+Write:
//   rocprof-compute profile -n fabric_rw --membw-analysis --experimental --no-roof -- ./l2_multigpu_fabric
//
// Analyze:
//   rocprof-compute analyze -p <profiled_result_path> --membw-analysis --experimental

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

// Streaming read from remote memory -> GMI read credit stalls
__global__ void fabric_read_kernel(const float* __restrict__ remote_data,
                                    float* __restrict__ local_out,
                                    long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (long long i = tid * 16; i < N; i += stride * 16) {
        acc += remote_data[i];
    }

    if (tid == 0) local_out[0] = acc;
}

// Streaming write to remote memory -> GMI write credit stalls
__global__ void fabric_write_kernel(float* __restrict__ remote_data,
                                     long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = tid * 16; i < N; i += stride * 16) {
        remote_data[i] = (float)i;
    }
}

// Interleaved read+write to remote -> both GMI credit types exhausted
__global__ void fabric_rw_kernel(float* __restrict__ remote_data,
                                  float* __restrict__ local_out,
                                  long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (long long i = tid * 16; i < N; i += stride * 16) {
        float val = remote_data[i];
        acc += val;
        remote_data[i] = val + 1.0f;
    }

    if (tid == 0) local_out[0] = acc;
}

int main(int argc, char** argv) {
    enum Mode { RW, READ, WRITE } mode = RW;
    if (argc > 1) {
        if (std::string(argv[1]) == "read") mode = READ;
        else if (std::string(argv[1]) == "write") mode = WRITE;
    }

    // Check GPU count
    int ndevices = 0;
    CHECK_HIP(hipGetDeviceCount(&ndevices));
    if (ndevices < 2) {
        std::cerr << "ERROR: This workload requires >= 2 GPUs. Found " << ndevices << ".\n";
        std::cerr << "Skipping. Run on a multi-GPU system.\n";
        return 0;
    }

    int local_gpu = 0;
    int remote_gpu = 1;

    // Enable P2P
    int can_access = 0;
    CHECK_HIP(hipDeviceCanAccessPeer(&can_access, local_gpu, remote_gpu));
    if (!can_access) {
        std::cerr << "ERROR: GPU " << local_gpu << " cannot access GPU " << remote_gpu << " via P2P.\n";
        return 1;
    }
    CHECK_HIP(hipSetDevice(local_gpu));
    CHECK_HIP(hipDeviceEnablePeerAccess(remote_gpu, 0));

    // Allocate on remote GPU
    size_t buffer_bytes = 2ULL * 1024 * 1024 * 1024;  // 2GB
    long long N = (long long)(buffer_bytes / sizeof(float));

    CHECK_HIP(hipSetDevice(remote_gpu));
    float* d_remote;
    hipError_t alloc_err = hipMalloc(&d_remote, buffer_bytes);
    if (alloc_err != hipSuccess) {
        buffer_bytes = 1ULL * 1024 * 1024 * 1024;  // Fallback 1GB
        N = (long long)(buffer_bytes / sizeof(float));
        CHECK_HIP(hipMalloc(&d_remote, buffer_bytes));
    }
    // Initialize remote data
    CHECK_HIP(hipMemset(d_remote, 0, buffer_bytes));

    // Local output buffer on GPU 0
    CHECK_HIP(hipSetDevice(local_gpu));
    float* d_local_out;
    CHECK_HIP(hipMalloc(&d_local_out, sizeof(float)));

    dim3 block(64);
    dim3 grid(4096);

    const char* mode_str = (mode == READ) ? "Read" :
                           (mode == WRITE) ? "Write" : "Read+Write";
    std::cout << "Mode: " << mode_str
              << " | Buffer: " << (buffer_bytes / (1024 * 1024)) << " MB"
              << " | GPU " << local_gpu << " -> GPU " << remote_gpu << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    if (mode == READ) {
        fabric_read_kernel<<<grid, block>>>(d_remote, d_local_out, N);
    } else if (mode == WRITE) {
        fabric_write_kernel<<<grid, block>>>(d_remote, N);
    } else {
        fabric_rw_kernel<<<grid, block>>>(d_remote, d_local_out, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_local_out));
    CHECK_HIP(hipSetDevice(remote_gpu));
    CHECK_HIP(hipFree(d_remote));
    return 0;
}

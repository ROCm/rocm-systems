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

// L2 IO (PCIe) Credit Stall Stress
//
// Target metrics (tables 3007, 3009, 3010):
//   - L2-EA read credit stall (IO)    [TCC_EA0_RDREQ_IO_CREDIT_STALL]
//   - L2-EA write credit stall (IO)   [TCC_EA0_WRREQ_IO_CREDIT_STALL]
//   - Normalized L2-EA stalls (IO)    (table 3010)
//
// Mechanism:
//   Baseline: GPU kernel accesses host-pinned memory (hipHostMalloc).
//   Traffic goes over PCIe (or xGMI to host) which is much slower than
//   local HBM. The IO interface has limited credits, and sustained
//   access from many CUs exhausts them quickly.
//
//   mode=read:  Streaming reads from host-pinned memory
//   mode=write: Streaming writes to host-pinned memory
//   (default):  Read+write (interleaved)
//
//   Optimized: Same kernel but on device-local memory (hipMalloc).
//   All traffic stays on HBM, IO credits are not consumed.
//
// Build:  hipcc -g l2_io_stress.hip -o l2_io_stress
// Run:    ./l2_io_stress              (baseline: host-pinned R+W)
//         ./l2_io_stress read         (baseline: host-pinned read)
//         ./l2_io_stress write        (baseline: host-pinned write)
//         ./l2_io_stress opt          (optimized: device-local)

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

// Streaming read from host memory -> IO read credit stalls
__global__ void io_read_kernel(const float* __restrict__ data,
                                float* __restrict__ out, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (long long i = tid * 16; i < N; i += stride * 16) {
        acc += data[i];
    }
    if (tid == 0) out[0] = acc;
}

// Streaming write to host memory -> IO write credit stalls
__global__ void io_write_kernel(float* __restrict__ data, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = tid * 16; i < N; i += stride * 16) {
        data[i] = (float)i;
    }
}

// Interleaved R+W on host memory -> both IO credit types
__global__ void io_rw_kernel(float* __restrict__ data,
                              float* __restrict__ out, long long N) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (long long i = tid * 16; i < N; i += stride * 16) {
        float val = data[i];
        acc += val;
        data[i] = val + 1.0f;
    }
    if (tid == 0) out[0] = acc;
}

int main(int argc, char** argv) {
    enum Mode { RW, READ, WRITE, OPT } mode = RW;
    if (argc > 1) {
        if (std::string(argv[1]) == "read") mode = READ;
        else if (std::string(argv[1]) == "write") mode = WRITE;
        else if (std::string(argv[1]) == "opt") mode = OPT;
    }

    // 1GB host-pinned buffer (large enough to saturate IO path)
    size_t buffer_bytes = 1ULL * 1024 * 1024 * 1024;
    long long N = (long long)(buffer_bytes / sizeof(float));

    float* data = nullptr;
    float* d_out = nullptr;

    if (mode == OPT) {
        // Device-local memory: no IO traffic
        hipError_t err = hipMalloc(&data, buffer_bytes);
        if (err != hipSuccess) {
            buffer_bytes = 512ULL * 1024 * 1024;
            N = (long long)(buffer_bytes / sizeof(float));
            CHECK_HIP(hipMalloc(&data, buffer_bytes));
        }
        CHECK_HIP(hipMemset(data, 0, buffer_bytes));
    } else {
        // Host-pinned memory: traffic goes over PCIe/IO
        hipError_t err = hipHostMalloc(&data, buffer_bytes, hipHostMallocNonCoherent);
        if (err != hipSuccess) {
            buffer_bytes = 512ULL * 1024 * 1024;
            N = (long long)(buffer_bytes / sizeof(float));
            CHECK_HIP(hipHostMalloc(&data, buffer_bytes, hipHostMallocNonCoherent));
        }
        memset(data, 0, buffer_bytes);
    }

    CHECK_HIP(hipMalloc(&d_out, sizeof(float)));

    dim3 block(64);
    dim3 grid(4096);

    const char* mode_str = (mode == READ) ? "Host-pinned Read" :
                           (mode == WRITE) ? "Host-pinned Write" :
                           (mode == OPT) ? "Device-local (optimized)" :
                           "Host-pinned R+W";
    std::cout << "Mode: " << mode_str
              << " | Buffer: " << (buffer_bytes / (1024 * 1024)) << " MB\n";

    auto start = std::chrono::high_resolution_clock::now();
    if (mode == READ) {
        io_read_kernel<<<grid, block>>>(data, d_out, N);
    } else if (mode == WRITE) {
        io_write_kernel<<<grid, block>>>(data, N);
    } else {
        // RW and OPT use same kernel (different memory backing)
        io_rw_kernel<<<grid, block>>>(data, d_out, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    CHECK_HIP(hipFree(d_out));
    if (mode == OPT) {
        CHECK_HIP(hipFree(data));
    } else {
        CHECK_HIP(hipHostFree(data));
    }
    return 0;
}

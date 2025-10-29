// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <assert.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <csignal>
#include <iostream>
#include <mutex>
#include <vector>

#include "common/defines.hpp"

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            auto _hip_api_print_lk = auto_lock_t{print_lock};                                      \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

namespace
{
using auto_lock_t = std::unique_lock<std::mutex>;
auto print_lock   = std::mutex{};
}  // namespace

#define WIDTH  (1024)
#define HEIGHT (1024)

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 64
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_Z 1

// Computes vectorAdd with matrix-multiply
template <typename T>
__global__ void
addition_kernel(T* __restrict__ a,
                const float* __restrict__ b,
                const float* __restrict__ c,
                int                  width,
                [[maybe_unused]] int height)
{
    // printf("addition kernel\n");
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = b[index] + c[index];
}

__global__ void
subtract_kernel(float* __restrict__ a,
                const float* __restrict__ b,
                const float* __restrict__ c,
                int                  width,
                [[maybe_unused]] int height)
{
    // printf("subtract kernel\n");
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = abs(b[index] - c[index]);
}

__global__ void
multiply_kernel(float* __restrict__ a,
                const float* __restrict__ b,
                const float* __restrict__ c,
                int                  width,
                [[maybe_unused]] int height)
{
    // printf("multiply kernel\n");
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = (b[index] - 1) * (c[index] - 1) + 1;
}

__global__ void
divide_kernel(float* __restrict__ a,
              const float* __restrict__ b,
              const float* __restrict__ c,
              int                  width,
              [[maybe_unused]] int height)
{
    // printf("divide kernel\n");
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = (b[index] - c[index]) / abs(c[index] + b[index]) + 1;
}

using namespace std;

void
run(int NUM_QUEUE, int DEVICE_ID)
{
    HIP_API_CALL(hipSetDevice(DEVICE_ID));

    HIP_API_CALL(hipDeviceSynchronize());

    std::vector<float*> hostA(NUM_QUEUE);
    std::vector<float*> hostB(NUM_QUEUE);
    std::vector<float*> hostC(NUM_QUEUE);

    std::vector<float*> deviceA(NUM_QUEUE);
    std::vector<float*> deviceB(NUM_QUEUE);
    std::vector<float*> deviceC(NUM_QUEUE);

    std::vector<hipStream_t> streams(NUM_QUEUE);

    auto sync_stream = [NUM_QUEUE, &streams](int q) {
        if(q < 0 || q >= NUM_QUEUE)
            throw std::runtime_error{std::string{"invalid stream id: "} + std::to_string(q)};

        HIP_API_CALL(hipStreamSynchronize(streams.at(q)));
    };

    auto sync_streams = [NUM_QUEUE, sync_stream]() {
        for(int i = 0; i < NUM_QUEUE; ++i)
            sync_stream(i);
        HIP_API_CALL(hipDeviceSynchronize());
    };

    for(int q = 0; q < NUM_QUEUE; q++)
    {
        HIP_API_CALL(hipStreamCreateWithFlags(&streams[q], hipStreamNonBlocking));

        HIP_API_CALL(HIP_HOST_ALLOC_FUNC(&hostA[q], NUM * sizeof(float), 0));
        HIP_API_CALL(HIP_HOST_ALLOC_FUNC(&hostB[q], NUM * sizeof(float), 0));
        HIP_API_CALL(HIP_HOST_ALLOC_FUNC(&hostC[q], NUM * sizeof(float), 0));

        // initialize the input data
        for(int i = 0; i < NUM; i++)
        {
            hostB[q][i] = static_cast<float>(i);
            hostC[q][i] = static_cast<float>(i * 100.0f);
        }

        HIP_API_CALL(hipMallocAsync(&deviceA[q], NUM * sizeof(float), streams[q]));
        HIP_API_CALL(hipMallocAsync(&deviceB[q], NUM * sizeof(float), streams[q]));
        HIP_API_CALL(hipMallocAsync(&deviceC[q], NUM * sizeof(float), streams[q]));

        HIP_API_CALL(hipMemcpyAsync(
            deviceB[q], hostB[q], NUM * sizeof(float), hipMemcpyHostToDevice, streams[q]));
        HIP_API_CALL(hipMemcpyAsync(
            deviceC[q], hostC[q], NUM * sizeof(float), hipMemcpyHostToDevice, streams[q]));
    }

    sync_streams();

    for(int q = 0; q < NUM_QUEUE; q++)
    {
        hipLaunchKernelGGL(addition_kernel,
                           dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                           dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                           0,
                           streams[q],
                           deviceA[q],
                           deviceB[q],
                           deviceC[q],
                           WIDTH,
                           HEIGHT);

        HIP_API_CALL(hipGetLastError());

        hipLaunchKernelGGL(subtract_kernel,
                           dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                           dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                           0,
                           streams[q],
                           deviceA[q],
                           deviceB[q],
                           deviceC[q],
                           WIDTH,
                           HEIGHT);

        HIP_API_CALL(hipGetLastError());

        if(getenv("ROCPROF_TESTING_RAISE_SIGNAL") != nullptr &&
           std::stoi(getenv("ROCPROF_TESTING_RAISE_SIGNAL")) > 0)
        {
            ::raise(SIGINT);
        }

        hipLaunchKernelGGL(multiply_kernel,
                           dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                           dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                           0,
                           streams[q],
                           deviceA[q],
                           deviceB[q],
                           deviceC[q],
                           WIDTH,
                           HEIGHT);

        HIP_API_CALL(hipGetLastError());

        hipLaunchKernelGGL(divide_kernel,
                           dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                           dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                           0,
                           streams[q],
                           deviceB[q],
                           deviceA[q],
                           deviceC[q],
                           WIDTH,
                           HEIGHT);

        HIP_API_CALL(hipGetLastError());
    }

    sync_streams();

    for(int q = 0; q < NUM_QUEUE; q++)
    {
        HIP_API_CALL(hipMemcpyAsync(
            hostA[q], deviceA[q], NUM * sizeof(float), hipMemcpyDeviceToHost, streams[q]));

        sync_stream(q);

        HIP_API_CALL(hipFree(deviceA[q]));
        HIP_API_CALL(hipFree(deviceB[q]));
        HIP_API_CALL(hipFree(deviceC[q]));

        HIP_API_CALL(HIP_HOST_FREE_FUNC(hostA[q]));
        HIP_API_CALL(HIP_HOST_FREE_FUNC(hostB[q]));
        HIP_API_CALL(HIP_HOST_FREE_FUNC(hostC[q]));

        HIP_API_CALL(hipStreamDestroy(streams[q]));
    }

    HIP_API_CALL(hipDeviceSynchronize());
}

int
main(int argc, char** argv)
{
    int stream_count = 8;
    int device_count = 0;
    HIP_API_CALL(hipGetDeviceCount(&device_count));

    if(argc > 1) stream_count = std::stoi(argv[1]);
    if(argc > 2) device_count = std::stoi(argv[2]);

    for(int i = 0; i < device_count; ++i)
        run(stream_count, i);

    return 0;
}

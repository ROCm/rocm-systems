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
#include <iostream>
#include <mutex>
#include <thread>
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

constexpr auto WIDTH  = (1 << 12);  // 4096
constexpr auto HEIGHT = (1 << 11);  // 2048
constexpr auto DEPTH  = (1 << 0);   // 1
constexpr auto NUM    = (WIDTH * HEIGHT * DEPTH);

struct dimensions
{
    int x = 1;
    int y = 1;
    int z = 1;
};

constexpr auto threads_per_block = dimensions{64, 1, 1};

// Computes vectorAdd with matrix-multiply
template <typename Tp>
__global__ void
addition_kernel(Tp* __restrict__ a,
                const Tp* __restrict__ b,
                const Tp* __restrict__ c,
                int width,
                int /*height*/)
{
    // printf("addition kernel\n");
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = b[index] + c[index];
}

template <typename Tp>
__global__ void
subtract_kernel(Tp* __restrict__ a,
                const Tp* __restrict__ b,
                const Tp* __restrict__ c,
                int width,
                int /*height*/)
{
    // printf("subtract kernel\n");
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = abs(b[index] - c[index]);
}

template <typename Tp>
__global__ void
multiply_kernel(Tp* __restrict__ a,
                const Tp* __restrict__ b,
                const Tp* __restrict__ c,
                int width,
                int /*height*/)
{
    // printf("multiply kernel\n");
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = (b[index] - 1) * (c[index] - 1) + 1;
}

template <typename Tp>
__global__ void
divide_kernel(Tp* __restrict__ a,
              const Tp* __restrict__ b,
              const Tp* __restrict__ c,
              int width,
              int /*height*/)
{
    // printf("divide kernel\n");
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = (b[index] - c[index]) / abs(c[index] + b[index]) + 1;
}

void
run_vector_ops_impl(int num_queue, int device_id)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    HIP_API_CALL(hipSetDevice(device_id));

    std::vector<float*> hostA(num_queue);
    std::vector<float*> hostB(num_queue);
    std::vector<float*> hostC(num_queue);

    std::vector<float*> deviceA(num_queue);
    std::vector<float*> deviceB(num_queue);
    std::vector<float*> deviceC(num_queue);

    std::vector<hipStream_t> streams(num_queue);

    auto sync_stream = [num_queue, &streams](int q) {
        if(q < 0 || q >= num_queue)
            throw std::runtime_error{std::string{"invalid stream id: "} + std::to_string(q)};

        HIP_API_CALL(hipStreamSynchronize(streams.at(q)));
    };

    auto sync_streams = [num_queue, sync_stream]() {
        for(int i = 0; i < num_queue; ++i)
            sync_stream(i);
    };

    for(int q = 0; q < num_queue; q++)
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

    for(int q = 0; q < num_queue; q++)
    {
        hipLaunchKernelGGL(addition_kernel,
                           dim3(WIDTH / threads_per_block.x, HEIGHT / threads_per_block.y),
                           dim3(threads_per_block.x, threads_per_block.y),
                           0,
                           streams[q],
                           deviceA[q],
                           deviceB[q],
                           deviceC[q],
                           WIDTH,
                           HEIGHT);

        hipLaunchKernelGGL(subtract_kernel,
                           dim3(WIDTH / threads_per_block.x, HEIGHT / threads_per_block.y),
                           dim3(threads_per_block.x, threads_per_block.y),
                           0,
                           streams[q],
                           deviceA[q],
                           deviceB[q],
                           deviceC[q],
                           WIDTH,
                           HEIGHT);

        hipLaunchKernelGGL(multiply_kernel,
                           dim3(WIDTH / threads_per_block.x, HEIGHT / threads_per_block.y),
                           dim3(threads_per_block.x, threads_per_block.y),
                           0,
                           streams[q],
                           deviceA[q],
                           deviceB[q],
                           deviceC[q],
                           WIDTH,
                           HEIGHT);

        hipLaunchKernelGGL(divide_kernel,
                           dim3(WIDTH / threads_per_block.x, HEIGHT / threads_per_block.y),
                           dim3(threads_per_block.x, threads_per_block.y),
                           0,
                           streams[q],
                           deviceB[q],
                           deviceA[q],
                           deviceC[q],
                           WIDTH,
                           HEIGHT);
    }

    sync_streams();

    for(int q = 0; q < num_queue; q++)
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

    auto   t2   = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    print_lock.lock();
    std::cout << "[vector-ops] Runtime of vector-ops is " << time << " sec\n";
    print_lock.unlock();
}
}  // namespace

void
run_vector_ops(int num_threads, int num_queue)
{
    int device_count = 0;
    HIP_API_CALL(hipGetDeviceCount(&device_count));

    if(device_count == 0) throw std::runtime_error{"No HIP devices found"};

    num_threads = std::max<int>(num_threads, 1);
    num_queue   = std::max<int>(num_queue, 1);

    auto _threads = std::vector<std::thread>{};
    _threads.reserve(num_threads);

    for(int i = 0; i < num_threads; ++i)
        _threads.emplace_back(run_vector_ops_impl, num_queue, i % device_count);

    for(auto& itr : _threads)
        itr.join();
}

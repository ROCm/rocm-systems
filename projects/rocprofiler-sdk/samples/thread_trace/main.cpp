// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>

#include <rocprofiler-sdk-roctx/roctx.h>

#include "hip/hip_runtime.h"

#define DATA_SIZE (304 * 64 * 4 * 3)
#define LDS_SIZE  1024

#define HIP_API_CALL(CALL)                                                                         \
    if((CALL) != hipSuccess)                                                                       \
    {                                                                                              \
        abort();                                                                                   \
    }

__global__ void
divide_kernel(float* a, const float* b, const float* c, int loopcnt)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= DATA_SIZE) return;

    float bdx = b[index];
    float cdx = c[index];

    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < loopcnt; j++)
        {
            float cdx2 = cdx + 1;
            float bdx2 = bdx * 1.5f;
            bdx        = bdx2 / (cdx2 + 1);
            cdx        = cdx2 / (bdx2 + 1);
        }
    }

    a[index] = bdx + cdx;
}

__global__ void
looping_lds_kernel(float* a, const float* b, const float* c, int loopcount)
{
    __shared__ float interm[LDS_SIZE];
    size_t           index = blockDim.x * blockIdx.x + threadIdx.x;

    for(size_t i = index; i < DATA_SIZE; i += blockDim.x * gridDim.x)
        interm[threadIdx.x % LDS_SIZE] = b[index] + threadIdx.x;

    for(int it = 0; it < loopcount; it++)
    {
        __syncthreads();
        float value = interm[(it + threadIdx.x + LDS_SIZE / 2) % LDS_SIZE];
        __syncthreads();
        interm[threadIdx.x % LDS_SIZE] += value;
    }

    a[index] = interm[threadIdx.x % LDS_SIZE] + c[index];
}

class hipMemory
{
public:
    hipMemory(size_t size = DATA_SIZE)
    {
        HIP_API_CALL(hipMalloc(&ptr, size * sizeof(float)));
        HIP_API_CALL(hipMemset(ptr, 0, size * sizeof(float)));
    }
    ~hipMemory()
    {
        if(ptr) HIP_API_CALL(hipFree(ptr));
    }
    hipMemory(hipMemory&& other)
    {
        ptr       = other.ptr;
        other.ptr = nullptr;
    }
    float* ptr = nullptr;
};

class HipStream
{
public:
    HipStream() { HIP_API_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); }
    ~HipStream() { HIP_API_CALL(hipStreamDestroy(stream)); }

    hipStream_t stream;

    hipMemory src1{};
    hipMemory src2{};
    hipMemory dst{};
};

int
run(int device, double run_seconds, std::atomic<int>* count)
{
    HIP_API_CALL(hipSetDevice(device));

    std::array<HipStream, 3> streams{};
    const auto               fixed_iterations = streams.size() * 2;

    uint64_t iter = 0;

    auto launch = [&]() {
        auto& s      = streams.at(iter % streams.size());
        auto& kernel = (iter % 2 == 0) ? divide_kernel : looping_lds_kernel;
        hipLaunchKernelGGL(
            kernel, DATA_SIZE / 512, 512, 0, s.stream, s.dst.ptr, s.src1.ptr, s.src2.ptr, 10);
        HIP_API_CALL(hipGetLastError());
        iter++;
    };

    // Warmup so the trace covers steady-state work, not first-launch noise.
    for(uint64_t i = 0; i < fixed_iterations; i++)
        launch();
    HIP_API_CALL(hipDeviceSynchronize());

    roctxProfilerResume(0);

    count->fetch_sub(1);
    while(count->load() != 0)
        sched_yield();

    auto       start    = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                      std::chrono::duration<double>(run_seconds));

    while((run_seconds > 0.0 && std::chrono::steady_clock::now() < deadline) ||
          (run_seconds == 0.0 && iter < fixed_iterations))
    {
        launch();
    }

    roctxProfilerPause(0);

    return iter;
}

int
main(int /*argc*/, char** /*argv*/)
{
    int dev_count = 1;

    // RUN_SECONDS unset or 0 -> one short alternating pass. Positive value
    // -> the same alternating pass repeated until the time budget expires.
    double run_seconds = 0.0;
    if(const char* env = std::getenv("RUN_SECONDS"))
    {
        char*  end = nullptr;
        double v   = std::strtod(env, &end);
        if(end != env && v > 0.0)
        {
            run_seconds = v;
            HIP_API_CALL(hipGetDeviceCount(&dev_count));
        }
    }

    auto             start = std::chrono::steady_clock::now();
    std::atomic<int> thread_count{dev_count};
    auto             threads = std::vector<std::future<int>>{};
    for(int i = 0; i < dev_count; i++)
        threads.push_back(std::async(std::launch::async, run, i, run_seconds, &thread_count));

    int iter = 0;
    for(auto& thread : threads)
        iter += thread.get();

    std::cout << "[main] launched " << iter << " kernels over "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << "s" << std::endl;
    return 0;
}

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

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <rocprofiler-sdk-roctx/roctx.h>

#include "hip/hip_runtime.h"

// Two waves per SIMD on MI300
#define DATA_SIZE (304 * 64 * 4 * 2)
#define HIP_API_CALL(CALL)                                                                         \
    if((CALL) != hipSuccess)                                                                       \
    {                                                                                              \
        abort();                                                                                   \
    }

#define LDS_SIZE 1024

__global__ void
divide_kernel(float* a, const float* b, const float* c, int /* unused */)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= DATA_SIZE) return;

    a[index] = (b[index] - c[index]) / abs(c[index] + b[index]) + 1;
}

__global__ void
looping_lds_kernel(float* a, const float* b, const float* c, int loopcount)
{
    __shared__ float interm[LDS_SIZE];

    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

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

__global__ void
fifo_kernel(float* /* a */, const float* /* b */, const float* /* c */, int loops)
{
    using _float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    __shared__ _float4 lds[LDS_SIZE];
    lds[threadIdx.x]       = _float4{float(threadIdx.x)};
    lds[threadIdx.x + 512] = _float4{float(threadIdx.x)};

    __syncthreads();

    _float4 dst[16];

    float res1 = 0, res2 = 0;

    for(int l = 0; l < loops; l++)
    {
#pragma unroll 16
        for(int i = 0; i < 16; i++)
            dst[i] = lds[threadIdx.x + i * 8];

        __syncthreads();

#pragma unroll 16
        for(int i = 0; i < 16; i++)
        {
            res1 += dst[i][0] + dst[i][1];
            res2 += dst[i][2] + dst[i][3];
        }
        asm volatile("v_add_f32 %0, %1, %2" : "=v"(res1) : "v"(res1), "v"(res2));
    }
};

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

#define Launch(kernel, stream, arglast)                                                            \
    hipLaunchKernelGGL(                                                                            \
        kernel, DATA_SIZE / 512, 512, 0, 0, stream.dst.ptr, stream.src1.ptr, stream.src2.ptr, 6);

int
main(int /*argc*/, char** /*argv*/)
{
    std::array<HipStream, 3>              streams{};
    std::vector<decltype(divide_kernel)*> kernels{};

    kernels.push_back(divide_kernel);
    kernels.push_back(looping_lds_kernel);
    kernels.push_back(fifo_kernel);

    for(size_t i = 0; i < streams.size() * kernels.size(); i++)
    {
        // Warmup then start
        if(i == streams.size())
        {
            HIP_API_CALL(hipDeviceSynchronize());
            roctxProfilerResume(0);
        }

        auto& stream = streams.at(i % streams.size());
        auto& kernel = kernels.at(i % kernels.size());

        Launch(kernel, stream, 3);
        HIP_API_CALL(hipGetLastError());
    }

    HIP_API_CALL(hipDeviceSynchronize());
    roctxProfilerPause(0);

    return 0;
}

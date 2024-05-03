// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file samples/code_object_isa_decode/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "hip/hip_runtime.h"

// Three waves per SIMD on MI300
#define DATA_SIZE          (304 * 64 * 4 * 3)
#define HIP_API_CALL(CALL) assert((CALL) == hipSuccess)

template <typename T>
__global__ void
branching_kernel(T* __restrict__ a,
                 const float* __restrict__ b,
                 const float* __restrict__ c,
                 int size)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(blockIdx.x % 2 == 0)
        a[index] = b[index] + c[index];
    else
        a[index] = b[index] * c[index] - 2.0f;
}

class hipMemory
{
public:
    hipMemory(size_t size)
    {
        HIP_API_CALL(hipMalloc(&ptr, size * sizeof(float)));
        HIP_API_CALL(hipMemset(ptr, 0, size * sizeof(float)));
    }
    ~hipMemory()
    {
        if(ptr) HIP_API_CALL(hipFree(ptr));
    }
    float* ptr = nullptr;
};

int
main(int argc, char** argv)
{
    hipMemory src1(DATA_SIZE);
    hipMemory src2(DATA_SIZE);
    hipMemory dst(DATA_SIZE);

    hipLaunchKernelGGL(branching_kernel,
                       dim3(DATA_SIZE / 64),
                       dim3(64),
                       0,
                       0,
                       dst.ptr,
                       src1.ptr,
                       src2.ptr,
                       DATA_SIZE);

    HIP_API_CALL(hipGetLastError());
    HIP_API_CALL(hipDeviceSynchronize());

    return 0;
}
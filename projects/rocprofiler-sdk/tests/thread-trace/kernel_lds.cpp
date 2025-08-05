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
#include "hip/hip_runtime.h"

#define SHM_SIZE 64

__global__ void
looping_lds_kernel(float* __restrict__ a,
                   const float* __restrict__ b,
                   const float* __restrict__ c,
                   size_t size,
                   size_t loopcount)
{
    __shared__ float interm[SHM_SIZE];

    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    for(size_t i = index; i < size; i += blockDim.x * gridDim.x)
        interm[threadIdx.x % SHM_SIZE] = b[index] + threadIdx.x;

    for(size_t it = 0; it < loopcount; it++)
    {
        __syncthreads();
        float value = interm[(it + threadIdx.x + SHM_SIZE / 2) % SHM_SIZE];
        __syncthreads();
        interm[threadIdx.x % SHM_SIZE] += value;
    }

    a[index] = interm[threadIdx.x % SHM_SIZE] + c[index];
}

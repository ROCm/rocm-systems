// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <hip/hip_runtime.h>

// VecAdd: simple element-wise add into a separate output buffer. Baseline replay correctness.
__global__ void
vecAdd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) c[i] = a[i] + b[i];
}

// SAXPY: in-place read-write kernel (y is both read and written). Each pass overwrites y, so a
// correct restore between passes must fully revert it for the result to match across passes.
__global__ void
saxpy(float alpha, const float* __restrict__ x, float* __restrict__ y, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) y[i] = alpha * x[i] + y[i];
}

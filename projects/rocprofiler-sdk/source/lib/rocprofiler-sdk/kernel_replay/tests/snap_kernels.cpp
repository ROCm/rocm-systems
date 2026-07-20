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

// This is the only HIP translation unit in the snapshot/restore test. It is built into its own
// static library so the test executable can link as CXX (see snap_kernels.hpp for why).

#include "snap_kernels.hpp"

#include <hip/hip_runtime.h>

namespace
{
__global__ void
fill_kernel(float* d, float val, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        d[i] = val;
}

__global__ void
iota_kernel(float* d, float base, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        d[i] = base + static_cast<float>(i);
}

__global__ void
saxpy_kernel(float* y, const float* x, float a, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        y[i] = a * x[i] + y[i];
}

__global__ void
add_kernel(float* d, float delta, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        d[i] = d[i] + delta;
}

constexpr int NUM_THREADS = 1024;

int
blocks_for(int n)
{
    return (n + NUM_THREADS - 1) / NUM_THREADS;
}
}  // namespace

namespace kernel_launch
{
void
fill(float* d, float val, int n)
{
    fill_kernel<<<blocks_for(n), NUM_THREADS>>>(d, val, n);
}

void
iota(float* d, float base, int n)
{
    iota_kernel<<<blocks_for(n), NUM_THREADS>>>(d, base, n);
}

void
saxpy(float* y, const float* x, float a, int n)
{
    saxpy_kernel<<<blocks_for(n), NUM_THREADS>>>(y, x, a, n);
}

void
add(float* d, float delta, int n)
{
    add_kernel<<<blocks_for(n), NUM_THREADS>>>(d, delta, n);
}
}  // namespace kernel_launch

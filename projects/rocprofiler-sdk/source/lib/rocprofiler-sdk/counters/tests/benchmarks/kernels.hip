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
#include <cstdint>

// VecAdd: simplest possible kernel, minimal GPU memory footprint.
// Validates replay correctness for the most basic dispatch pattern.
__global__ void
vecAdd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) c[i] = a[i] + b[i];
}

// SAXPY: single-pass read-write kernel. Output array is both read and written,
// so replay restore must correctly revert the partial overwrites.
__global__ void
saxpy(float alpha, const float* __restrict__ x, float* __restrict__ y, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) y[i] = alpha * x[i] + y[i];
}

// MatMul: tiled shared-memory matrix multiply. Exercises a larger working set
// with shared memory traffic -- stresses the snapshot/restore DMA path.
// C = A * B, all row-major, dim x dim.
__global__ void
matMul(const float* __restrict__ A,
       const float* __restrict__ B,
       float* __restrict__       C,
       int                       dim)
{
    constexpr int TILE = 16;
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    for(int t = 0; t < dim; t += TILE)
    {
        sA[threadIdx.y][threadIdx.x] =
            (row < dim && t + threadIdx.x < dim) ? A[row * dim + t + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (t + threadIdx.y < dim && col < dim) ? B[(t + threadIdx.y) * dim + col] : 0.0f;
        __syncthreads();

        for(int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if(row < dim && col < dim) C[row * dim + col] = sum;
}

// Stencil1D: 5-point stencil on a 1D array. Reads from a neighborhood --
// exercises the case where adjacent allocations are read-only and only
// the output allocation is dirty (validates selective dirty-page restore).
__global__ void
stencil1D(const float* __restrict__ in, float* __restrict__ out, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= 2 && i < n - 2)
        out[i] = 0.0625f * in[i - 2] + 0.25f * in[i - 1] + 0.375f * in[i] +
                 0.25f * in[i + 1] + 0.0625f * in[i + 2];
}

// Reduction: parallel sum reduction. Single output value from a large input --
// tests replay where the dirty footprint is tiny (one element) relative to the
// snapshot size (Section 4.3: dirty-page diffing should skip most pages).
__global__ void
reduceSum(const float* __restrict__ in, float* __restrict__ out, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;
    if(i < n) val = in[i];
    if(i + blockDim.x < n) val += in[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if(tid == 0) atomicAdd(out, sdata[0]);
}

// MultiBuffer: touches many separate allocations in a single kernel.
// Validates that the inventory tracks all live allocations and snapshot/restore
// handles many regions correctly (Kerncap design: flat inventory, no per-agent
// partitioning). Reads from N input buffers, writes to one output buffer.
__global__ void
multiBufferGather(const float* const* __restrict__ inputs,
                  float* __restrict__              output,
                  int                              n_per_buf,
                  int                              n_bufs)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n_per_buf) return;

    float sum = 0.0f;
    for(int b = 0; b < n_bufs; ++b)
        sum += inputs[b][i];
    output[i] = sum;
}

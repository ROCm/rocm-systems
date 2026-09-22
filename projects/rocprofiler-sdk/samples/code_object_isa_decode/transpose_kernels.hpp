// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "hip/hip_runtime.h"

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            lock_guard_t _hip_api_print_lk{print_lock};                                            \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

#define TILE_DIM 64

template <typename T>
__global__ void
transposeNaive(T* odata, const T* idata, size_t size)
{
    size_t idx        = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t block_posy = blockIdx.y * TILE_DIM;

    for(size_t idy = threadIdx.y; idy < TILE_DIM; idy += blockDim.y)
        odata[size * idx + block_posy + idy] = idata[idx + (block_posy + idy) * size];
}

template <typename T>
__global__ void
transposeLdsNoBankConflicts(T* odata, const T* idata, size_t size)
{
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    size_t idx_in   = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t idy_in   = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t index_in = idx_in + idy_in * size;

    size_t idx_out   = blockIdx.y * TILE_DIM + threadIdx.x;
    size_t idy_out   = blockIdx.x * TILE_DIM + threadIdx.y;
    size_t index_out = idx_out + idy_out * size;

    for(size_t y = 0; y < TILE_DIM; y += blockDim.y)
        tile[threadIdx.y + y][threadIdx.x] = idata[index_in + y * size];

    __syncthreads();

    for(size_t y = 0; y < TILE_DIM; y += blockDim.y)
        odata[index_out + y * size] = tile[threadIdx.x][threadIdx.y + y];
}

// Generates more interesting ISA
template <typename T>
__global__ void
transposeLdsSwapInplace(T* odata, const T* idata, size_t size)
{
    __shared__ T tile[TILE_DIM][TILE_DIM];

    const size_t idx_in = blockIdx.x * TILE_DIM + threadIdx.x;

    for(size_t idy = threadIdx.y; idy < TILE_DIM; idy += blockDim.y)
        tile[idy][threadIdx.x] = idata[idx_in + (idy + blockIdx.y * TILE_DIM) * size];

    __syncthreads();

    for(size_t idy = threadIdx.y; idy < TILE_DIM; idy += blockDim.y)
        if(idy < threadIdx.x)
        {
            T temp                 = tile[idy][threadIdx.x];
            tile[idy][threadIdx.x] = tile[threadIdx.x][idy];
            tile[threadIdx.x][idy] = temp;
        }

    __syncthreads();

    const size_t idx_out = blockIdx.y * TILE_DIM + threadIdx.x;

    for(size_t idy = threadIdx.y; idy < TILE_DIM; idy += blockDim.y)
        odata[(blockIdx.x * TILE_DIM + idy) * size + idx_out] = tile[idy][threadIdx.x];
}

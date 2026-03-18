// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#define HIP_CHECK(call)                                                                  \
    do                                                                                   \
    {                                                                                    \
        hipError_t err = call;                                                           \
        if(err != hipSuccess)                                                            \
        {                                                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, \
                    __LINE__);                                                           \
            exit(1);                                                                     \
        }                                                                                \
    } while(0)

constexpr int KERNEL_ELEMENTS = 1024 * 1024;
constexpr int KERNEL_ITERS    = 1000;
constexpr int BLOCK_SIZE      = 256;
constexpr int GRID_SIZE       = (KERNEL_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

#define DEFINE_KERNEL(name, offset)                                                      \
    __global__ void name(float* data, int n)                                             \
    {                                                                                    \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                 \
        if(idx < n)                                                                      \
            for(int i = 0; i < KERNEL_ITERS; ++i)                                        \
                data[idx] = __sinf(data[idx] + float(offset));                           \
    }

#define LAUNCH_KERNEL(name, data)                                                        \
    do                                                                                   \
    {                                                                                    \
        name<<<GRID_SIZE, BLOCK_SIZE>>>(data, KERNEL_ELEMENTS);                          \
        HIP_CHECK(hipDeviceSynchronize());                                               \
    } while(0)

struct gpu_buffer
{
    float* d_data = nullptr;

    gpu_buffer()
    {
        HIP_CHECK(hipMalloc(&d_data, KERNEL_ELEMENTS * sizeof(float)));
        HIP_CHECK(hipMemset(d_data, 0, KERNEL_ELEMENTS * sizeof(float)));
    }

    ~gpu_buffer() { hipFree(d_data); }

    gpu_buffer(const gpu_buffer&)            = delete;
    gpu_buffer& operator=(const gpu_buffer&) = delete;

    float* get() { return d_data; }
};

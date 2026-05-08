// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

namespace kernels
{
__global__ void
copy(float* dst, const float* src, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) dst[idx] = src[idx];
}

__global__ void
atomic(float* dst, const float* src, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) atomicAdd(dst + (idx % blockDim.x), src[idx]);
}

__global__ void
iops1()
{
    asm volatile("v_fma_f32 v3, v1, v2, v3");
    asm volatile("v_add_f64 v[0:1], v[2:3], v[4:5]");
    asm volatile("v_fma_f64 v[0:1], v[2:3], v[4:5], v[6:7]");
    asm volatile("v_fma_f64 v[0:1], v[2:3], v[4:5], v[6:7]");
}

__global__ void
iops2()
{
    asm volatile("v_add_f32 v4, v5, v6");
    asm volatile("v_fma_f64 v[0:1], v[0:1], v[2:3], v[4:5]");
}

__global__ void
iops_trans()
{
    asm volatile("v_cos_f32 v3, v3; v_cos_f32 v4, v4");
}
}  // namespace kernels

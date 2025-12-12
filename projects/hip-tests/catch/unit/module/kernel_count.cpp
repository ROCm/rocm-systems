/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

extern "C" __device__ void hello_world_2(float* a, float* b) {
    int tx = threadIdx.x;
    b[tx] = a[tx];
}

extern "C" __global__ void hello_world(float* a, float* b) {
    int tx = threadIdx.x;
    b[tx] = a[tx];
}

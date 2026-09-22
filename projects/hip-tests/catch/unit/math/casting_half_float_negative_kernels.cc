/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_fp16.h>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_F2H_KERNELS_SHELL(func_name)                                                      \
  __global__ void func_name##_kernel_v1(__half* result, float* x) { *result = func_name(x); }      \
  __global__ void func_name##_kernel_v2(__half* result, Dummy x) { *result = func_name(x); }       \
  __global__ void func_name##_kernel_v3(Dummy* result, float x) { *result = func_name(x); }

#define NEGATIVE_H2F_KERNELS_SHELL(func_name)                                                      \
  __global__ void func_name##_kernel_v1(float* result, __half* x) { *result = func_name(x); }      \
  __global__ void func_name##_kernel_v2(float* result, Dummy x) { *result = func_name(x); }        \
  __global__ void func_name##_kernel_v3(Dummy* result, __half x) { *result = func_name(x); }

NEGATIVE_F2H_KERNELS_SHELL(__float2half_rd)
NEGATIVE_F2H_KERNELS_SHELL(__float2half_rn)
NEGATIVE_F2H_KERNELS_SHELL(__float2half_ru)
NEGATIVE_F2H_KERNELS_SHELL(__float2half_rz)
NEGATIVE_F2H_KERNELS_SHELL(__float2half)

NEGATIVE_H2F_KERNELS_SHELL(__half2float)
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_KERNELS_SHELL(func_name, T)                                                       \
  __global__ void func_name##_kernel_v1(T* result, float* x) { *result = func_name(x); }           \
  __global__ void func_name##_kernel_v2(T* result, Dummy x) { *result = func_name(x); }            \
  __global__ void func_name##_kernel_v3(Dummy* result, float x) { *result = func_name(x); }

NEGATIVE_KERNELS_SHELL(__float2int_rd, int)
NEGATIVE_KERNELS_SHELL(__float2int_rn, int)
NEGATIVE_KERNELS_SHELL(__float2int_ru, int)
NEGATIVE_KERNELS_SHELL(__float2int_rz, int)
NEGATIVE_KERNELS_SHELL(__float2uint_rd, unsigned int)
NEGATIVE_KERNELS_SHELL(__float2uint_rn, unsigned int)
NEGATIVE_KERNELS_SHELL(__float2uint_ru, unsigned int)
NEGATIVE_KERNELS_SHELL(__float2uint_rz, unsigned int)
NEGATIVE_KERNELS_SHELL(__float2ll_rd, long long int)
NEGATIVE_KERNELS_SHELL(__float2ll_rn, long long int)
NEGATIVE_KERNELS_SHELL(__float2ll_ru, long long int)
NEGATIVE_KERNELS_SHELL(__float2ll_rz, long long int)
NEGATIVE_KERNELS_SHELL(__float2ull_rd, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__float2ull_rn, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__float2ull_ru, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__float2ull_rz, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__float_as_int, int)
NEGATIVE_KERNELS_SHELL(__float_as_uint, unsigned int)
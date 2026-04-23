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

#define NEGATIVE_KERNELS_SHELL(func_name, T)                                                       \
  __global__ void func_name##_kernel_v1(T* result, __half* x) { *result = func_name(x); }          \
  __global__ void func_name##_kernel_v2(T* result, Dummy x) { *result = func_name(x); }            \
  __global__ void func_name##_kernel_v3(Dummy* result, __half x) { *result = unc_name(x); }

NEGATIVE_KERNELS_SHELL(__half2int_rn, int)
NEGATIVE_KERNELS_SHELL(__half2int_rz, int)
NEGATIVE_KERNELS_SHELL(__half2int_rd, int)
NEGATIVE_KERNELS_SHELL(__half2int_ru, int)
NEGATIVE_KERNELS_SHELL(__half2uint_rn, unsigned int)
NEGATIVE_KERNELS_SHELL(__half2uint_rz, unsigned int)
NEGATIVE_KERNELS_SHELL(__half2uint_rd, unsigned int)
NEGATIVE_KERNELS_SHELL(__half2uint_ru, unsigned int)
NEGATIVE_KERNELS_SHELL(__half2short_rn, short)
NEGATIVE_KERNELS_SHELL(__half2short_rz, short)
NEGATIVE_KERNELS_SHELL(__half2short_rd, short)
NEGATIVE_KERNELS_SHELL(__half2short_ru, short)
NEGATIVE_KERNELS_SHELL(__half_as_short, short)
NEGATIVE_KERNELS_SHELL(__half2ushort_rn, unsigned short)
NEGATIVE_KERNELS_SHELL(__half2ushort_rz, unsigned short)
NEGATIVE_KERNELS_SHELL(__half2ushort_rd, unsigned short)
NEGATIVE_KERNELS_SHELL(__half2ushort_ru, unsigned short)
NEGATIVE_KERNELS_SHELL(__half_as_ushort, unsigned short)
NEGATIVE_KERNELS_SHELL(__half2ll_rn, long long)
NEGATIVE_KERNELS_SHELL(__half2ll_rz, long long)
NEGATIVE_KERNELS_SHELL(__half2ll_rd, long long)
NEGATIVE_KERNELS_SHELL(__half2ll_ru, long long)
NEGATIVE_KERNELS_SHELL(__half2ull_rn, unsigned long long)
NEGATIVE_KERNELS_SHELL(__half2ull_rz, unsigned long long)
NEGATIVE_KERNELS_SHELL(__half2ull_rd, unsigned long long)
NEGATIVE_KERNELS_SHELL(__half2ull_ru, unsigned long long)
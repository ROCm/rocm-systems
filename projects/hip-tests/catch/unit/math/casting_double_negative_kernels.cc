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
  __global__ void func_name##_kernel_v1(T* result, double* x) { *result = func_name(x); }          \
  __global__ void func_name##_kernel_v2(T* result, Dummy x) { *result = func_name(x); }            \
  __global__ void func_name##_kernel_v3(Dummy* result, double x) { *result = func_name(x); }

NEGATIVE_KERNELS_SHELL(__double2int_rd, int)
NEGATIVE_KERNELS_SHELL(__double2int_rn, int)
NEGATIVE_KERNELS_SHELL(__double2int_ru, int)
NEGATIVE_KERNELS_SHELL(__double2int_rz, int)
NEGATIVE_KERNELS_SHELL(__double2uint_rd, unsigned int)
NEGATIVE_KERNELS_SHELL(__double2uint_rn, unsigned int)
NEGATIVE_KERNELS_SHELL(__double2uint_ru, unsigned int)
NEGATIVE_KERNELS_SHELL(__double2uint_rz, unsigned int)
NEGATIVE_KERNELS_SHELL(__double2ll_rd, long long int)
NEGATIVE_KERNELS_SHELL(__double2ll_rn, long long int)
NEGATIVE_KERNELS_SHELL(__double2ll_ru, long long int)
NEGATIVE_KERNELS_SHELL(__double2ll_rz, long long int)
NEGATIVE_KERNELS_SHELL(__double2ull_rd, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__double2ull_rn, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__double2ull_ru, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__double2ull_rz, unsigned long long int)
NEGATIVE_KERNELS_SHELL(__double2float_rd, float)
NEGATIVE_KERNELS_SHELL(__double2float_rn, float)
NEGATIVE_KERNELS_SHELL(__double2float_ru, float)
NEGATIVE_KERNELS_SHELL(__double2float_rz, float)
NEGATIVE_KERNELS_SHELL(__double2hiint, int)
NEGATIVE_KERNELS_SHELL(__double2loint, int)
NEGATIVE_KERNELS_SHELL(__double_as_longlong, long long int)
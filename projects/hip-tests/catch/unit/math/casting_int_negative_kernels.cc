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

#define NEGATIVE_KERNELS_SHELL_ONE_ARG(func_name, T1, T2)                                          \
  __global__ void func_name##_kernel_v1(T1* result, T2* x) { *result = func_name(x); }             \
  __global__ void func_name##_kernel_v2(T1* result, Dummy x) { *result = func_name(x); }           \
  __global__ void func_name##_kernel_v3(Dummy* result, T2 x) { *result = func_name(x); }

#define NEGATIVE_KERNELS_SHELL_TWO_ARGS(func_name, T1, T2)                                         \
  __global__ void func_name##_kernel_v1(T1* result, T2* x, T2 y) { *result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v2(T1* result, T2 x, T2* y) { *result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v3(T1* result, Dummy x, T2 y) { *result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v4(T1* result, T2 x, Dummy y) { *result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v5(Dummy* result, T2 x, T2 y) { *result = func_name(x, y); }

NEGATIVE_KERNELS_SHELL_ONE_ARG(__int2float_rd, float, int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__int2float_rn, float, int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__int2float_ru, float, int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__int2float_rz, float, int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__uint2float_rd, float, unsigned int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__uint2float_rn, float, unsigned int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__uint2float_ru, float, unsigned int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__uint2float_rz, float, unsigned int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ll2float_rd, float, long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ll2float_rn, float, long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ll2float_ru, float, long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ll2float_rz, float, long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ull2float_rd, float, unsigned long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ull2float_rn, float, unsigned long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ull2float_ru, float, unsigned long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ull2float_rz, float, unsigned long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__int2double_rn, double, int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__uint2double_rn, double, unsigned int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ll2double_rd, double, long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ll2double_rn, double, long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ll2double_ru, double, long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ll2double_rz, double, long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ull2double_rd, double, unsigned long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ull2double_rn, double, unsigned long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ull2double_ru, double, unsigned long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__ull2double_rz, double, unsigned long long int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__int_as_float, float, int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__uint_as_float, float, unsigned int)
NEGATIVE_KERNELS_SHELL_ONE_ARG(__longlong_as_double, double, long long int)
NEGATIVE_KERNELS_SHELL_TWO_ARGS(__hiloint2double, double, int)
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

#define NEGATIVE_KERNELS_SHELL_ONE_ARG(func_name)                                                  \
  __global__ void func_name##_kernel_v1(double* x) { double result = func_name(x); }               \
  __global__ void func_name##_kernel_v2(Dummy x) { double result = func_name(x); }                 \
  __global__ void func_name##f_kernel_v1(float* x) { float result = func_name##f(x); }             \
  __global__ void func_name##f_kernel_v2(Dummy x) { float result = func_name##f(x); }

#define NEGATIVE_KERNELS_SHELL_TWO_ARGS(func_name)                                                 \
  __global__ void func_name##_kernel_v1(int* x, double y) { double result = func_name(x, y); }     \
  __global__ void func_name##_kernel_v2(int x, double* y) { double result = func_name(x, y); }     \
  __global__ void func_name##_kernel_v3(Dummy x, double y) { double result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v4(int x, Dummy y) { double result = func_name(x, y); }       \
  __global__ void func_name##f_kernel_v1(int* x, float y) { float result = func_name##f(x, y); }   \
  __global__ void func_name##f_kernel_v2(int x, float* y) { float result = func_name##f(x, y); }   \
  __global__ void func_name##f_kernel_v3(Dummy x, float y) { float result = func_name##f(x, y); }  \
  __global__ void func_name##f_kernel_v4(int x, Dummy y) { float result = func_name##f(x, y); }

NEGATIVE_KERNELS_SHELL_ONE_ARG(erf)
NEGATIVE_KERNELS_SHELL_ONE_ARG(erfc)
NEGATIVE_KERNELS_SHELL_ONE_ARG(erfinv)
NEGATIVE_KERNELS_SHELL_ONE_ARG(erfcinv)
NEGATIVE_KERNELS_SHELL_ONE_ARG(erfcx)
NEGATIVE_KERNELS_SHELL_ONE_ARG(normcdf)
NEGATIVE_KERNELS_SHELL_ONE_ARG(normcdfinv)
NEGATIVE_KERNELS_SHELL_ONE_ARG(lgamma)
NEGATIVE_KERNELS_SHELL_ONE_ARG(tgamma)
NEGATIVE_KERNELS_SHELL_ONE_ARG(j0)
NEGATIVE_KERNELS_SHELL_ONE_ARG(j1)
NEGATIVE_KERNELS_SHELL_TWO_ARGS(jn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(y0)
NEGATIVE_KERNELS_SHELL_ONE_ARG(y1)
NEGATIVE_KERNELS_SHELL_TWO_ARGS(yn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(cyl_bessel_i0)
NEGATIVE_KERNELS_SHELL_ONE_ARG(cyl_bessel_i1)

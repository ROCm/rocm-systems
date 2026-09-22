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

#define INTRINSIC_UNARY_DOUBLE_NEGATIVE_KERNELS(func_name)                                         \
  __global__ void func_name##_kernel_v1(double* x) { double result = func_name(x); }               \
  __global__ void func_name##_kernel_v2(Dummy x) { double result = func_name(x); }

#define INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(func_name)                                        \
  __global__ void func_name##_kernel_v1(double* x, double y) { double result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v2(double x, double* y) { double result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v3(Dummy x, double y) { double result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v4(double x, Dummy y) { double result = func_name(x, y); }


INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(__dadd_rn)
INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(__dsub_rn)
INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(__dmul_rn)
INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(__ddiv_rn)
INTRINSIC_UNARY_DOUBLE_NEGATIVE_KERNELS(__dsqrt_rn)
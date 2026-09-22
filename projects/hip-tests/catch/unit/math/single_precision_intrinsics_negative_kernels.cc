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

#define INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(func_name)                                          \
  __global__ void func_name##_kernel_v1(float* x) { float result = func_name(x); }                 \
  __global__ void func_name##_kernel_v2(Dummy x) { float result = func_name(x); }

#define INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(func_name)                                         \
  __global__ void func_name##_kernel_v1(float* x, float y) { float result = func_name(x, y); }     \
  __global__ void func_name##_kernel_v2(float x, float* y) { float result = func_name(x, y); }     \
  __global__ void func_name##_kernel_v3(Dummy x, float y) { float result = func_name(x, y); }      \
  __global__ void func_name##_kernel_v4(float x, Dummy y) { float result = func_name(x, y); }

INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__fsqrt_rn)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__expf)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__exp10f)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__logf)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__log2f)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__log10f)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__sinf)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__cosf)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__tanf)

INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fadd_rn)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fsub_rn)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fmul_rn)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fdiv_rn)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fdividef)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__powf)
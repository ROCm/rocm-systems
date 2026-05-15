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

#define INTRINSIC_UNARY_INT_NEGATIVE_KERNELS(func_name)                                            \
  __global__ void func_name##_kernel_v1(int* x) { int result = func_name(x); }                     \
  __global__ void func_name##_kernel_v2(Dummy x) { int result = func_name(x); }

#define INTRINSIC_UNARY_LONGLONG_NEGATIVE_KERNELS(func_name)                                       \
  __global__ void func_name##_kernel_v1(long long int* x) { long long int result = func_name(x); } \
  __global__ void func_name##_kernel_v2(Dummy x) { long long int result = func_name(x); }

#define INTRINSIC_BINARY_INT_NEGATIVE_KERNELS(func_name)                                           \
  __global__ void func_name##_kernel_v1(int* x, int y) { int result = func_name(x, y); }           \
  __global__ void func_name##_kernel_v2(int x, int* y) { int result = func_name(x, y); }           \
  __global__ void func_name##_kernel_v3(Dummy x, int y) { int result = func_name(x, y); }          \
  __global__ void func_name##_kernel_v4(int x, Dummy y) { int result = func_name(x, y); }

#define INTRINSIC_BINARY_LONGLONG_NEGATIVE_KERNELS(func_name)                                      \
  __global__ void func_name##_kernel_v1(long long int* x, long long int y) {                       \
    long long int result = func_name(x, y);                                                        \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(long long int x, long long int* y) {                       \
    long long int result = func_name##(x, y);                                                      \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(Dummy x, long long int y) {                                \
    long long int result = func_name##(x, y);                                                      \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(long long int x, Dummy y) {                                \
    long long int result = func_name##(x, y);                                                      \
  }

INTRINSIC_UNARY_INT_NEGATIVE_KERNELS(__brev)
INTRINSIC_UNARY_INT_NEGATIVE_KERNELS(__clz)
INTRINSIC_UNARY_INT_NEGATIVE_KERNELS(__ffs)
INTRINSIC_UNARY_INT_NEGATIVE_KERNELS(__popc)
INTRINSIC_UNARY_LONGLONG_NEGATIVE_KERNELS(__brevll)
INTRINSIC_UNARY_LONGLONG_NEGATIVE_KERNELS(__clzll)
INTRINSIC_UNARY_LONGLONG_NEGATIVE_KERNELS(__ffsll)
INTRINSIC_UNARY_LONGLONG_NEGATIVE_KERNELS(__popcll)
INTRINSIC_BINARY_INT_NEGATIVE_KERNELS(__mul24)
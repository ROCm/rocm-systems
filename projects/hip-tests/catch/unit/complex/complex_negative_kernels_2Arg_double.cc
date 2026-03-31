/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_complex.h>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_SHELL_TWO_ARG_DOUBLE(func_name)                                                   \
  __global__ void func_name##_kernel_v1(hipDoubleComplex* result, hipDoubleComplex* x,             \
                                        hipDoubleComplex y) {                                      \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(hipDoubleComplex* result, hipDoubleComplex x,              \
                                        hipDoubleComplex* y) {                                     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {  \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {  \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v5(hipDoubleComplex* result, hipFloatComplex x,               \
                                        hipDoubleComplex y) {                                      \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(hipDoubleComplex* result, hipDoubleComplex x,              \
                                        hipFloatComplex y) {                                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {   \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {   \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {  \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v10(hipFloatComplex* result, hipDoubleComplex x,              \
                                         hipDoubleComplex y) {                                     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {  \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v1(hipDoubleComplex* result, hipDoubleComplex* x, hipDoubleComplex y) {         \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v2(hipDoubleComplex* result, hipDoubleComplex x, hipDoubleComplex* y) {         \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v3(hipDoubleComplex* result, double x, hipDoubleComplex y) {                    \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v4(hipDoubleComplex* result, hipDoubleComplex x, double y) {                    \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v5(hipDoubleComplex* result, hipFloatComplex x, hipDoubleComplex y) {           \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v6(hipDoubleComplex* result, hipDoubleComplex x, hipFloatComplex y) {           \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v7(hipDoubleComplex* result, Dummy x, hipDoubleComplex y) {                     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v8(hipDoubleComplex* result, hipDoubleComplex x, Dummy y) {                     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v9(double* result, hipDoubleComplex x, hipDoubleComplex y) {                    \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v10(hipFloatComplex* result, hipDoubleComplex x, hipDoubleComplex y) {          \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v11(Dummy* result, hipDoubleComplex x, hipDoubleComplex y) {                    \
    *result = func_name(x, y);                                                                     \
  }

NEGATIVE_SHELL_TWO_ARG_DOUBLE(hipCadd)
NEGATIVE_SHELL_TWO_ARG_DOUBLE(hipCsub)
NEGATIVE_SHELL_TWO_ARG_DOUBLE(hipCmul)
NEGATIVE_SHELL_TWO_ARG_DOUBLE(hipCdiv)
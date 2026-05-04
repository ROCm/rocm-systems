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

#define NEGATIVE_SHELL_TWO_ARG_FLOAT(func_name)                                                    \
  __global__ void func_name##_kernel_v1(hipFloatComplex* result, hipFloatComplex* x,               \
                                        hipFloatComplex y) {                                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(hipFloatComplex* result, hipFloatComplex x,                \
                                        hipFloatComplex* y) {                                      \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(hipFloatComplex* result, float x, hipFloatComplex y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(hipFloatComplex* result, hipFloatComplex x, float y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v5(hipFloatComplex* result, hipDoubleComplex x,               \
                                        hipFloatComplex y) {                                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(hipFloatComplex* result, hipFloatComplex x,                \
                                        hipDoubleComplex y) {                                      \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v9(float* result, hipFloatComplex x, hipFloatComplex y) {     \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v10(hipDoubleComplex* result, hipFloatComplex x,              \
                                         hipFloatComplex y) {                                      \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {    \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v1(hipFloatComplex* result, hipFloatComplex* x, hipFloatComplex y) {            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v2(hipFloatComplex* result, hipFloatComplex x, hipFloatComplex* y) {            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v3(hipFloatComplex* result, float x, hipFloatComplex y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v4(hipFloatComplex* result, hipFloatComplex x, float y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v5(hipFloatComplex* result, hipDoubleComplex x, hipFloatComplex y) {            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v6(hipFloatComplex* result, hipFloatComplex x, hipDoubleComplex y) {            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v7(hipFloatComplex* result, Dummy x, hipFloatComplex y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v8(hipFloatComplex* result, hipFloatComplex x, Dummy y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v9(float* result, hipFloatComplex x, hipFloatComplex y) {                       \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v10(hipDoubleComplex* result, hipFloatComplex x, hipFloatComplex y) {           \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v11(Dummy* result, hipFloatComplex x, hipFloatComplex y) {                      \
    *result = func_name(x, y);                                                                     \
  }

NEGATIVE_SHELL_TWO_ARG_FLOAT(hipCaddf)
NEGATIVE_SHELL_TWO_ARG_FLOAT(hipCsubf)
NEGATIVE_SHELL_TWO_ARG_FLOAT(hipCmulf)
NEGATIVE_SHELL_TWO_ARG_FLOAT(hipCdivf)
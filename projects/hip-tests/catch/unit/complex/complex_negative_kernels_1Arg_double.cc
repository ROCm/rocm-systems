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

#define NEGATIVE_SHELL_ONE_ARG_DOUBLE(func_name)                                                   \
  __global__ void func_name##_kernel_v1(double* result, hipDoubleComplex* x) {                     \
    *result = func_name(x);                                                                        \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(double* result, hipFloatComplex x) {                       \
    *result = func_name(x);                                                                        \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(double* result, double x) { *result = func_name(x); }      \
  __global__ void func_name##_kernel_v4(double* result, Dummy x) { *result = func_name(x); }       \
  __global__ void func_name##_kernel_v5(hipDoubleComplex* result, hipDoubleComplex x) {            \
    *result = func_name(x);                                                                        \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(Dummy* result, hipDoubleComplex x) {                       \
    *result = func_name(x);                                                                        \
  }                                                                                                \
  void func_name##_v1(double* result, hipDoubleComplex* x) { *result = func_name(x); }             \
  void func_name##_v2(double* result, hipFloatComplex x) { *result = func_name(x); }               \
  void func_name##_v3(double* result, double x) { *result = func_name(x); }                        \
  void func_name##_v4(double* result, Dummy x) { *result = func_name(x); }                         \
  void func_name##_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = func_name(x); }    \
  void func_name##_v6(Dummy* result, hipDoubleComplex x) { *result = func_name(x); }

__global__ void hipConj_kernel_v1(hipDoubleComplex* result, hipDoubleComplex* x) {
  *result = hipConj(x);
}
__global__ void hipConj_kernel_v2(hipDoubleComplex* result, hipFloatComplex x) {
  *result = hipConj(x);
}
__global__ void hipConj_kernel_v3(hipDoubleComplex* result, double x) { *result = hipConj(x); }
__global__ void hipConj_kernel_v4(hipDoubleComplex* result, Dummy x) { *result = hipConj(x); }
__global__ void hipConj_kernel_v5(double* result, hipDoubleComplex x) { *result = hipConj(x); }
__global__ void hipConj_kernel_v6(hipFloatComplex* result, hipDoubleComplex x) {
  *result = hipConj(x);
}
__global__ void hipConj_kernel_v7(Dummy* result, hipDoubleComplex x) { *result = hipConj(x); }
void hipConj_v1(hipDoubleComplex* result, hipDoubleComplex* x) { *result = hipConj(x); }
void hipConj_v2(hipDoubleComplex* result, hipFloatComplex x) { *result = hipConj(x); }
void hipConj_v3(hipDoubleComplex* result, double x) { *result = hipConj(x); }
void hipConj_v4(hipDoubleComplex* result, Dummy x) { *result = hipConj(x); }
void hipConj_v5(double* result, hipDoubleComplex x) { *result = hipConj(x); }
void hipConj_v6(hipFloatComplex* result, hipDoubleComplex x) { *result = hipConj(x); }
void hipConj_v7(Dummy* result, hipDoubleComplex x) { *result = hipConj(x); }

NEGATIVE_SHELL_ONE_ARG_DOUBLE(hipCreal)
NEGATIVE_SHELL_ONE_ARG_DOUBLE(hipCimag)
NEGATIVE_SHELL_ONE_ARG_DOUBLE(hipCabs)
NEGATIVE_SHELL_ONE_ARG_DOUBLE(hipCsqabs)

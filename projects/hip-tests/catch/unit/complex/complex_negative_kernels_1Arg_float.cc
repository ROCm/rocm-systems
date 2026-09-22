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

#define NEGATIVE_SHELL_ONE_ARG_FLOAT(func_name)                                                    \
  __global__ void func_name##_kernel_v1(float* result, hipFloatComplex* x) {                       \
    *result = func_name(x);                                                                        \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(float* result, hipDoubleComplex x) {                       \
    *result = func_name(x);                                                                        \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(float* result, float x) { *result = func_name(x); }        \
  __global__ void func_name##_kernel_v4(float* result, Dummy x) { *result = func_name(x); }        \
  __global__ void func_name##_kernel_v5(hipFloatComplex* result, hipFloatComplex x) {              \
    *result = func_name(x);                                                                        \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(Dummy* result, hipFloatComplex x) {                        \
    *result = func_name(x);                                                                        \
  }                                                                                                \
  void func_name##_v1(float* result, hipFloatComplex* x) { *result = func_name(x); }               \
  void func_name##_v2(float* result, hipDoubleComplex x) { *result = func_name(x); }               \
  void func_name##_v3(float* result, float x) { *result = func_name(x); }                          \
  void func_name##_v4(float* result, Dummy x) { *result = func_name(x); }                          \
  void func_name##_v5(hipFloatComplex* result, hipFloatComplex x) { *result = func_name(x); }      \
  void func_name##_v6(Dummy* result, hipFloatComplex x) { *result = func_name(x); }

__global__ void hipConjf_kernel_v1(hipFloatComplex* result, hipFloatComplex* x) {
  *result = hipConjf(x);
}
__global__ void hipConjf_kernel_v2(hipFloatComplex* result, hipDoubleComplex x) {
  *result = hipConjf(x);
}
__global__ void hipConjf_kernel_v3(hipFloatComplex* result, float x) { *result = hipConjf(x); }
__global__ void hipConjf_kernel_v4(hipFloatComplex* result, Dummy x) { *result = hipConjf(x); }
__global__ void hipConjf_kernel_v5(float* result, hipFloatComplex x) { *result = hipConjf(x); }
__global__ void hipConjf_kernel_v6(hipDoubleComplex* result, hipFloatComplex x) {
  *result = hipConjf(x);
}
__global__ void hipConjf_kernel_v7(Dummy* result, hipFloatComplex x) { *result = hipConjf(x); }
void hipConjf_v1(hipFloatComplex* result, hipFloatComplex* x) { *result = hipConjf(x); }
void hipConjf_v2(hipFloatComplex* result, hipDoubleComplex x) { *result = hipConjf(x); }
void hipConjf_v3(hipFloatComplex* result, float x) { *result = hipConjf(x); }
void hipConjf_v4(hipFloatComplex* result, Dummy x) { *result = hipConjf(x); }
void hipConjf_v5(float* result, hipFloatComplex x) { *result = hipConjf(x); }
void hipConjf_v6(hipDoubleComplex* result, hipFloatComplex x) { *result = hipConjf(x); }
void hipConjf_v7(Dummy* result, hipFloatComplex x) { *result = hipConjf(x); }

NEGATIVE_SHELL_ONE_ARG_FLOAT(hipCrealf)
NEGATIVE_SHELL_ONE_ARG_FLOAT(hipCimagf)
NEGATIVE_SHELL_ONE_ARG_FLOAT(hipCabsf)
NEGATIVE_SHELL_ONE_ARG_FLOAT(hipCsqabsf)

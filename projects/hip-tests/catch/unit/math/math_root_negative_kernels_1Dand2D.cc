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
  __global__ void func_name##_kernel_v1(double* x, double y) { double result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v2(double x, double* y) { double result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v3(Dummy x, double y) { double result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v4(double x, Dummy y) { double result = func_name(x, y); }    \
  __global__ void func_name##f_kernel_v1(float* x, float y) { float result = func_name##f(x, y); } \
  __global__ void func_name##f_kernel_v2(float x, float* y) { float result = func_name##f(x, y); } \
  __global__ void func_name##f_kernel_v3(Dummy x, float y) { float result = func_name##f(x, y); }  \
  __global__ void func_name##f_kernel_v4(float x, Dummy y) { float result = func_name##f(x, y); }

#define NEGATIVE_KERNELS_SHELL_ARRAY_ARG(func_name)                                                \
  __global__ void func_name##_kernel_v1(int* dim, const double* a) {                               \
    double result = func_name(dim, a);                                                             \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(Dummy dim, const double* a) {                              \
    double result = func_name(dim, a);                                                             \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(int dim, const int* a) {                                   \
    double result = func_name(dim, a);                                                             \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(int dim, const char* a) {                                  \
    double result = func_name(dim, a);                                                             \
  }                                                                                                \
  __global__ void func_name##_kernel_v5(int dim, const short* a) {                                 \
    double result = func_name(dim, a);                                                             \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(int dim, const long* a) {                                  \
    double result = func_name(dim, a);                                                             \
  }                                                                                                \
  __global__ void func_name##_kernel_v7(int dim, const long long* a) {                             \
    double result = func_name(dim, a);                                                             \
  }                                                                                                \
  __global__ void func_name##_kernel_v8(int dim, const float* a) {                                 \
    double result = func_name(dim, a);                                                             \
  }                                                                                                \
  __global__ void func_name##_kernel_v9(int dim, const Dummy* a) {                                 \
    double result = func_name(dim, a);                                                             \
  }                                                                                                \
  __global__ void func_name##f_kernel_v1(int* dim, const float* a) {                               \
    float result = func_name##f(dim, a);                                                           \
  }                                                                                                \
  __global__ void func_name##f_kernel_v2(Dummy dim, const float* a) {                              \
    float result = func_name##f(dim, a);                                                           \
  }                                                                                                \
  __global__ void func_name##f_kernel_v3(int dim, const int* a) {                                  \
    float result = func_name##f(dim, a);                                                           \
  }                                                                                                \
  __global__ void func_name##f_kernel_v4(int dim, const char* a) {                                 \
    float result = func_name##f(dim, a);                                                           \
  }                                                                                                \
  __global__ void func_name##f_kernel_v5(int dim, const short* a) {                                \
    float result = func_name##f(dim, a);                                                           \
  }                                                                                                \
  __global__ void func_name##f_kernel_v6(int dim, const long* a) {                                 \
    float result = func_name##f(dim, a);                                                           \
  }                                                                                                \
  __global__ void func_name##f_kernel_v7(int dim, const long long* a) {                            \
    float result = func_name##f(dim, a);                                                           \
  }                                                                                                \
  __global__ void func_name##f_kernel_v8(int dim, const double* a) {                               \
    float result = func_name##f(dim, a);                                                           \
  }                                                                                                \
  __global__ void func_name##f_kernel_v9(int dim, const Dummy* a) {                                \
    double result = func_name##f(dim, a);                                                          \
  }

NEGATIVE_KERNELS_SHELL_ONE_ARG(sqrt)
NEGATIVE_KERNELS_SHELL_ONE_ARG(rsqrt)
NEGATIVE_KERNELS_SHELL_ONE_ARG(cbrt)
NEGATIVE_KERNELS_SHELL_ONE_ARG(rcbrt)
NEGATIVE_KERNELS_SHELL_TWO_ARGS(hypot)
NEGATIVE_KERNELS_SHELL_TWO_ARGS(rhypot)
NEGATIVE_KERNELS_SHELL_ARRAY_ARG(norm)
NEGATIVE_KERNELS_SHELL_ARRAY_ARG(rnorm)

/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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

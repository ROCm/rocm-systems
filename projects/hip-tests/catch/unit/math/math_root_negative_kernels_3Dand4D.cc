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

#define NEGATIVE_KERNELS_SHELL_THREE_ARGS(func_name)                                               \
  __global__ void func_name##_kernel_v1(double* x, double y, double z) {                           \
    double result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(double x, double* y, double z) {                           \
    double result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(double x, double y, double* z) {                           \
    double result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(Dummy x, double y, double z) {                             \
    double result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v5(double x, Dummy y, double z) {                             \
    double result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(double x, double y, Dummy z) {                             \
    double result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##f_kernel_v1(float* x, float y, float z) {                             \
    float result = func_name##f(x, y, z);                                                          \
  }                                                                                                \
  __global__ void func_name##f_kernel_v2(float x, float* y, float z) {                             \
    float result = func_name##f(x, y, z);                                                          \
  }                                                                                                \
  __global__ void func_name##f_kernel_v3(float x, float y, float* z) {                             \
    float result = func_name##f(x, y, z);                                                          \
  }                                                                                                \
  __global__ void func_name##f_kernel_v4(Dummy x, float y, float z) {                              \
    float result = func_name##f(x, y, z);                                                          \
  }                                                                                                \
  __global__ void func_name##f_kernel_v5(float x, Dummy y, float z) {                              \
    float result = func_name##f(x, y, z);                                                          \
  }                                                                                                \
  __global__ void func_name##f_kernel_v6(float x, float y, Dummy z) {                              \
    float result = func_name##f(x, y, z);                                                          \
  }

#define NEGATIVE_KERNELS_SHELL_FOUR_ARGS(func_name)                                                \
  __global__ void func_name##_kernel_v1(double* x, double y, double z, double w) {                 \
    double result = func_name(x, y, z, w);                                                         \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(double x, double* y, double z, double w) {                 \
    double result = func_name(x, y, z, w);                                                         \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(double x, double y, double* z, double w) {                 \
    double result = func_name(x, y, z, w);                                                         \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(double x, double y, double z, double* w) {                 \
    double result = func_name(x, y, z, w);                                                         \
  }                                                                                                \
  __global__ void func_name##_kernel_v5(Dummy x, double y, double z, double w) {                   \
    double result = func_name(x, y, z, w);                                                         \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(double x, Dummy y, double z, double w) {                   \
    double result = func_name(x, y, z, w);                                                         \
  }                                                                                                \
  __global__ void func_name##_kernel_v7(double x, double y, Dummy z, double w) {                   \
    double result = func_name(x, y, z, w);                                                         \
  }                                                                                                \
  __global__ void func_name##_kernel_v8(double x, double y, double z, Dummy w) {                   \
    double result = func_name(x, y, z, w);                                                         \
  }                                                                                                \
  __global__ void func_name##f_kernel_v1(float* x, float y, float z, float w) {                    \
    float result = func_name##f(x, y, z, w);                                                       \
  }                                                                                                \
  __global__ void func_name##f_kernel_v2(float x, float* y, float z, float w) {                    \
    float result = func_name##f(x, y, z, w);                                                       \
  }                                                                                                \
  __global__ void func_name##f_kernel_v3(float x, float y, float* z, float w) {                    \
    float result = func_name##f(x, y, z, w);                                                       \
  }                                                                                                \
  __global__ void func_name##f_kernel_v4(float x, float y, float z, float* w) {                    \
    float result = func_name##f(x, y, z, w);                                                       \
  }                                                                                                \
  __global__ void func_name##f_kernel_v5(Dummy x, float y, float z, float w) {                     \
    float result = func_name##f(x, y, z, w);                                                       \
  }                                                                                                \
  __global__ void func_name##f_kernel_v6(float x, Dummy y, float z, float w) {                     \
    float result = func_name##f(x, y, z, w);                                                       \
  }                                                                                                \
  __global__ void func_name##f_kernel_v7(float x, float y, Dummy z, float w) {                     \
    float result = func_name##f(x, y, z, w);                                                       \
  }                                                                                                \
  __global__ void func_name##f_kernel_v8(float x, float y, float z, Dummy w) {                     \
    float result = func_name##f(x, y, z, w);                                                       \
  }

NEGATIVE_KERNELS_SHELL_THREE_ARGS(norm3d)
NEGATIVE_KERNELS_SHELL_THREE_ARGS(rnorm3d)
NEGATIVE_KERNELS_SHELL_FOUR_ARGS(norm4d)
NEGATIVE_KERNELS_SHELL_FOUR_ARGS(rnorm4d)

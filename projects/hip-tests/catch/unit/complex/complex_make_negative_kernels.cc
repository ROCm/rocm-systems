/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip/hip_complex.h>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_SHELL_MAKE_FLOAT(T, func_name)                                                    \
  __global__ void func_name##_kernel_v1(T* result, float* x, float y) {                            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(T* result, float x, float* y) {                            \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(T* result, T x, float y) { *result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v4(T* result, float x, T y) { *result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v5(T* result, Dummy x, float y) {                             \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(T* result, float x, Dummy y) {                             \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v7(float* result, float x, float y) {                         \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v8(hipDoubleComplex* result, float x, float y) {              \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  __global__ void func_name##_kernel_v9(Dummy* result, float x, float y) {                         \
    *result = func_name(x, y);                                                                     \
  }                                                                                                \
  void func_name##_v1(T* result, float* x, float y) { *result = func_name(x, y); }                 \
  void func_name##_v2(T* result, float x, float* y) { *result = func_name(x, y); }                 \
  void func_name##_v3(T* result, T x, float y) { *result = func_name(x, y); }                      \
  void func_name##_v4(T* result, float x, T y) { *result = func_name(x, y); }                      \
  void func_name##_v5(T* result, Dummy x, float y) { *result = func_name(x, y); }                  \
  void func_name##_v6(T* result, float x, Dummy y) { *result = func_name(x, y); }                  \
  void func_name##_v7(float* result, float x, float y) { *result = func_name(x, y); }              \
  void func_name##_v8(hipDoubleComplex* result, float x, float y) { *result = func_name(x, y); }   \
  void func_name##_v9(Dummy* result, float x, float y) { *result = func_name(x, y); }

__global__ void make_hipDoubleComplex_kernel_v1(hipDoubleComplex* result, double* x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
__global__ void make_hipDoubleComplex_kernel_v2(hipDoubleComplex* result, double x, double* y) {
  *result = make_hipDoubleComplex(x, y);
}
__global__ void make_hipDoubleComplex_kernel_v3(hipDoubleComplex* result, hipDoubleComplex x,
                                                double y) {
  *result = make_hipDoubleComplex(x, y);
}
__global__ void make_hipDoubleComplex_kernel_v4(hipDoubleComplex* result, double x,
                                                hipDoubleComplex y) {
  *result = make_hipDoubleComplex(x, y);
}
__global__ void make_hipDoubleComplex_kernel_v5(hipDoubleComplex* result, Dummy x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
__global__ void make_hipDoubleComplex_kernel_v6(hipDoubleComplex* result, double x, Dummy y) {
  *result = make_hipDoubleComplex(x, y);
}
__global__ void make_hipDoubleComplex_kernel_v7(double* result, double x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
__global__ void make_hipDoubleComplex_kernel_v8(hipFloatComplex* result, double x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
__global__ void make_hipDoubleComplex_kernel_v9(Dummy* result, double x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
void make_hipDoubleComplex_v1(hipDoubleComplex* result, double* x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
void make_hipDoubleComplex_v2(hipDoubleComplex* result, double x, double* y) {
  *result = make_hipDoubleComplex(x, y);
}
void make_hipDoubleComplex_v3(hipDoubleComplex* result, hipDoubleComplex x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
void make_hipDoubleComplex_v4(hipDoubleComplex* result, double x, hipDoubleComplex y) {
  *result = make_hipDoubleComplex(x, y);
}
void make_hipDoubleComplex_v5(hipDoubleComplex* result, Dummy x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
void make_hipDoubleComplex_v6(hipDoubleComplex* result, double x, Dummy y) {
  *result = make_hipDoubleComplex(x, y);
}
void make_hipDoubleComplex_v7(float* result, double x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
void make_hipDoubleComplex_v8(hipFloatComplex* result, double x, double y) {
  *result = make_hipDoubleComplex(x, y);
}
void make_hipDoubleComplex_v9(Dummy* result, double x, double y) {
  *result = make_hipDoubleComplex(x, y);
}

NEGATIVE_SHELL_MAKE_FLOAT(hipFloatComplex, make_hipFloatComplex)
NEGATIVE_SHELL_MAKE_FLOAT(hipComplex, make_hipComplex)
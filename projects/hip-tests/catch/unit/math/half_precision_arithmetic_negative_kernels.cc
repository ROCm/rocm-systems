/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/hip_fp16.h>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};


#define UNARY_HALF_NEGATIVE_KERNELS(func_name)                                                     \
  __global__ void func_name##_kernel_v1(__half* x) { __half result = func_name(x); }               \
  __global__ void func_name##_kernel_v2(Dummy x) { __half result = func_name(x); }

#define BINARY_HALF_NEGATIVE_KERNELS(func_name)                                                    \
  __global__ void func_name##_kernel_v1(__half* x, __half y) { __half result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v2(__half x, __half* y) { __half result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v3(Dummy x, __half y) { __half result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v4(__half x, Dummy y) { __half result = func_name(x, y); }

#define TERNARY_HALF_NEGATIVE_KERNELS(func_name)                                                   \
  __global__ void func_name##_kernel_v1(__half* x, __half y, __half z) {                           \
    __half result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(__half x, __half* y, __half z) {                           \
    __half result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(__half x, __half y, __half* z) {                           \
    __half result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(Dummy x, __half y, __half z) {                             \
    __half result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v5(__half x, Dummy y, __half z) {                             \
    __half result = func_name(x, y, z);                                                            \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(__half x, __half y, Dummy z) {                             \
    __half result = func_name(x, y, z);                                                            \
  }

UNARY_HALF_NEGATIVE_KERNELS(__habs)
UNARY_HALF_NEGATIVE_KERNELS(__hneg)

BINARY_HALF_NEGATIVE_KERNELS(__hadd)
BINARY_HALF_NEGATIVE_KERNELS(__hadd_sat)
BINARY_HALF_NEGATIVE_KERNELS(__hsub)
BINARY_HALF_NEGATIVE_KERNELS(__hsub_sat)
BINARY_HALF_NEGATIVE_KERNELS(__hmul)
BINARY_HALF_NEGATIVE_KERNELS(__hmul_sat)
BINARY_HALF_NEGATIVE_KERNELS(__hdiv)

TERNARY_HALF_NEGATIVE_KERNELS(__hfma)
TERNARY_HALF_NEGATIVE_KERNELS(__hfma_sat)


#define UNARY_HALF2_NEGATIVE_KERNELS(func_name)                                                    \
  __global__ void func_name##_kernel_v1(__half2* x) { __half2 result = func_name(x); }             \
  __global__ void func_name##_kernel_v2(Dummy x) { __half2 result = func_name(x); }

#define BINARY_HALF2_NEGATIVE_KERNELS(func_name)                                                   \
  __global__ void func_name##_kernel_v1(__half2* x, __half2 y) {                                   \
    __half2 result = func_name(x, y);                                                              \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(__half2 x, __half2* y) {                                   \
    __half2 result = func_name(x, y);                                                              \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(Dummy x, __half2 y) { __half2 result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v4(__half2 x, Dummy y) { __half2 result = func_name(x, y); }

#define TERNARY_HALF2_NEGATIVE_KERNELS(func_name)                                                  \
  __global__ void func_name##_kernel_v1(__half2* x, __half2 y, __half2 z) {                        \
    __half2 result = func_name(x, y, z);                                                           \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(__half2 x, __half2* y, __half2 z) {                        \
    __half2 result = func_name(x, y, z);                                                           \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(__half2 x, __half2 y, __half2* z) {                        \
    __half2 result = func_name(x, y, z);                                                           \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(Dummy x, __half2 y, __half2 z) {                           \
    __half2 result = func_name(x, y, z);                                                           \
  }                                                                                                \
  __global__ void func_name##_kernel_v5(__half2 x, Dummy y, __half2 z) {                           \
    __half2 result = func_name(x, y, z);                                                           \
  }                                                                                                \
  __global__ void func_name##_kernel_v6(__half2 x, __half2 y, Dummy z) {                           \
    __half2 result = func_name(x, y, z);                                                           \
  }

UNARY_HALF2_NEGATIVE_KERNELS(__habs2)
UNARY_HALF2_NEGATIVE_KERNELS(__hneg2)

BINARY_HALF2_NEGATIVE_KERNELS(__hadd2)
BINARY_HALF2_NEGATIVE_KERNELS(__hadd2_sat)
BINARY_HALF2_NEGATIVE_KERNELS(__hsub2)
BINARY_HALF2_NEGATIVE_KERNELS(__hsub2_sat)
BINARY_HALF2_NEGATIVE_KERNELS(__hmul2)
BINARY_HALF2_NEGATIVE_KERNELS(__hmul2_sat)
BINARY_HALF2_NEGATIVE_KERNELS(__h2div)

TERNARY_HALF2_NEGATIVE_KERNELS(__hfma2)
TERNARY_HALF2_NEGATIVE_KERNELS(__hfma2_sat)
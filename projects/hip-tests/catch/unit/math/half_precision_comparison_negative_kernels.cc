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


#define UNARY_BOOL_HALF_NEGATIVE_KERNELS(func_name)                                                \
  __global__ void func_name##_kernel_v1(__half* x) { bool result = func_name(x); }                 \
  __global__ void func_name##_kernel_v2(Dummy x) { bool result = func_name(x); }

#define BINARY_BOOL_HALF_NEGATIVE_KERNELS(func_name)                                               \
  __global__ void func_name##_kernel_v1(__half* x, __half y) { bool result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v2(__half x, __half* y) { bool result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v3(Dummy x, __half y) { bool result = func_name(x, y); }      \
  __global__ void func_name##_kernel_v4(__half x, Dummy y) { bool result = func_name(x, y); }


#define BINARY_HALF_NEGATIVE_KERNELS(func_name)                                                    \
  __global__ void func_name##_kernel_v1(__half* x, __half y) { __half result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v2(__half x, __half* y) { __half result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v3(Dummy x, __half y) { __half result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v4(__half x, Dummy y) { __half result = func_name(x, y); }


UNARY_BOOL_HALF_NEGATIVE_KERNELS(__hisinf)
UNARY_BOOL_HALF_NEGATIVE_KERNELS(__hisnan)

BINARY_BOOL_HALF_NEGATIVE_KERNELS(__heq)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hequ)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hne)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hneu)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hge)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hgeu)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hgt)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hgtu)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hle)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hleu)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hlt)
BINARY_BOOL_HALF_NEGATIVE_KERNELS(__hltu)

BINARY_HALF_NEGATIVE_KERNELS(__hmax)
BINARY_HALF_NEGATIVE_KERNELS(__hmax_nan)
BINARY_HALF_NEGATIVE_KERNELS(__hmin)
BINARY_HALF_NEGATIVE_KERNELS(__hmin_nan)


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

#define BINARY_BOOL_HALF2_NEGATIVE_KERNELS(func_name)                                              \
  __global__ void func_name##_kernel_v1(__half2* x, __half2 y) { bool result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v2(__half2 x, __half2* y) { bool result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v3(Dummy x, __half2 y) { bool result = func_name(x, y); }     \
  __global__ void func_name##_kernel_v4(__half2 x, Dummy y) { bool result = func_name(x, y); }

UNARY_HALF2_NEGATIVE_KERNELS(__hisinf2)
UNARY_HALF2_NEGATIVE_KERNELS(__hisnan2)

BINARY_HALF2_NEGATIVE_KERNELS(__heq2)
BINARY_HALF2_NEGATIVE_KERNELS(__hequ2)
BINARY_HALF2_NEGATIVE_KERNELS(__hne2)
BINARY_HALF2_NEGATIVE_KERNELS(__hneu2)
BINARY_HALF2_NEGATIVE_KERNELS(__hge2)
BINARY_HALF2_NEGATIVE_KERNELS(__hgeu2)
BINARY_HALF2_NEGATIVE_KERNELS(__hgt2)
BINARY_HALF2_NEGATIVE_KERNELS(__hgtu2)
BINARY_HALF2_NEGATIVE_KERNELS(__hle2)
BINARY_HALF2_NEGATIVE_KERNELS(__hleu2)
BINARY_HALF2_NEGATIVE_KERNELS(__hlt2)
BINARY_HALF2_NEGATIVE_KERNELS(__hltu2)

BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbeq2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbequ2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbne2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbneu2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbge2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbgeu2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbgt2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbgtu2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hble2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbleu2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hblt2)
BINARY_BOOL_HALF2_NEGATIVE_KERNELS(__hbltu2)
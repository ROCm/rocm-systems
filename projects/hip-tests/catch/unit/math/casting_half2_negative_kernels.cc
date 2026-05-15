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

#define NEGATIVE_UNARY_KERNELS_SHELL(func_name, T1, T2)                                            \
  __global__ void func_name##_kernel_v1(T1* result, T2* x) { *result = func_name(x); }             \
  __global__ void func_name##_kernel_v2(T1* result, Dummy x) { *result = func_name(x); }           \
  __global__ void func_name##_kernel_v3(Dummy* result, T2 x) { *result = func_name(x); }


#define NEGATIVE_BINARY_KERNELS_SHELL(func_name, T1, T2)                                           \
  __global__ void func_name##_kernel_v1(T2* x, T2 y) { T1 result = func_name(x, y); }              \
  __global__ void func_name##_kernel_v2(T2 x, T2* y) { T1 result = func_name(x, y); }              \
  __global__ void func_name##_kernel_v3(Dummy x, T2 y) { T1 result = func_name(x, y); }            \
  __global__ void func_name##_kernel_v4(T2 x, Dummy y) { T1 result = func_name(x, y); }

NEGATIVE_UNARY_KERNELS_SHELL(__half2half2, __half2, __half)
NEGATIVE_UNARY_KERNELS_SHELL(__low2half, __half, __half2)
NEGATIVE_UNARY_KERNELS_SHELL(__high2half, __half, __half2)
NEGATIVE_UNARY_KERNELS_SHELL(__low2half2, __half2, __half2)
NEGATIVE_UNARY_KERNELS_SHELL(__high2half2, __half2, __half2)
NEGATIVE_UNARY_KERNELS_SHELL(__lowhigh2highlow, __half2, __half2)
NEGATIVE_UNARY_KERNELS_SHELL(__float2half2_rn, __half2, float)
NEGATIVE_UNARY_KERNELS_SHELL(__float22half2_rn, __half2, float2)
NEGATIVE_UNARY_KERNELS_SHELL(__low2float, float, __half2)
NEGATIVE_UNARY_KERNELS_SHELL(__high2float, float, __half2)
NEGATIVE_UNARY_KERNELS_SHELL(__half22float2, float2, __half2)

NEGATIVE_BINARY_KERNELS_SHELL(make_half2, __half2, __half)
NEGATIVE_BINARY_KERNELS_SHELL(__halves2half2, __half2, __half)
NEGATIVE_BINARY_KERNELS_SHELL(__lows2half2, __half2, __half2)
NEGATIVE_BINARY_KERNELS_SHELL(__highs2half2, __half2, __half2)
NEGATIVE_BINARY_KERNELS_SHELL(__floats2half2_rn, __half2, float)
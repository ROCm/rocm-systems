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

#define NEGATIVE_F2H_KERNELS_SHELL(func_name)                                                      \
  __global__ void func_name##_kernel_v1(__half* result, float* x) { *result = func_name(x); }      \
  __global__ void func_name##_kernel_v2(__half* result, Dummy x) { *result = func_name(x); }       \
  __global__ void func_name##_kernel_v3(Dummy* result, float x) { *result = func_name(x); }

#define NEGATIVE_H2F_KERNELS_SHELL(func_name)                                                      \
  __global__ void func_name##_kernel_v1(float* result, __half* x) { *result = func_name(x); }      \
  __global__ void func_name##_kernel_v2(float* result, Dummy x) { *result = func_name(x); }        \
  __global__ void func_name##_kernel_v3(Dummy* result, __half x) { *result = func_name(x); }

NEGATIVE_F2H_KERNELS_SHELL(__float2half_rd)
NEGATIVE_F2H_KERNELS_SHELL(__float2half_rn)
NEGATIVE_F2H_KERNELS_SHELL(__float2half_ru)
NEGATIVE_F2H_KERNELS_SHELL(__float2half_rz)
NEGATIVE_F2H_KERNELS_SHELL(__float2half)

NEGATIVE_H2F_KERNELS_SHELL(__half2float)
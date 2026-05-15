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

#define INTRINSIC_UNARY_DOUBLE_NEGATIVE_KERNELS(func_name)                                         \
  __global__ void func_name##_kernel_v1(double* x) { double result = func_name(x); }               \
  __global__ void func_name##_kernel_v2(Dummy x) { double result = func_name(x); }

#define INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(func_name)                                        \
  __global__ void func_name##_kernel_v1(double* x, double y) { double result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v2(double x, double* y) { double result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v3(Dummy x, double y) { double result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v4(double x, Dummy y) { double result = func_name(x, y); }


INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(__dadd_rn)
INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(__dsub_rn)
INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(__dmul_rn)
INTRINSIC_BINARY_DOUBLE_NEGATIVE_KERNELS(__ddiv_rn)
INTRINSIC_UNARY_DOUBLE_NEGATIVE_KERNELS(__dsqrt_rn)
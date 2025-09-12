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

#define INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(func_name)                                          \
  __global__ void func_name##_kernel_v1(float* x) { float result = func_name(x); }                 \
  __global__ void func_name##_kernel_v2(Dummy x) { float result = func_name(x); }

#define INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(func_name)                                         \
  __global__ void func_name##_kernel_v1(float* x, float y) { float result = func_name(x, y); }     \
  __global__ void func_name##_kernel_v2(float x, float* y) { float result = func_name(x, y); }     \
  __global__ void func_name##_kernel_v3(Dummy x, float y) { float result = func_name(x, y); }      \
  __global__ void func_name##_kernel_v4(float x, Dummy y) { float result = func_name(x, y); }

INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__fsqrt_rn)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__expf)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__exp10f)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__logf)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__log2f)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__log10f)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__sinf)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__cosf)
INTRINSIC_UNARY_FLOAT_NEGATIVE_KERNELS(__tanf)

INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fadd_rn)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fsub_rn)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fmul_rn)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fdiv_rn)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__fdividef)
INTRINSIC_BINARY_FLOAT_NEGATIVE_KERNELS(__powf)
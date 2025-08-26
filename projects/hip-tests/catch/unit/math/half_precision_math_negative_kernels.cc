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

UNARY_HALF_NEGATIVE_KERNELS(hcos)
UNARY_HALF_NEGATIVE_KERNELS(hsin)
UNARY_HALF_NEGATIVE_KERNELS(hexp)
UNARY_HALF_NEGATIVE_KERNELS(hexp10)
UNARY_HALF_NEGATIVE_KERNELS(hexp2)
UNARY_HALF_NEGATIVE_KERNELS(hlog)
UNARY_HALF_NEGATIVE_KERNELS(hlog10)
UNARY_HALF_NEGATIVE_KERNELS(hlog2)
UNARY_HALF_NEGATIVE_KERNELS(hsqrt)
UNARY_HALF_NEGATIVE_KERNELS(hceil)
UNARY_HALF_NEGATIVE_KERNELS(hfloor)
UNARY_HALF_NEGATIVE_KERNELS(htrunc)
UNARY_HALF_NEGATIVE_KERNELS(hrcp)
UNARY_HALF_NEGATIVE_KERNELS(hrsqrt)
UNARY_HALF_NEGATIVE_KERNELS(hrint)


#define UNARY_HALF2_NEGATIVE_KERNELS(func_name)                                                    \
  __global__ void func_name##_kernel_v1(__half2* x) { __half2 result = func_name(x); }             \
  __global__ void func_name##_kernel_v2(Dummy x) { __half2 result = func_name(x); }

UNARY_HALF2_NEGATIVE_KERNELS(h2cos)
UNARY_HALF2_NEGATIVE_KERNELS(h2sin)
UNARY_HALF2_NEGATIVE_KERNELS(h2exp)
UNARY_HALF2_NEGATIVE_KERNELS(h2exp10)
UNARY_HALF2_NEGATIVE_KERNELS(h2exp2)
UNARY_HALF2_NEGATIVE_KERNELS(h2log)
UNARY_HALF2_NEGATIVE_KERNELS(h2log10)
UNARY_HALF2_NEGATIVE_KERNELS(h2log2)
UNARY_HALF2_NEGATIVE_KERNELS(h2sqrt)
UNARY_HALF2_NEGATIVE_KERNELS(h2ceil)
UNARY_HALF2_NEGATIVE_KERNELS(h2floor)
UNARY_HALF2_NEGATIVE_KERNELS(h2trunc)
UNARY_HALF2_NEGATIVE_KERNELS(h2rcp)
UNARY_HALF2_NEGATIVE_KERNELS(h2rsqrt)
UNARY_HALF2_NEGATIVE_KERNELS(h2rint)

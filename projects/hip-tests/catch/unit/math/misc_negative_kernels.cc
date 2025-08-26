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

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define MISC_UNARY_NEGATIVE_KERNELS(func_name)                                                     \
  __global__ void func_name##f_kernel_v1(float* x) { float result = func_name##f(x); }             \
  __global__ void func_name##f_kernel_v2(Dummy x) { float result = func_name##f(x); }              \
  __global__ void func_name##_kernel_v1(double* x) { double result = func_name(x); }               \
  __global__ void func_name##_kernel_v2(Dummy x) { double result = func_name(x); }

#define MISC_UNARY_BOOL_RET_NEGATIVE_KERNELS(func_name)                                            \
  __global__ void func_name##_kernel_v1(float* x) { bool result = func_name(x); }                  \
  __global__ void func_name##_kernel_v2(Dummy x) { bool result = func_name(x); }                   \
  __global__ void func_name##_kernel_v3(double* x) { bool result = func_name(x); }                 \
  __global__ void func_name##_kernel_v4(Dummy x) { bool result = func_name(x); }

#define MISC_BINARY_NEGATIVE_KERNELS(func_name)                                                    \
  __global__ void func_name##f_kernel_v1(float* x, float y) { float result = func_name##f(x, y); } \
  __global__ void func_name##f_kernel_v2(Dummy x, float y) { float result = func_name##f(x, y); }  \
  __global__ void func_name##f_kernel_v3(float x, float* y) { float result = func_name##f(x, y); } \
  __global__ void func_name##f_kernel_v4(float x, Dummy y) { float result = func_name##f(x, y); }  \
  __global__ void func_name##_kernel_v1(double* x, double y) { double result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v2(Dummy x, double y) { double result = func_name(x, y); }    \
  __global__ void func_name##_kernel_v3(double x, double* y) { double result = func_name(x, y); }  \
  __global__ void func_name##_kernel_v4(double x, Dummy y) { double result = func_name(x, y); }

/*Expecting 4 errors*/
MISC_UNARY_NEGATIVE_KERNELS(fabs)

/*Expecting 8 errors per macro invocation - 40 total*/
MISC_BINARY_NEGATIVE_KERNELS(copysign)
MISC_BINARY_NEGATIVE_KERNELS(fmax)
MISC_BINARY_NEGATIVE_KERNELS(fmin)
MISC_BINARY_NEGATIVE_KERNELS(nextafter)
MISC_BINARY_NEGATIVE_KERNELS(fma)

/*Expecting 4 errors*/
__global__ void fdividef_kernel_v1(float* x, float y) { float result = fdividef(x, y); }
__global__ void fdividef_kernel_v2(Dummy x, float y) { float result = fdivide(x); }
__global__ void fdividef_kernel_v3(float x, float* y) { float result = fdivide(x); }
__global__ void fdividef_kernel_v4(float x, Dummy y) { float result = fdivide(x); }

/*Expecting 4 errors per macro invocation - 16 total*/
MISC_UNARY_BOOL_RET_NEGATIVE_KERNELS(isfinite)
MISC_UNARY_BOOL_RET_NEGATIVE_KERNELS(isinf)
MISC_UNARY_BOOL_RET_NEGATIVE_KERNELS(isnan)
MISC_UNARY_BOOL_RET_NEGATIVE_KERNELS(signbit)

/*Expecting 12 errors*/
__global__ void fmaf_kernel_v1(float* x, float y, float z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v2(Dummy x, float y, float z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v3(float x, float* y, float z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v4(float x, Dummy y, float z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v5(float x, float y, float* z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v6(float x, float y, Dummy z) { float result = fmaf(x, y, z); }
__global__ void fma_kernel_v1(double* x, double y, double z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v2(Dummy x, double y, double z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v3(double x, double* y, double z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v4(double x, Dummy y, double z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v5(double x, double y, double* z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v6(double x, double y, Dummy z) { double result = fmaf(x, y, z); }
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

#define NEGATIVE_KERNELS_SHELL_EXP(func_name)                                                      \
  __global__ void func_name##_kernel_v1(double* x) { double result = func_name(x); }               \
  __global__ void func_name##_kernel_v2(Dummy x) { double result = func_name(x); }                 \
  __global__ void func_name##f_kernel_v1(float* x) { float result = func_name##f(x); }             \
  __global__ void func_name##f_kernel_v2(Dummy x) { float result = func_name##f(x); }

#define NEGATIVE_KERNELS_SHELL_INT_2ND(func_name)                                                  \
  __global__ void func_name##_kernel_v1(double* x, int e) { double result = func_name(x, e); }     \
  __global__ void func_name##_kernel_v2(Dummy x, int e) { double result = func_name(x, e); }       \
  __global__ void func_name##_kernel_v3(double x, int* e) { double result = func_name(x, e); }     \
  __global__ void func_name##_kernel_v4(double x, Dummy e) { double result = func_name(x, e); }    \
  __global__ void func_name##f_kernel_v1(float* x, int e) { float result = func_name##f(x, e); }   \
  __global__ void func_name##f_kernel_v2(Dummy x, int e) { float result = func_name##f(x, e); }    \
  __global__ void func_name##f_kernel_v3(float x, int* e) { float result = func_name##f(x, e); }   \
  __global__ void func_name##f_kernel_v4(float x, Dummy e) { float result = func_name##f(x, e); }


NEGATIVE_KERNELS_SHELL_EXP(exp)
NEGATIVE_KERNELS_SHELL_EXP(exp2)
NEGATIVE_KERNELS_SHELL_EXP(exp10)
NEGATIVE_KERNELS_SHELL_EXP(expm1)

__global__ void frexp_kernel_v1(double* x, int* nptr) { double result = frexp(x, nptr); }
__global__ void frexp_kernel_v2(Dummy x, int* nptr) { double result = frexp(x, nptr); }
__global__ void frexp_kernel_v3(double x, char* nptr) { double result = frexp(x, nptr); }
__global__ void frexp_kernel_v4(double x, short* nptr) { double result = frexp(x, nptr); }
__global__ void frexp_kernel_v5(double x, long* nptr) { double result = frexp(x, nptr); }
__global__ void frexp_kernel_v6(double x, long long* nptr) { double result = frexp(x, nptr); }
__global__ void frexp_kernel_v7(double x, float* nptr) { double result = frexp(x, nptr); }
__global__ void frexp_kernel_v8(double x, double* nptr) { double result = frexp(x, nptr); }
__global__ void frexp_kernel_v9(double x, Dummy* nptr) { double result = frexp(x, nptr); }
__global__ void frexp_kernel_v10(double x, const int* nptr) { double result = frexp(x, nptr); }
__global__ void frexpf_kernel_v1(float* x, int* nptr) { float result = frexpf(x, nptr); }
__global__ void frexpf_kernel_v2(Dummy x, int* nptr) { float result = frexpf(x, nptr); }
__global__ void frexpf_kernel_v3(float x, char* nptr) { float result = frexpf(x, nptr); }
__global__ void frexpf_kernel_v4(float x, short* nptr) { float result = frexpf(x, nptr); }
__global__ void frexpf_kernel_v5(float x, long* nptr) { float result = frexpf(x, nptr); }
__global__ void frexpf_kernel_v6(float x, long long* nptr) { float result = frexpf(x, nptr); }
__global__ void frexpf_kernel_v7(float x, float* nptr) { float result = frexpf(x, nptr); }
__global__ void frexpf_kernel_v8(float x, double* nptr) { float result = frexpf(x, nptr); }
__global__ void frexpf_kernel_v9(float x, Dummy* nptr) { float result = frexpf(x, nptr); }
__global__ void frexpf_kernel_v10(float x, const int* nptr) { float result = frexpf(x, nptr); }

NEGATIVE_KERNELS_SHELL_INT_2ND(ldexp)

__global__ void pow_kernel_v1(double* x, double e) { double result = pow(x, e); }
__global__ void pow_kernel_v2(Dummy x, double e) { double result = pow(x, e); }
__global__ void pow_kernel_v3(double x, double* e) { double result = pow(x, e); }
__global__ void pow_kernel_v4(double x, Dummy e) { double result = pow(x, e); }
__global__ void powf_kernel_v1(float* x, float e) { float result = powf(x, e); }
__global__ void powf_kernel_v2(Dummy x, float e) { float result = powf(x, e); }
__global__ void powf_kernel_v3(float x, float* e) { float result = powf(x, e); }
__global__ void powf_kernel_v4(float x, Dummy e) { float result = powf(x, e); }

NEGATIVE_KERNELS_SHELL_INT_2ND(powi)
NEGATIVE_KERNELS_SHELL_INT_2ND(scalbn)

__global__ void scalbln_kernel_v1(double* x, long int n) { double result = scalbln(x, n); }
__global__ void scalbln_kernel_v2(Dummy x, long int n) { double result = scalbln(x, n); }
__global__ void scalbln_kernel_v3(double x, long int* n) { double result = scalbln(x, n); }
__global__ void scalbln_kernel_v4(double x, Dummy n) { double result = scalbln(x, n); }
__global__ void scalblnf_kernel_v1(float* x, long int n) { float result = scalblnf(x, n); }
__global__ void scalblnf_kernel_v2(Dummy x, long int n) { float result = scalblnf(x, n); }
__global__ void scalblnf_kernel_v3(float x, long int* n) { float result = scalblnf(x, n); }
__global__ void scalblnf_kernel_v4(float x, Dummy n) { float result = scalblnf(x, n); }

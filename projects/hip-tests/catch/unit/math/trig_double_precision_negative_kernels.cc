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

#define TRIG_DP_UNARY_NEGATIVE_KERNELS(func_name)                                                  \
  __global__ void func_name##_kernel_v1(double* x) { double result = func_name(x); }               \
  __global__ void func_name##_kernel_v2(Dummy x) { double result = func_name(x); }

/*Expecting 2 errors per macro invocation - 26 total*/
TRIG_DP_UNARY_NEGATIVE_KERNELS(sin)
TRIG_DP_UNARY_NEGATIVE_KERNELS(cos)
TRIG_DP_UNARY_NEGATIVE_KERNELS(tan)
TRIG_DP_UNARY_NEGATIVE_KERNELS(asin)
TRIG_DP_UNARY_NEGATIVE_KERNELS(acos)
TRIG_DP_UNARY_NEGATIVE_KERNELS(atan)
TRIG_DP_UNARY_NEGATIVE_KERNELS(sinh)
TRIG_DP_UNARY_NEGATIVE_KERNELS(cosh)
TRIG_DP_UNARY_NEGATIVE_KERNELS(tanh)
TRIG_DP_UNARY_NEGATIVE_KERNELS(asinh)
TRIG_DP_UNARY_NEGATIVE_KERNELS(atanh)
TRIG_DP_UNARY_NEGATIVE_KERNELS(sinpi)
TRIG_DP_UNARY_NEGATIVE_KERNELS(cospi)

/*Expecting 4 errors*/
__global__ void atan2_kernel_v1(double* x, double y) { double result = atan2(x, y); }
__global__ void atan2_kernel_v2(double x, double* y) { double result = atan2(x, y); }
__global__ void atan2_kernel_v3(Dummy x, double y) { double result = atan2(x, y); }
__global__ void atan2_kernel_v4(double x, Dummy y) { double result = atan2(x, y); }

/*Expecting 18 errors*/
__global__ void sincos_kernel_v1(double* x, double* sptr, double* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v2(Dummy x, double* sptr, double* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v3(double x, char* sptr, double* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v4(double x, short* sptr, double* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v5(double x, int* sptr, double* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v6(double x, long* sptr, double* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v7(double x, long long* sptr, double* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v8(double x, float* sptr, double* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v9(double x, Dummy* sptr, double* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v10(double x, const double* sptr, double* cptr) {
  sincos(x, sptr, cptr);
}
__global__ void sincos_kernel_v11(double x, double* sptr, char* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v12(double x, double* sptr, short* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v13(double x, double* sptr, int* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v14(double x, double* sptr, long* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v15(double x, double* sptr, long long* cptr) {
  sincos(x, sptr, cptr);
}
__global__ void sincos_kernel_v16(double x, double* sptr, float* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v17(double x, double* sptr, Dummy* cptr) { sincos(x, sptr, cptr); }
__global__ void sincos_kernel_v18(double x, double* sptr, const double* cptr) {
  sincos(x, sptr, cptr);
}

/*Expecting 18 errors*/
__global__ void sincospi_kernel_v1(float* x, float* sptr, float* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v2(Dummy x, float* sptr, float* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v3(float x, char* sptr, float* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v4(float x, short* sptr, float* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v5(float x, int* sptr, float* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v6(float x, long* sptr, float* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v7(float x, long long* sptr, float* cptr) {
  sincospi(x, sptr, cptr);
}
__global__ void sincospi_kernel_v8(float x, double* sptr, float* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v9(float x, Dummy* sptr, float* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v10(float x, const float* sptr, float* cptr) {
  sincospi(x, sptr, cptr);
}
__global__ void sincospi_kernel_v11(float x, float* sptr, char* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v12(float x, float* sptr, short* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v13(float x, float* sptr, int* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v14(float x, float* sptr, long* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v15(float x, float* sptr, long long* cptr) {
  sincospi(x, sptr, cptr);
}
__global__ void sincospi_kernel_v16(float x, float* sptr, double* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v17(float x, float* sptr, Dummy* cptr) { sincospi(x, sptr, cptr); }
__global__ void sincospi_kernel_v18(float x, float* sptr, const float* cptr) {
  sincospi(x, sptr, cptr);
}
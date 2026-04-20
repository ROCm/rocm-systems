/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define TRIG_SP_UNARY_NEGATIVE_KERNELS(func_name)                                                  \
  __global__ void func_name##f_kernel_v1(float* x) { float result = func_name##f(x); }             \
  __global__ void func_name##f_kernel_v2(Dummy x) { float result = func_name##f(x); }

/*Expecting 2 errors per macro invocation - 26 total*/
TRIG_SP_UNARY_NEGATIVE_KERNELS(sin)
TRIG_SP_UNARY_NEGATIVE_KERNELS(cos)
TRIG_SP_UNARY_NEGATIVE_KERNELS(tan)
TRIG_SP_UNARY_NEGATIVE_KERNELS(asin)
TRIG_SP_UNARY_NEGATIVE_KERNELS(acos)
TRIG_SP_UNARY_NEGATIVE_KERNELS(atan)
TRIG_SP_UNARY_NEGATIVE_KERNELS(sinh)
TRIG_SP_UNARY_NEGATIVE_KERNELS(cosh)
TRIG_SP_UNARY_NEGATIVE_KERNELS(tanh)
TRIG_SP_UNARY_NEGATIVE_KERNELS(asinh)
TRIG_SP_UNARY_NEGATIVE_KERNELS(atanh)
TRIG_SP_UNARY_NEGATIVE_KERNELS(sinpi)
TRIG_SP_UNARY_NEGATIVE_KERNELS(cospi)

/*Expecting 4 errors*/
__global__ void atan2f_kernel_v1(float* x, float y) { float result = atan2f(x, y); }
__global__ void atan2f_kernel_v2(float x, float* y) { float result = atan2f(x, y); }
__global__ void atan2f_kernel_v3(Dummy x, float y) { float result = atan2f(x, y); }
__global__ void atan2f_kernel_v4(float x, Dummy y) { float result = atan2f(x, y); }

/*Expecting 18 errors*/
__global__ void sincosf_kernel_v1(float* x, float* sptr, float* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v2(Dummy x, float* sptr, float* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v3(float x, char* sptr, float* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v4(float x, short* sptr, float* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v5(float x, int* sptr, float* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v6(float x, long* sptr, float* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v7(float x, long long* sptr, float* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v8(float x, double* sptr, float* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v9(float x, Dummy* sptr, float* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v10(float x, const float* sptr, float* cptr) {
  sincosf(x, sptr, cptr);
}
__global__ void sincosf_kernel_v11(float x, float* sptr, char* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v12(float x, float* sptr, short* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v13(float x, float* sptr, int* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v14(float x, float* sptr, long* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v15(float x, float* sptr, long long* cptr) {
  sincosf(x, sptr, cptr);
}
__global__ void sincosf_kernel_v16(float x, float* sptr, double* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v17(float x, float* sptr, Dummy* cptr) { sincosf(x, sptr, cptr); }
__global__ void sincosf_kernel_v18(float x, float* sptr, const float* cptr) {
  sincosf(x, sptr, cptr);
}

/*Expecting 18 errors*/
__global__ void sincospif_kernel_v1(float* x, float* sptr, float* cptr) {
  sincospif(x, sptr, cptr);
}
__global__ void sincospif_kernel_v2(Dummy x, float* sptr, float* cptr) { sincospif(x, sptr, cptr); }
__global__ void sincospif_kernel_v3(float x, char* sptr, float* cptr) { sincospif(x, sptr, cptr); }
__global__ void sincospif_kernel_v4(float x, short* sptr, float* cptr) { sincospif(x, sptr, cptr); }
__global__ void sincospif_kernel_v5(float x, int* sptr, float* cptr) { sincospif(x, sptr, cptr); }
__global__ void sincospif_kernel_v6(float x, long* sptr, float* cptr) { sincospif(x, sptr, cptr); }
__global__ void sincospif_kernel_v7(float x, long long* sptr, float* cptr) {
  sincospif(x, sptr, cptr);
}
__global__ void sincospif_kernel_v8(float x, double* sptr, float* cptr) {
  sincospif(x, sptr, cptr);
}
__global__ void sincospif_kernel_v9(float x, Dummy* sptr, float* cptr) { sincospif(x, sptr, cptr); }
__global__ void sincospif_kernel_v10(float x, const float* sptr, float* cptr) {
  sincospif(x, sptr, cptr);
}
__global__ void sincospif_kernel_v11(float x, float* sptr, char* cptr) { sincospif(x, sptr, cptr); }
__global__ void sincospif_kernel_v12(float x, float* sptr, short* cptr) {
  sincospif(x, sptr, cptr);
}
__global__ void sincospif_kernel_v13(float x, float* sptr, int* cptr) { sincospif(x, sptr, cptr); }
__global__ void sincospif_kernel_v14(float x, float* sptr, long* cptr) { sincospif(x, sptr, cptr); }
__global__ void sincospif_kernel_v15(float x, float* sptr, long long* cptr) {
  sincospif(x, sptr, cptr);
}
__global__ void sincospif_kernel_v16(float x, float* sptr, double* cptr) {
  sincospif(x, sptr, cptr);
}
__global__ void sincospif_kernel_v17(float x, float* sptr, Dummy* cptr) {
  sincospif(x, sptr, cptr);
}
__global__ void sincospif_kernel_v18(float x, float* sptr, const float* cptr) {
  sincospif(x, sptr, cptr);
}
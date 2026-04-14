/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip/device_functions.h>

#pragma GCC diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wunused-variable"

__device__ void single_precision_intrinsics() {
  float fX, fY;

  __cosf(0.0f);
  __exp10f(0.0f);
  __expf(0.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __fadd_rd(0.0f, 1.0f);
#endif
  __fadd_rn(0.0f, 1.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __fadd_ru(0.0f, 1.0f);
  __fadd_rz(0.0f, 1.0f);
  __fdiv_rd(4.0f, 2.0f);
#endif
  __fdiv_rn(4.0f, 2.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __fdiv_ru(4.0f, 2.0f);
  __fdiv_rz(4.0f, 2.0f);
#endif
  __fdividef(4.0f, 2.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __fmaf_rd(1.0f, 2.0f, 3.0f);
#endif
  __fmaf_rn(1.0f, 2.0f, 3.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __fmaf_ru(1.0f, 2.0f, 3.0f);
  __fmaf_rz(1.0f, 2.0f, 3.0f);
  __fmul_rd(1.0f, 2.0f);
#endif
  __fmul_rn(1.0f, 2.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __fmul_ru(1.0f, 2.0f);
  __fmul_rz(1.0f, 2.0f);
  __frcp_rd(2.0f);
#endif
  __frcp_rn(2.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __frcp_ru(2.0f);
  __frcp_rz(2.0f);
#endif
  __frsqrt_rn(4.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __fsqrt_rd(4.0f);
#endif
  __fsqrt_rn(4.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __fsqrt_ru(4.0f);
  __fsqrt_rz(4.0f);
  __fsub_rd(2.0f, 1.0f);
#endif
  __fsub_rn(2.0f, 1.0f);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  __fsub_ru(2.0f, 1.0f);
  __fsub_rz(2.0f, 1.0f);
#endif
  __log10f(1.0f);
  __log2f(1.0f);
  __logf(1.0f);
  __powf(1.0f, 0.0f);
  __saturatef(0.1f);
  __sincosf(0.0f, &fX, &fY);
  __sinf(0.0f);
  __tanf(0.0f);
}

__global__ void compileSinglePrecisionIntrinsics(int) { single_precision_intrinsics(); }

TEST_CASE(Unit_SinglePrecisionIntrinsics) {
  hipLaunchKernelGGL(compileSinglePrecisionIntrinsics, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, 1);
}

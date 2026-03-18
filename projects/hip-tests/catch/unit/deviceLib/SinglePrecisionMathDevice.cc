/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */


#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip/math_functions.h>

#pragma GCC diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wunused-variable"

__device__ void single_precision_math_functions() {
  int iX;
  float fX, fY;

  acosf(1.0f);
  acoshf(1.0f);
  asinf(0.0f);
  asinhf(0.0f);
  atan2f(0.0f, 1.0f);
  atanf(0.0f);
  atanhf(0.0f);
  cbrtf(0.0f);
  ceilf(0.0f);
  copysignf(1.0f, -2.0f);
  cosf(0.0f);
  coshf(0.0f);
  cospif(0.0f);
  erfcf(0.0f);
  erfcinvf(2.0f);
  erfcxf(0.0f);
  erff(0.0f);
  erfinvf(1.0f);
  exp10f(0.0f);
  exp2f(0.0f);
  expf(0.0f);
  expm1f(0.0f);
  fabsf(1.0f);
  fdimf(1.0f, 0.0f);
  fdividef(0.0f, 1.0f);
  floorf(0.0f);
  fmaf(1.0f, 2.0f, 3.0f);
  fmaxf(0.0f, 0.0f);
  fminf(0.0f, 0.0f);
  fmodf(0.0f, 1.0f);
  frexpf(0.0f, &iX);
  hypotf(1.0f, 0.0f);
  ilogbf(1.0f);
  isfinite(0.0f);
  isinf(0.0f);
  isnan(0.0f);
  j0f(0.0f);
  j1f(0.0f);
  jnf(-1.0f, 1.0f);
  ldexpf(0.0f, 0);
  llrintf(0.0f);
  llroundf(0.0f);
  log10f(1.0f);
  log1pf(-1.0f);
  log2f(1.0f);
  logbf(1.0f);
  logf(1.0f);
  lrintf(0.0f);
  lroundf(0.0f);
  nanf("1");
  nearbyintf(0.0f);
  norm3df(1.0f, 0.0f, 0.0f);
  norm4df(1.0f, 0.0f, 0.0f, 0.0f);
  normcdff(0.0f);
  normcdfinvf(1.0f);
  fX = 1.0f;
  normf(1, &fX);
  powf(1.0f, 0.0f);
  remainderf(2.0f, 1.0f);
  rhypotf(0.0f, 1.0f);
  rintf(1.0f);
  rnorm3df(0.0f, 0.0f, 1.0f);
  rnorm4df(0.0f, 0.0f, 0.0f, 1.0f);
  fX = 1.0f;
  rnormf(1, &fX);
  roundf(0.0f);
  rsqrtf(1.0f);
  signbit(1.0f);
  sincosf(0.0f, &fX, &fY);
  sincospif(0.0f, &fX, &fY);
  sinf(0.0f);
  sinhf(0.0f);
  sinpif(0.0f);
  sqrtf(0.0f);
  tanf(0.0f);
  tanhf(0.0f);
  tgammaf(2.0f);
  truncf(0.0f);
  y0f(1.0f);
  y1f(1.0f);
  ynf(1, 1.0f);
}

__global__ void compileSinglePrecisionMathOnDevice(int) { single_precision_math_functions(); }

TEST_CASE(Unit_SinglePrecisionMathDevice) {
  hipLaunchKernelGGL(compileSinglePrecisionMathOnDevice, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, 1);
}

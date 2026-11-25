/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Built with g++ -Wall -Wextra -Werror: low precision headers must stay usable from host TUs.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp6.h>
#include <hip/hip_fp4.h>

#include <iostream>

static float halfraw_to_float(__half_raw r) { return __half2float(__half(r)); }

static bool check(bool ok, const char* what, float got, float want) {
  if (!ok) std::cout << what << ": got " << got << ", want " << want << std::endl;
  return ok;
}

bool test_fp8_host() {
  const float in[] = {1.5f, -3.0f, 1000.0f, -1000.0f};
  for (float v : in) {
    __hip_fp8_storage_t s = __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
    float want = v > 448.0f ? 448.0f : (v < -448.0f ? -448.0f : v);
    float got = halfraw_to_float(__hip_cvt_fp8_to_halfraw(s, __HIP_E4M3));
    if (!check(got == want, "fp8 e4m3 -> half", got, want)) return false;
  }
  float2 f2;
  f2.x = 1000.0f;
  f2.y = -1.5f;
  __hip_fp8x2_storage_t s2 = __hip_cvt_float2_to_fp8x2(f2, __HIP_SATFINITE, __HIP_E4M3);
  __half2 h2 = __hip_cvt_fp8x2_to_halfraw2(s2, __HIP_E4M3);
  return check(__low2float(h2) == 448.0f, "fp8x2 e4m3 x", __low2float(h2), 448.0f) &&
      check(__high2float(h2) == -1.5f, "fp8x2 e4m3 y", __high2float(h2), -1.5f);
}

bool test_fp6_host() {
  const __half2_raw in = __half2(__float2half(1.5f), __float2half(-3.0f));
  __hip_fp6x2_storage_t s = __hip_cvt_halfraw2_to_fp6x2(in, __HIP_E2M3, hipRoundNearest);
  __half2 out = __hip_cvt_fp6x2_to_halfraw2(s, __HIP_E2M3);
  return check(__low2float(out) == 1.5f, "fp6x2 e2m3 x", __low2float(out), 1.5f) &&
      check(__high2float(out) == -3.0f, "fp6x2 e2m3 y", __high2float(out), -3.0f);
}

bool test_fp4_host() {
  __hip_fp4_storage_t s = __hip_cvt_float_to_fp4(1.5f, __HIP_E2M1, hipRoundNearest);
  float got = halfraw_to_float(__hip_cvt_fp4_to_halfraw(s, __HIP_E2M1));
  return check(got == 1.5f, "fp4 e2m1 -> half", got, 1.5f);
}

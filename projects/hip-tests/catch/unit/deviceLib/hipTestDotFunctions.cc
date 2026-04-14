/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip/math_functions.h>
#include <hip_test_common.hh>

__global__ static void DotFunctions(bool* result) {
// Dot Functions
#if HT_AMD
  short2 sa{1}, sb{1};
  result[0] = amd_mixed_dot(sa, sb, 1, result[0]) && result[0];

  ushort2 usa{1}, usb{1};
  result[0] = amd_mixed_dot(usa, usb, (uint)1, result[0]) && result[0];

  char4 ca{1}, cb{1};
  result[0] = amd_mixed_dot(ca, cb, 1, result[0]) && result[0];

  uchar4 uca{1}, ucb{1};
  result[0] = amd_mixed_dot(uca, ucb, (uint)1, result[0]) && result[0];

  int ia{1}, ib{1};
  result[0] = amd_mixed_dot(ia, ib, 1, result[0]) && result[0];

  uint ua{1}, ub{1};
  result[0] = amd_mixed_dot(ua, ub, (uint)1, result[0]) && result[0];
#endif
}

TEST_CASE(Unit_hipTestDotFunctions) {
  bool* result{nullptr};
  HIP_CHECK(hipHostMalloc(&result, 1));
  result[0] = true;
  hipLaunchKernelGGL(DotFunctions, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result);
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(result[0] == true);
  HIP_CHECK(hipHostFree(result));
}

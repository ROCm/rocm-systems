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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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

TEST_CASE("Unit_hipTestDotFunctions") {
  bool* result{nullptr};
  HIP_CHECK(hipHostMalloc(&result, 1));
  result[0] = true;
  hipLaunchKernelGGL(DotFunctions, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result);
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(result[0] == true);
  HIP_CHECK(hipHostFree(result));
}

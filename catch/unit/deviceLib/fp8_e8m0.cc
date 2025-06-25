/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/hip_fp8.h>

// Test fp8_e8m0 CVT operations
TEST_CASE("Unit_fp8_e8m0_cvt") {
  __hip_saturation_t saturation = GENERATE(__HIP_NOSAT, __HIP_SATFINITE);
  hipRoundMode rounding = GENERATE(hipRoundZero, hipRoundPosInf);
  {
      __hip_bfloat16 in = 42.5f;
      __hip_fp8_e8m0 out{__hip_cvt_bfloat16raw_to_e8m0(in, saturation, rounding)};
      __hip_fp8_e8m0 exp{0U};
      REQUIRE(out.__x == exp.__x);
  }
}

// Test fp8_e8m0 constructors
TEST_CASE("Unit_fp8_e8m0_constructors") {
  {
    __hip_bfloat16 in = 0.5;
    __hip_fp8_e8m0 res(in);
    __hip_bfloat16_raw res_f = (__hip_bfloat16_raw)res;
    REQUIRE(res_f == 0.5);
  }
}

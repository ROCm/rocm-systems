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

#include "hip_hc_8pk_negative_kernels_rtc.hh"

#include <hip_test_common.hh>
#include <resource_guards.hh>


/**
 * @addtogroup hip_hc_8pk hip_hc_8pk
 * @{
 * @ingroup DeviceLanguageTest
 */

__global__ void __hip_hc_add8pk_kernel(char4* out, char4 in1, char4 in2) {
  out[0] = __hip_hc_add8pk(in1, in2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__hip_hc_add8pk(in1, in2)`.
 *
 * Test source
 * ------------------------
 *    - unit/hip_specific/hip_hc_8pk.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device__hip_hc_add8pk_Sanity_Positive") {
  const char input1[] = {-0x70, -0x50, -0x30, -0x0f, 0x0, 0x01, 0x10, 0x20, 0x70, 0x7f};
  const char input2[] = {-0x05, -0x11, -0x20, -0x03, 0x0, 0x30, 0x05, 0x33, 0x0f, 0x7a};
  const char reference[] = {-0x75, -0x61, -0x50, -0x12, 0x0, 0x31, 0x15, 0x53, 0x7f, -0x07};
  LinearAllocGuard<char4> out(LinearAllocs::hipMallocManaged, sizeof(char4));

  for (int i = 0; i < 10; ++i) {
    __hip_hc_add8pk_kernel<<<1, 1>>>(out.ptr(), make_char4(0, 0, 0, input1[i]),
                                     make_char4(0, 0, 0, input2[i]));
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(out.ptr()[0].x == 0);
    REQUIRE(out.ptr()[0].y == 0);
    REQUIRE(out.ptr()[0].z == 0);
    REQUIRE(out.ptr()[0].w == reference[i]);
  }
}

__global__ void __hip_hc_sub8pk_kernel(char4* out, char4 in1, char4 in2) {
  out[0] = __hip_hc_sub8pk(in1, in2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__hip_hc_sub8pk(in1, in2)`.
 *
 * Test source
 * ------------------------
 *    - unit/hip_specific/hip_hc_8pk.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device__hip_hc_sub8pk_Sanity_Positive") {
  const char input1[] = {-0x70, -0x50, -0x30, -0x0f, 0x0, 0x30, 0x10, 0x33, 0x70, 0x7a};
  const char input2[] = {-0x05, -0x11, -0x20, -0x03, 0x0, 0x01, 0x05, 0x20, 0x0f, 0x7f};
  const char reference[] = {-0x6b, -0x3f, -0x10, -0x0c, 0x0, 0x2f, 0x0b, 0x13, 0x61, -0x05};
  LinearAllocGuard<char4> out(LinearAllocs::hipMallocManaged, sizeof(char4));

  for (int i = 0; i < 10; ++i) {
    __hip_hc_sub8pk_kernel<<<1, 1>>>(out.ptr(), make_char4(0, 0, 0, input1[i]),
                                     make_char4(0, 0, 0, input2[i]));
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(out.ptr()[0].x == 0);
    REQUIRE(out.ptr()[0].y == 0);
    REQUIRE(out.ptr()[0].z == 0);
    REQUIRE(out.ptr()[0].w == reference[i]);
  }
}

__global__ void __hip_hc_mul8pk_kernel(char4* out, char4 in1, char4 in2) {
  out[0] = __hip_hc_mul8pk(in1, in2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__hip_hc_mul8pk(in1, in2)`.
 *
 * Test source
 * ------------------------
 *    - unit/hip_specific/hip_hc_8pk.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device__hip_hc_mul8pk_Sanity_Positive") {
  const char input1[] = {-0x70, -0x50, -0x30, -0x0f, 0x0, 0x01, 0x10, 0x20, 0x70, 0x7f};
  const char input2[] = {0x05, -0x11, 0x22, -0x03, 0x0, 0x30, 0x05, 0x33, 0x0f, 0x7a};
  const char reference[] = {-0x30, 0x50, -0x60, 0x2d, 0x0, 0x30, 0x50, 0x60, -0x70, -0x7a};
  LinearAllocGuard<char4> out(LinearAllocs::hipMallocManaged, sizeof(char4));

  for (int i = 0; i < 10; ++i) {
    __hip_hc_mul8pk_kernel<<<1, 1>>>(out.ptr(), make_char4(0, 0, 0, input1[i]),
                                     make_char4(0, 0, 0, input2[i]));
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(out.ptr()[0].x == 0);
    REQUIRE(out.ptr()[0].y == 0);
    REQUIRE(out.ptr()[0].z == 0);
    REQUIRE(out.ptr()[0].w == reference[i]);
  }
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for
 * __hip_hc_<add/sub/mul>8pk
 * Test source
 * ------------------------
 *    - unit/hip_specific/hip_hc_8pk.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device__hip_hc_8pk_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source = GENERATE(kHipHcAdd8pkBasic, kHipHcAdd8pkVector, kHipHcSub8pkBasic,
                                       kHipHcSub8pkVector, kHipHcMul8pkBasic, kHipHcMul8pkVector);

  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "hip_hc_8pk_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  int expected_error_count{15};
  std::string error_message{"error:"};

  size_t n_pos = log.find(error_message, 0);
  while (n_pos != std::string::npos) {
    ++error_count;
    n_pos = log.find(error_message, n_pos + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_error_count);
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */

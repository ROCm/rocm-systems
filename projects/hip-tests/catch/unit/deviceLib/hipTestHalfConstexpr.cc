/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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


#include <hip/hip_fp16.h>
#include <hip_test_common.hh>
#include <cmath>


constexpr unsigned short kHalfTwo = 0x4000U;       // 2.0
constexpr unsigned short kHalfPi = 0x4248U;        // ~3.140625 (closest to pi)
constexpr unsigned short kHalfInf = 0x7C00U;       // +Infinity

/**
 * Constexpr function to create __half from bit pattern.
 * This pattern is used by libhipcxx for mathematical constants.
 */
__host__ __device__ constexpr __half makeHalfFromBits(unsigned short bits) {
  return __half{__half_raw{.x = bits}};
}

// Constexpr half-precision constants using the pattern from the bug report

constexpr __half kConstTwo = makeHalfFromBits(kHalfTwo);
constexpr __half kConstPi = makeHalfFromBits(kHalfPi);
constexpr __half kConstInf = makeHalfFromBits(kHalfInf);


/**
 * Device kernel that uses constexpr __half values.
 * Tests that constexpr __half can be used in device code.
 */
__global__ void testConstexprHalfDevice(float* results) {

  // Verify the constexpr values are usable in device code
  results[0] = __half2float(kConstTwo);
  results[1]= __half2float(kConstPi);

  // Test arithmetic with constexpr values
  results[2] = __half2float(__hmul(kConstTwo, kConstPi));   // 2 * pi
}

/**
 * Test Description
 * ------------------------
 * - Tests constexpr construction of __half from __half_raw{.x = bits}
 * - This is the pattern used by libhipcxx for mathematical constants
 * - Verifies fix for union active member issue in constexpr evaluation
 */
TEST_CASE("Unit_hipTestHalfConstexpr_HostConstexpr") {
  // Test that constexpr __half values have correct bit patterns on host

  SECTION("Two") {
    constexpr __half two = makeHalfFromBits(kHalfTwo);
    float f = __half2float(two);
    REQUIRE(f == 2.0f);
    REQUIRE(__half_as_ushort(two) == kHalfTwo);
  }

  SECTION("Pi approximation") {
    constexpr __half pi = makeHalfFromBits(kHalfPi);
    float f = __half2float(pi);
    // Half precision pi is approximately 3.140625
    REQUIRE(f == Catch::Approx(3.14159f).epsilon(0.01));
    REQUIRE(__half_as_ushort(pi) == kHalfPi);
  }

  SECTION("Infinity") {
    constexpr __half inf = makeHalfFromBits(kHalfInf);
    REQUIRE(__hisinf(inf));
    REQUIRE(__half_as_ushort(inf) == kHalfInf);
  }

  SECTION("File scope constexpr values") {
    // Test that file-scope constexpr values are correct
    REQUIRE(__half2float(kConstTwo) == 2.0f);
  }
}

/**
 * Test Description
 * ------------------------
 * - Tests that constexpr __half values can be used in device kernels
 * - Verifies both file-scope and kernel-local constexpr values work
 */
TEST_CASE("Unit_hipTestHalfConstexpr_DeviceConstexpr") {
  constexpr size_t numResults = 3;
  float* results_d = nullptr;
  std::vector<float> results_h(numResults, 0.0f);

  HIP_CHECK(hipMalloc(&results_d, numResults * sizeof(float)));

  testConstexprHalfDevice<<<1, 1>>>(results_d);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(results_h.data(), results_d, numResults * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Verify results
  REQUIRE(results_h[0] == 2.0f);       // kConstTwo
  REQUIRE(results_h[1] == Catch::Approx(3.14159f).epsilon(0.01));  // kConstPi
  REQUIRE(results_h[2] == Catch::Approx(6.28f).epsilon(0.01));  // 2 * pi

  HIP_CHECK(hipFree(results_d));
}



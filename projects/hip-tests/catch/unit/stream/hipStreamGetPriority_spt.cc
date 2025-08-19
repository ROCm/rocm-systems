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
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipStreamGetPriority_spt hipStreamGetPriority_spt
 * @{
 * @ingroup StreamTest
 * `hipStreamGetPriority_spt(hipStream_t stream, int* priority)` -
 * Query the priority of a stream
 */

/**
 * Test Description
 * ------------------------
 * - Test to create a stream and check priority.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamGetPriority_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamGetPriority_spt_BasicTst") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream{};
  int priority = 0;
  SECTION("Null Stream") {
    HIP_CHECK(hipStreamGetPriority_spt(nullptr, &priority));
    // valid priority
    REQUIRE(priority_low >= priority);
    REQUIRE(priority >= priority_high);
  }
  SECTION("Created Stream") {
    SECTION("Default Priority") {
      HIP_CHECK(hipStreamCreate(&stream));
      HIP_CHECK(hipStreamGetPriority_spt(stream, &priority));
      // valid priority
      // Lower the value higher the priority,
      // higher the value lower the priority
      REQUIRE(priority_low >= priority);
      REQUIRE(priority >= priority_high);
    }
    SECTION("High Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_high));
      HIP_CHECK(hipStreamGetPriority_spt(stream, &priority));
      REQUIRE(priority == priority_high);
    }
    SECTION("Higher Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_high - 1));
      HIP_CHECK(hipStreamGetPriority_spt(stream, &priority));
      REQUIRE(priority == priority_high);
    }
    SECTION("Low Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_low));
      HIP_CHECK(hipStreamGetPriority_spt(stream, &priority));
      REQUIRE(priority_low == priority);
    }
    SECTION("Lower Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_low + 1));
      HIP_CHECK(hipStreamGetPriority_spt(stream, &priority));
      REQUIRE(priority_low == priority);
    }
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * Test Description
 * ------------------------
 * - Test check stream priority with negative cases.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamGetPriority_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamGetPriority_spt_NegativeCases") {
  // stream as nullptr and priority as nullptr
  auto res = hipStreamGetPriority_spt(nullptr, nullptr);
  REQUIRE(res == hipErrorInvalidValue);
  // priority as nullptr
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  res = hipStreamGetPriority_spt(stream, nullptr);
  REQUIRE(res == hipErrorInvalidValue);
  HIP_CHECK(hipStreamDestroy(stream));
  // stream as nullptr
  int priority = -1;
  HIP_CHECK(hipStreamGetPriority_spt(nullptr, &priority));
}
/**
 * Test Description
 * ------------------------
 * - Create stream with CUMask and check priority is returned as expected.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamGetPriority_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */

#if HT_AMD
TEST_CASE("Unit_hipStreamGetPriority_spt_StreamsWithCUMask") {
  hipStream_t stream{};
  int priority = 0;
  int priority_normal = 0;
  int priority_low = 0;
  int priority_high = 0;
  // Test is to get the Stream Priority Range
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  priority_normal = (priority_low + priority_high) / 2;
  // Check if priorities are indeed supported
  REQUIRE_FALSE(priority_low == priority_high);
  // Creating a stream with hipExtStreamCreateWithCUMask and checking
  // priority.
  const uint32_t cuMask = 0xffffffff;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, 1, &cuMask));
  HIP_CHECK(hipStreamGetPriority_spt(stream, &priority));
  REQUIRE_FALSE(priority_normal != priority);
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif

/**
 * End doxygen group StreamTest.
 * @}
 */

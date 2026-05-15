/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
1) Negative tests for hipStreamGetPriority api.
2) Create stream and check default priority of stream is within range.
3) Create stream with high or low priority and check priority is set as expected.
4) Create stream with higher priority or lower priority for the priority range returned, the stream
priority should be clamped to the priority range.
5) Create stream with CUMask and check priority is returned as expected.
*/

#include <hip_test_common.hh>

/**
 * Create stream and check priority.
 */
HIP_TEST_CASE(Unit_hipStreamGetPriority_happy) {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream{};
  int priority = 0;
  SECTION("Null Stream") {
    HIP_CHECK(hipStreamGetPriority(nullptr, &priority));
    // valid priority
    REQUIRE(priority_low >= priority);
    REQUIRE(priority >= priority_high);
  }
  SECTION("Created Stream") {
    SECTION("Default Priority") {
      HIP_CHECK(hipStreamCreate(&stream));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      // valid priority
      // Lower the value higher the priority, higher the value lower the priority
      REQUIRE(priority_low >= priority);
      REQUIRE(priority >= priority_high);
    }
    SECTION("High Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_high));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      REQUIRE(priority == priority_high);
    }
    SECTION("Higher Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_high - 1));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      REQUIRE(priority == priority_high);
    }
    SECTION("Low Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_low));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      REQUIRE(priority_low == priority);
    }
    SECTION("Lower Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_low + 1));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      REQUIRE(priority_low == priority);
    }
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * both stream and priority passed as nullptr.
 */
HIP_TEST_CASE(Unit_hipStreamGetPriority_nullptr_nullptr) {
  auto res = hipStreamGetPriority(nullptr, nullptr);
  REQUIRE(res == hipErrorInvalidValue);
}


/**
 * valid stream and priority passed as nullptr.
 */
HIP_TEST_CASE(Unit_hipStreamGetPriority_stream_nullptr) {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  auto res = hipStreamGetPriority(stream, nullptr);
  REQUIRE(res == hipErrorInvalidValue);

  HIP_CHECK(hipStreamDestroy(stream));
}


/**
 * nullptr stream and valid priority
 */
HIP_TEST_CASE(Unit_hipStreamGetPriority_nullptr_priority) {
  int priority = -1;
  HIP_CHECK(hipStreamGetPriority(nullptr, &priority));
}

/**
 * both stream and priority passed as valid.
 */
HIP_TEST_CASE(Unit_hipStreamGetPriority_stream_priority) {
  int priority = -1;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipStreamGetPriority(stream, &priority));

  HIP_CHECK(hipStreamDestroy(stream));
}

#if HT_AMD
/**
 * Create stream with CUMask and check priority is returned as expected.
 */
HIP_TEST_CASE(Unit_hipStreamGetPriority_StreamsWithCUMask) {
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
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE_FALSE(priority_normal != priority);
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif

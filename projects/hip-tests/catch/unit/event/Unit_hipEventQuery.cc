/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <utils.hh>
/**
 * @addtogroup hipEventQuery hipEventQuery
 * @{
 * @ingroup EventTest
 * `hipEventQuery(hipEvent_t event)` -
 * Query the status of the specified event.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipEventIpc
 */

/**
 * Test Description
 * ------------------------
 *  - Query events with a single and with multiple devices.
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventQuery.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipEventQuery_DifferentDevice) {
  hipEvent_t event1{}, event2{};
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  REQUIRE(event1 != nullptr);
  REQUIRE(event2 != nullptr);

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);

  {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventRecord(event1, stream));
    LaunchDelayKernel(std::chrono::milliseconds(3000), stream);
    HIP_CHECK(hipEventRecord(event2, stream));

    HIP_CHECK(hipEventSynchronize(event1));
    HIP_CHECK(hipEventQuery(event1));  // Should be done

    HIP_CHECK_ERROR(hipEventQuery(event2),
                    hipErrorNotReady);  // Wont be done since stream is blocked
  }

  // If other devices are available, set it
  if (HipTest::getDeviceCount() > 1) {
    HIP_CHECK(hipSetDevice(1));
  }

  // Query from same or other device depending on availability
  {
    HIP_CHECK(hipEventQuery(event1));
    HIP_CHECK_ERROR(hipEventQuery(event2), hipErrorNotReady);

    HIP_CHECK(hipEventSynchronize(event2));

    // Query, should be done now
    HIP_CHECK(hipEventQuery(event2));
  }

  // Query on same device if multiple devices are present
  if (HipTest::getDeviceCount() > 1) {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventQuery(event2));
  }

  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group EventTest.
 * @}
 */

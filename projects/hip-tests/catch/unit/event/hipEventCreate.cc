/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipEventCreate hipEventCreate
 * @{
 * @ingroup EventTest
 * `hipEventCreate(hipEvent_t* event)` -
 * Create an event.
 */

/**
 * Test Description
 * ------------------------
 *  - Successfully creates and event for each device.
 * Test source
 * ------------------------
 *  - unit/event/hipEventCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventCreate_Positive) {
  int id = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(id));

  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);

  HIP_CHECK(hipEventDestroy(event));
}

/**
 * Test Description
 * ------------------------
 *  - Test event creation while stream is capturing.
 * Test source
 * ------------------------
 *  - unit/event/hipEventCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventCreate_Verify_Capture) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamCaptureMode mode = GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
                                       hipStreamCaptureModeRelaxed);
  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group hipEventCreate.
 * @}
 */

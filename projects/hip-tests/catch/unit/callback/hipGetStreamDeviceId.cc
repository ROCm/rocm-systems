/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <resource_guards.hh>

/**
 * @addtogroup hipGetStreamDeviceId hipGetStreamDeviceId
 * @{
 * @ingroup CallbackTest
 * `hipGetStreamDeviceId(hipStream_t stream)` -
 * returns the ID of the device on which the stream is active
 */

/**
 * Test Description
 * ------------------------
 *  - Creates a new stream for each available device
 *  - Verifies that the Device Stream ID is equal to the Device ID
 * Test source
 * ------------------------
 *  - unit/callback/hipGetStreamDeviceId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
HIP_TEST_CASE(Unit_hipGetStreamDeviceId_Positive_Threaded_Basic) {
  int id = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(id));

  StreamGuard stream_guard{Streams::created};
  REQUIRE(hipGetStreamDeviceId(stream_guard.stream()) == id);
}

/**
 * Test Description
 * ------------------------
 *  - Creates a new stream for each available device, through multiple threads
 *  - Verifies that the Device Stream ID is equal to the Device ID, from each thread
 * Test source
 * ------------------------
 *  - unit/callback/hipGetStreamDeviceId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 *  - Multithreaded GPU
 */
HIP_TEST_CASE(Unit_hipGetStreamDeviceId_Positive_Multithreaded_Basic) {
  const unsigned int max_threads = std::thread::hardware_concurrency();
  const int device_count = HipTest::getDeviceCount();

  auto thread_function = [&]() {
    for (int id = 0; id < device_count; ++id) {
      HIP_CHECK_THREAD(hipSetDevice(id));

      StreamGuard stream_guard{Streams::perThread};
      REQUIRE_THREAD(hipGetStreamDeviceId(stream_guard.stream()) == id);
    }
  };

  std::vector<std::thread> thread_pool;
  for (unsigned int i = 0; i < max_threads; ++i) {
    thread_pool.emplace_back(thread_function);
  }

  for (auto& thread : thread_pool) {
    thread.join();
  }

  HIP_CHECK_THREAD_FINALIZE();
}

/**
 * Test Description
 * ------------------------
 *  - Checks that function returns valid ID if the stream is `nullptr`
 * Test source
 * ------------------------
 *  - unit/callback/hipGetStreamDeviceId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
HIP_TEST_CASE(Unit_hipGetStreamDeviceId_Negative_Parameters) {
  int id = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(id));

  StreamGuard stream_guard{Streams::nullstream};
  REQUIRE(hipGetStreamDeviceId(stream_guard.stream()) == id);
}

/**
 * End doxygen group CallbackTest.
 * @}
 */

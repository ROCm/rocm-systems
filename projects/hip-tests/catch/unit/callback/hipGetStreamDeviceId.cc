/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
TEST_CASE("Unit_hipGetStreamDeviceId_Positive_Threaded_Basic") {
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
TEST_CASE("Unit_hipGetStreamDeviceId_Positive_Multithreaded_Basic") {
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
TEST_CASE("Unit_hipGetStreamDeviceId_Negative_Parameters") {
  int id = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(id));

  StreamGuard stream_guard{Streams::nullstream};
  REQUIRE(hipGetStreamDeviceId(stream_guard.stream()) == id);
}

/**
 * End doxygen group CallbackTest.
 * @}
 */

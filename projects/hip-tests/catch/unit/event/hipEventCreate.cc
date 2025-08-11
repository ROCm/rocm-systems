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
TEST_CASE("Unit_hipEventCreate_Positive") {
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
TEST_CASE("Unit_hipEventCreate_Verify_Capture") {
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

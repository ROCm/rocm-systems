/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip_test_checkers.hh>
#include <iostream>
#include <utils.hh>
/**
 * @addtogroup hipEventElapsedTime hipEventElapsedTime
 * @{
 * @ingroup EventTest
 * `hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop)` -
 * Return the elapsed time between two events.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipEvent
 *  - @ref Unit_hipEventIpc
 *  - @ref Unit_hipEventMGpuMThreads_1
 *  - @ref Unit_hipEventMGpuMThreads_2
 *  - @ref Unit_hipEventMGpuMThreads_3
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output parameter for time is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When first event is `nullptr`
 *      - Expected output: return `hipErrorInvalidHandle`
 *    -# When second event is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventElapsedTime.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipEventElapsedTime_NullCheck") {
  hipEvent_t start, end;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&end));

  float tms = 1.0f;
  HIP_ASSERT(hipEventElapsedTime(nullptr, start, end) == hipErrorInvalidValue);
#ifndef __HIP_PLATFORM_NVIDIA__
  // On NVCC platform API throws seg fault hence skipping
  HIP_ASSERT(hipEventElapsedTime(&tms, nullptr, end) == hipErrorInvalidHandle);
  HIP_ASSERT(hipEventElapsedTime(&tms, start, nullptr) == hipErrorInvalidHandle);
#endif
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(end));
}

/**
 * Test Description
 * ------------------------
 *  - Calculates elapsed time when events are created with disable timing flag
 *    -# When flag is set to disable timing
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventElapsedTime.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipEventElapsedTime_DisableTiming") {
  float timeElapsed = 1.0f;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreateWithFlags(&start, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&stop, hipEventDisableTiming));
  HIP_ASSERT(hipEventElapsedTime(&timeElapsed, start, stop) == hipErrorInvalidHandle);
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}

/**
 * Test Description
 * ------------------------
 *  - Calculates elapsed time when events are recorded on different devices
 *    -# When start and stop events are recorded on different devices
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventElapsedTime.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipEventElapsedTime_DifferentDevices") {
  int devCount = 0;
  HIP_CHECK(hipGetDeviceCount(&devCount));
  if (devCount > 1) {
    // create event on dev=0
    HIP_CHECK(hipSetDevice(0));
    hipEvent_t start;
    HIP_CHECK(hipEventCreate(&start));

    HIP_CHECK(hipEventRecord(start, nullptr));
    HIP_CHECK(hipEventSynchronize(start));

    // create event on dev=1
    HIP_CHECK(hipSetDevice(1));
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));

    float tElapsed = 1.0f;
    // start on device 0 but stop on device 1
    HIP_ASSERT(hipEventElapsedTime(&tElapsed, start, stop) == hipErrorInvalidHandle);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
  }
}


#if HT_AMD /* Disabled because frequency based wait is timing out on nvidia platforms */
/**
 * Test Description
 * ------------------------
 *  - Calculates elapsed time when an event has not been completed.
 *    -# When the stop event has not finished yet
 *      - Expected output: return `hipErrorNotReady`
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventElapsedTime.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipEventElapsedTime_NotReady_Negative") {
  hipEvent_t start;
  HIP_CHECK(hipEventCreate(&start));

  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&stop));

  // Record start event
  HIP_CHECK(hipEventRecord(start, nullptr));

  LaunchDelayKernel(std::chrono::milliseconds(1000));
  // Record stop event
  HIP_CHECK(hipEventRecord(stop, nullptr));

  // stop event has not been completed
  float tElapsed = 1.0f;
  HIP_CHECK_ERROR(hipEventQuery(stop), hipErrorNotReady);
  HIP_ASSERT(hipEventElapsedTime(&tElapsed, start, stop) == hipErrorNotReady);

  HIP_CHECK(hipStreamSynchronize(nullptr));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}
#endif  // HT_AMD

/**
 * Test Description
 * ------------------------
 *  - Calculates elapsed time between two successfully created and recorded events.
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventElapsedTime.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipEventElapsedTime") {
  hipEvent_t start;
  HIP_CHECK(hipEventCreate(&start));

  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, nullptr));
  HIP_CHECK(hipEventSynchronize(start));

  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));

  float tElapsed = 1.0f;
  HIP_CHECK(hipEventElapsedTime(&tElapsed, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}

TEST_CASE("Unit_hipEventElapsedTime_Verify_Capture") {
  hipEvent_t start, stop;

  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, nullptr));
  HIP_CHECK(hipEventSynchronize(start));
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipStreamCaptureMode mode = GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
                                       hipStreamCaptureModeRelaxed);
  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  float tElapsed = 1.0f;
  HIP_CHECK(hipEventElapsedTime(&tElapsed, start, stop));
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}

/**
 * End doxygen group EventTest.
 * @}
 */

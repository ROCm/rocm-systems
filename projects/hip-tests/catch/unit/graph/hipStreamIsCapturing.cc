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
#include <hip_test_kernels.hh>

#include "stream_capture_common.hh"

/**
 * @addtogroup hipStreamIsCapturing hipStreamIsCapturing
 * @{
 * @ingroup GraphTest
 * `hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus
 * *pCaptureStatus)` - get stream's capture state
 */

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Capture status is nullptr
 *        -# Capture status is checked on null stream
 *        -# Stream is uninitialized
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamIsCapturing.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamIsCapturing_Negative_Parameters") {
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  SECTION("Check capture status with null pCaptureStatus.") {
    HIP_CHECK_ERROR(hipStreamIsCapturing(stream, nullptr), hipErrorInvalidValue);
  }

  SECTION("Check capture status when checked on different streams") {
    hipStreamCaptureStatus cStatus;
    hipGraph_t graph{nullptr};
    hipStreamCaptureMode flags = GENERATE(
        hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);
    HIP_CHECK(hipStreamBeginCapture(stream, flags));
    SECTION("Check capture status with null stream") {
      HIP_CHECK_ERROR(hipStreamIsCapturing(nullptr, &cStatus), hipErrorStreamCaptureImplicit);
    }
    SECTION("Check capture status with hipStreamLegacy") {
      HIP_CHECK_ERROR(hipStreamIsCapturing(hipStreamLegacy, &cStatus), hipErrorStreamCaptureImplicit);
    }
    SECTION("Check capture status with hipStreamPerThread") {
      HIP_CHECK_ERROR(hipStreamIsCapturing(hipStreamPerThread, &cStatus), hipSuccess);
    }
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphDestroy(graph));
  }
}
TEST_CASE("Unit_hipStreamIsCapturing_NegTst_NonBlockingStream") {
  hipStreamCaptureStatus cStatus;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hipGraph_t graph{nullptr};
  hipStreamCaptureMode flags = GENERATE(
        hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);
    HIP_CHECK(hipStreamBeginCapture(stream, flags));
    SECTION("Check capture status with null stream") {
      HIP_CHECK_ERROR(hipStreamIsCapturing(nullptr, &cStatus), hipSuccess);
    }
    SECTION("Check capture status with hipStreamLegacy") {
      HIP_CHECK_ERROR(hipStreamIsCapturing(hipStreamLegacy, &cStatus), hipSuccess);
    }
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamIsCapturing(stream, &cStatus));
  REQUIRE(hipStreamCaptureStatusNone == cStatus);
}
TEST_CASE("Unit_hipStreamIsCapturing_NegTst_UnsupportedApi") {
  hipStream_t stream{nullptr};
  hipGraph_t graph{nullptr};
  hipError_t err;
  HIP_CHECK(hipStreamCreate(&stream));
  unsigned int *A_mem{nullptr};
  hipStreamCaptureMode flags = GENERATE(
        hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);
  HIP_CHECK(hipStreamBeginCapture(stream, flags));
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone},
                         captureStatus1{hipStreamCaptureStatusNone};
  // Check the capture status of the stream
  HIP_CHECK(hipStreamIsCapturing(stream, &captureStatus));
  assert(captureStatus == hipStreamCaptureStatusActive);
  err = hipMalloc(&A_mem, 1024);
  REQUIRE(err == hipErrorStreamCaptureUnsupported);
  HIP_CHECK(hipStreamIsCapturing(stream, &captureStatus1));
  assert(captureStatus1 == hipStreamCaptureStatusInvalidated);
  // End the capture
  HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph), hipErrorStreamCaptureInvalidated);
}

/**
 * Test Description
 * ------------------------
 *    - Initiate simple API call for stream capture status on custom
 * stream/hipStreamPerThread
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamIsCapturing.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamIsCapturing_Positive_Basic") {
  hipStreamCaptureStatus cStatus;
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  HIP_CHECK(hipStreamIsCapturing(stream, &cStatus));
  REQUIRE(hipStreamCaptureStatusNone == cStatus);
}

void checkStreamCaptureStatus(hipStreamCaptureMode mode, hipStream_t stream) {
  constexpr size_t N = 1000000;

  hipStreamCaptureStatus cStatus;
  size_t Nbytes = N * sizeof(float);
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);

  // Status is none before capture begins
  HIP_CHECK(hipStreamIsCapturing(stream, &cStatus));
  REQUIRE(hipStreamCaptureStatusNone == cStatus);

  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), N, stream);

  // Status is active during stream capture
  HIP_CHECK(hipStreamIsCapturing(stream, &cStatus));
  REQUIRE(hipStreamCaptureStatusActive == cStatus);

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  // Status is none after capture ends
  HIP_CHECK(hipStreamIsCapturing(stream, &cStatus));
  REQUIRE(hipStreamCaptureStatusNone == cStatus);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), N, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i), N);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec))
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    -  Initiate stream capture with different modes on custom
 * stream/hipStreamPerThread. Check that capture status is correct in different
 * capturing phases
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamIsCapturing.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamIsCapturing_Positive_Functional") {
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);

  checkStreamCaptureStatus(captureMode, stream);
}

static void thread_func(hipStream_t stream) {
  hipStreamCaptureStatus cStatus;
  HIP_CHECK(hipStreamIsCapturing(stream, &cStatus));
  REQUIRE(hipStreamCaptureStatusActive == cStatus);
}

/**
 * Test Description
 * ------------------------
 *    -  Initiate stream capture with different modes on custom
 * stream/hipStreamPerThread. Check that capture status is correct when status
 * is checked in a separate thread
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamIsCapturing.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamIsCapturing_Positive_Thread") {
  constexpr size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);

  hipGraph_t graph{nullptr};
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);

  const hipStreamCaptureMode captureMode = hipStreamCaptureModeGlobal;

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), N, stream);

  std::thread t(thread_func, stream);
  t.join();

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */

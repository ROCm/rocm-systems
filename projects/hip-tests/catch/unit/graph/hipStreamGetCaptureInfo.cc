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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include "stream_capture_common.hh"

/**
 * @addtogroup hipStreamGetCaptureInfo hipStreamGetCaptureInfo
 * @{
 * @ingroup GraphTest
 * `hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus
 * *pCaptureStatus, unsigned long long *pId)` - get capture status of a stream
 */

void checkStreamCaptureInfo(hipStreamCaptureMode mode, hipStream_t stream) {
  constexpr size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  unsigned long long capSequenceID = 0;  // NOLINT

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);

  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), N, stream);

  // Capture status is active and sequence id is valid
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, &capSequenceID));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capSequenceID > 0);

  // End capture and verify graph is returned
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  // verify capture status is inactive and sequence id is not updated
  capSequenceID = 0;
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, &capSequenceID));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
  REQUIRE(capSequenceID == 0);

  // Verify api still returns capture status when capture ID is nullptr
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, nullptr));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);

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
 *    - Test to verify that hipStreamCaptureStatusActive is returned during
 * stream capture. When capture is ended, status is changed to
 * hipStreamCaptureStatusNone and error is not reported when some arguments are
 * not passed
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_Positive_Functional") {
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);

  checkStreamCaptureInfo(captureMode, stream);
}

/**
 * Test Description
 * ------------------------
 *    - Test starts stream capture on multiple streams and verifies uniqueness
 * of identifiers returned
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_Positive_UniqueID") {
  constexpr int numStreams = 100;
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  std::vector<int> idlist;
  unsigned long long capSequenceID{};  // NOLINT
  hipGraph_t graph{nullptr};

  StreamsGuard streams(numStreams);

  for (int i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamBeginCapture(streams[i], hipStreamCaptureModeGlobal));
    HIP_CHECK(hipStreamGetCaptureInfo(streams[i], &captureStatus, &capSequenceID));
    REQUIRE(captureStatus == hipStreamCaptureStatusActive);
    REQUIRE(capSequenceID > 0);
    idlist.push_back(capSequenceID);
  }

  for (int i = 0; i < numStreams; i++) {
    for (int j = i + 1; j < numStreams; j++) {
      if (idlist[i] == idlist[j]) {
        INFO("Same identifier returned for stream " << i << " and stream " << j);
        REQUIRE(false);
      }
    }
  }

  for (int i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamEndCapture(streams[i], &graph));
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Capture status is nullptr
 *        -# Capture status checked on legacy/null stream
 *        -# Stream is uninitialized
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_Negative_Parameters") {
#if HT_NVIDIA
  hipStreamCaptureStatus cStatus;
#endif
  unsigned long long capSequenceID;  // NOLINT
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  SECTION("Capture Status location as nullptr") {
    HIP_CHECK_ERROR(hipStreamGetCaptureInfo(stream, nullptr, &capSequenceID), hipErrorInvalidValue);
  }
#if HT_NVIDIA  // EXSWHTEC-216, EXSWHTEC-228
  SECTION("Capture status when checked on null stream") {
    hipGraph_t graph{nullptr};
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    HIP_CHECK_ERROR(hipStreamGetCaptureInfo(nullptr, &cStatus, &capSequenceID),
                    hipErrorStreamCaptureImplicit);
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphDestroy(graph));
  }
#endif
}

/**
 * End doxygen group GraphTest.
 * @}
 */

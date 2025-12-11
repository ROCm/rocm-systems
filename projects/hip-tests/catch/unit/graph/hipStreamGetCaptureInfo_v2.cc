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
 * @addtogroup hipStreamGetCaptureInfo_v2 hipStreamGetCaptureInfo_v2
 * @{
 * @ingroup GraphTest
 * `hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus
 * *captureStatus_out, unsigned long long *id_out __dparm(0), hipGraph_t
 * *graph_out __dparm(0), const hipGraphNode_t **dependencies_out __dparm(0),
 * size_t *numDependencies_out __dparm(0)))` - Get stream's capture state
 */

void checkStreamCaptureInfo_v2(hipStreamCaptureMode mode, hipStream_t stream) {
  constexpr size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);

  hipGraph_t graph{nullptr}, capInfoGraph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  const hipGraphNode_t* nodelist{};
  size_t numDepsCreated = 0;
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  hipGraphNodeType type(hipGraphNodeTypeEmpty);
  unsigned long long capSequenceID = 0;  // NOLINT
  size_t numDependencies;

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<float> B_d(LinearAllocs::hipMalloc, Nbytes);

  EventsGuard events_guard(3);
  StreamsGuard streams_guard(2);

  SECTION("Linear sequence graph") {
    HIP_CHECK(hipStreamBeginCapture(stream, mode));
    captureSequenceLinear(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream);
    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, &capSequenceID, &capInfoGraph,
                                         &nodelist, &numDependencies));
    numDepsCreated = 1;
    HIP_CHECK(hipGraphNodeGetType(nodelist[0], &type));
    if ((type != hipGraphNodeTypeMemset) && (type != hipGraphNodeTypeMemcpy)) {
      INFO("Type0 returned as " << type);
      REQUIRE(false);
    }
  }

  SECTION("Branched sequence graph") {
    HIP_CHECK(hipStreamBeginCapture(stream, mode));
    captureSequenceBranched(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream,
                            streams_guard.stream_list(), events_guard.event_list());
    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, &capSequenceID, &capInfoGraph,
                                         &nodelist, &numDependencies));
    numDepsCreated = 2;
    HIP_CHECK(hipGraphNodeGetType(nodelist[0], &type));
    if ((type != hipGraphNodeTypeMemset) && (type != hipGraphNodeTypeMemcpy)) {
      INFO("Type0 returned as " << type);
      REQUIRE(false);
    }
    HIP_CHECK(hipGraphNodeGetType(nodelist[1], &type));
    if ((type != hipGraphNodeTypeMemset) && (type != hipGraphNodeTypeMemcpy)) {
      INFO("Type1 returned as " << type);
      REQUIRE(false);
    }
  }

  // verify capture status is active, sequence id is valid, graph is returned,
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capSequenceID > 0);
  REQUIRE(capInfoGraph != nullptr);
  REQUIRE(numDependencies == numDepsCreated);

  captureSequenceCompute(A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream);

  // End capture and verify graph is returned
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  // verify capture status is inactive and other params are not updated
  capSequenceID = 0;
  capInfoGraph = nullptr;
  numDependencies = 0;
  nodelist = nullptr;
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, &capSequenceID, &capInfoGraph,
                                       &nodelist, &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
  REQUIRE(capSequenceID == 0);
  REQUIRE(capInfoGraph == nullptr);
  REQUIRE(nodelist == nullptr);
  REQUIRE(numDependencies == 0);

  // Verify api still returns capture status when optional args are not passed
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), N, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i) * static_cast<float>(i), N);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec))
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify that hipStreamCaptureStatusActive is returned during
 * stream capture, correct number of created dependencies is returned and
 * sequence ID is valid. When capture is ended, status is changed to
 * hipStreamCaptureStatusNone and error is not reported when some arguments are
 * not passed.
 *        -# Sequence graph is linear, number of created dependencies is 1, node
 * type is correct
 *        -# Sequence graph is branched, number of created dependencies is 2,
 * node types are correct
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v2.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_v2_Positive_Functional") {
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);

  checkStreamCaptureInfo_v2(captureMode, stream);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify stream capture on multiple streams and verifies
 * uniqueness of identifiers returned from capture Info V2:
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v2.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_v2_Positive_UniqueID") {
  constexpr int numStreams = 100;
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  std::vector<int> idlist;
  unsigned long long capSequenceID{};  // NOLINT
  hipGraph_t graph{nullptr};

  StreamsGuard streams(numStreams);

  for (int i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamBeginCapture(streams[i], hipStreamCaptureModeGlobal));
    HIP_CHECK(hipStreamGetCaptureInfo_v2(streams[i], &captureStatus, &capSequenceID, nullptr,
                                         nullptr, nullptr));
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
 *        -# Capture status when stream is uninitialized
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v2.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_v2_Negative_Parameters") {
  hipGraph_t capInfoGraph{};
#if HT_NVIDIA
  hipStreamCaptureStatus captureStatus;
#endif
  unsigned long long capSequenceID;  // NOLINT
  size_t numDependencies;
  const hipGraphNode_t* nodelist{};

  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  SECTION("Capture Status location as nullptr") {
    HIP_CHECK_ERROR(hipStreamGetCaptureInfo_v2(stream, nullptr, &capSequenceID, &capInfoGraph,
                                               &nodelist, &numDependencies),
                    hipErrorInvalidValue);
  }
#if HT_NVIDIA  // EXSWHTEC-216, EXSWHTEC-228
  SECTION("Capture status when checked on null stream") {
    hipGraph_t graph{nullptr};
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    HIP_CHECK_ERROR(hipStreamGetCaptureInfo_v2(nullptr, &captureStatus, &capSequenceID,
                                               &capInfoGraph, &nodelist, &numDependencies),
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

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include "stream_capture_common.hh"

/**
 * @addtogroup hipStreamGetCaptureInfo_v3 hipStreamGetCaptureInfo_v3
 * @{
 * @ingroup GraphTest
 * `hipStreamGetCaptureInfo_v3(hipStream_t stream, hipStreamCaptureStatus
 * *captureStatus_out, unsigned long long *id_out __dparm(0), hipGraph_t
 * *graph_out __dparm(0), const hipGraphNode_t **dependencies_out __dparm(0),
 * const hipGraphEdgeData **edgeData_out __dparm(0),
 * size_t *numDependencies_out __dparm(0)))` - Get stream's capture state
 * including the edge data annotating each dependency.
 */

namespace {
// Only default edge annotations exist, so every entry must be fully zero-initialized - CUDA's
// encoding for a full serialization of two nodes with memory visibility.
void requireDefaultEdgeData(const hipGraphEdgeData& edge) {
  REQUIRE(edge.from_port == 0);
  REQUIRE(edge.to_port == 0);
  REQUIRE(edge.type == 0);
  for (unsigned char reserved_byte : edge.reserved) {
    REQUIRE(reserved_byte == 0);
  }
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Verifies that an idle stream reports hipStreamCaptureStatusNone. Outputs other than the
 * capture status are only guaranteed when capture is active, so they are deliberately not
 * asserted here.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipStreamGetCaptureInfo_v3_Positive_NotCapturing) {
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusActive};
  const hipGraphNode_t* nodelist{};
  const hipGraphEdgeData* edgeData{};
  size_t numDependencies{};

  HIP_CHECK(hipStreamGetCaptureInfo_v3(stream, &captureStatus, nullptr, nullptr, &nodelist,
                                       &edgeData, &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
}

/**
 * Test Description
 * ------------------------
 *    - Captures work on a stream and verifies that every returned edge carries default
 * (zero-initialized) edge data, and that the edge array length matches the dependency count.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipStreamGetCaptureInfo_v3_Positive_DefaultEdgeData) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(float);

  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<float> B_d(LinearAllocs::hipMalloc, Nbytes);

  hipGraph_t graph{nullptr};
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  unsigned long long capSequenceID{};  // NOLINT
  const hipGraphNode_t* nodelist{};
  const hipGraphEdgeData* edgeData{};
  size_t numDependencies{};

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(A_d.ptr(), 0, Nbytes, stream));
  HIP_CHECK(hipMemcpyAsync(B_d.ptr(), A_d.ptr(), Nbytes, hipMemcpyDeviceToDevice, stream));

  HIP_CHECK(hipStreamGetCaptureInfo_v3(stream, &captureStatus, &capSequenceID, nullptr, &nodelist,
                                       &edgeData, &numDependencies));

  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capSequenceID > 0);
  REQUIRE(numDependencies > 0);
  REQUIRE(nodelist != nullptr);
  REQUIRE(edgeData != nullptr);

  for (size_t i = 0; i < numDependencies; i++) {
    requireDefaultEdgeData(edgeData[i]);
  }

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Queries immediately after hipStreamBeginCapture, before any work is captured, so the
 * dependency frontier is empty. Exercises the zero-length edge-data path.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipStreamGetCaptureInfo_v3_Positive_ZeroDependencies) {
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  hipGraph_t graph{nullptr};
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  const hipGraphNode_t* nodelist{};
  const hipGraphEdgeData* edgeData{};
  size_t numDependencies{1};

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamGetCaptureInfo_v3(stream, &captureStatus, nullptr, nullptr, &nodelist,
                                       &edgeData, &numDependencies));

  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(numDependencies == 0);
  // With a count of zero there is nothing to dereference in either array.

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Forks capture across two streams and rejoins, so the dependency frontier holds more than
 * one node. Exercises edge-data sizing for a multi-node frontier.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipStreamGetCaptureInfo_v3_Positive_MultipleDependencies) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(float);

  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<float> B_d(LinearAllocs::hipMalloc, Nbytes);

  StreamsGuard streams(2);
  EventsGuard events(2);

  hipGraph_t graph{nullptr};
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  const hipGraphNode_t* nodelist{};
  const hipGraphEdgeData* edgeData{};
  size_t numDependencies{};

  HIP_CHECK(hipStreamBeginCapture(streams[0], hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(A_d.ptr(), 0, Nbytes, streams[0]));

  // Fork: streams[1] joins the capture behind streams[0].
  HIP_CHECK(hipEventRecord(events[0], streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[1], events[0], 0));
  HIP_CHECK(hipMemsetAsync(B_d.ptr(), 0, Nbytes, streams[1]));

  // Join: the frontier of streams[0] now contains work from both branches.
  HIP_CHECK(hipEventRecord(events[1], streams[1]));
  HIP_CHECK(hipStreamWaitEvent(streams[0], events[1], 0));

  HIP_CHECK(hipStreamGetCaptureInfo_v3(streams[0], &captureStatus, nullptr, nullptr, &nodelist,
                                       &edgeData, &numDependencies));

  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(numDependencies >= 2);
  REQUIRE(nodelist != nullptr);
  REQUIRE(edgeData != nullptr);

  for (size_t i = 0; i < numDependencies; i++) {
    requireDefaultEdgeData(edgeData[i]);
  }

  HIP_CHECK(hipStreamEndCapture(streams[0], &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verifies that omitting edgeData_out still succeeds, i.e. v3 degrades to v2 behaviour.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipStreamGetCaptureInfo_v3_Positive_EdgeDataOmitted) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(float);

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);

  hipGraph_t graph{nullptr};
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  const hipGraphNode_t* nodelist{};
  size_t numDependencies{};

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(A_d.ptr(), 0, Nbytes, stream));

  // Dependencies without edge data must still succeed. CUDA reports hipErrorLossyQuery here only
  // when an edge carries a non-default annotation, which this implementation never produces.
  HIP_CHECK(hipStreamGetCaptureInfo_v3(stream, &captureStatus, nullptr, nullptr, &nodelist, nullptr,
                                       &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(numDependencies > 0);

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verifies the parameter validation that is specific to v3.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipStreamGetCaptureInfo_v3_Negative_Parameters) {
  hipGraph_t capInfoGraph{};
  unsigned long long capSequenceID;  // NOLINT
  size_t numDependencies;
  const hipGraphNode_t* nodelist{};
  const hipGraphEdgeData* edgeData{};

  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  SECTION("Capture Status location as nullptr") {
    HIP_CHECK_ERROR(hipStreamGetCaptureInfo_v3(stream, nullptr, &capSequenceID, &capInfoGraph,
                                               &nodelist, &edgeData, &numDependencies),
                    hipErrorInvalidValue);
  }

  SECTION("Edge data requested without dependencies") {
    hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
    HIP_CHECK_ERROR(hipStreamGetCaptureInfo_v3(stream, &captureStatus, &capSequenceID,
                                               &capInfoGraph, nullptr, &edgeData,
                                               &numDependencies),
                    hipErrorInvalidValue);
  }
}

/**
 * End doxygen group GraphTest.
 * @}
 */

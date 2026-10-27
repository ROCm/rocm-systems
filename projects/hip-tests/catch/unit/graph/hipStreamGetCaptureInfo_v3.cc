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
// The graph implementation records only default edge annotations, so every entry returned must be
// fully zero-initialized. CUDA defines that value as a full serialization of the two nodes with
// memory visibility, which is what this implementation does for every edge.
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
 *    - Verifies that an idle stream reports no capture and that edge data is left untouched.
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
  const hipGraphNode_t* nodelist{reinterpret_cast<const hipGraphNode_t*>(0x1)};
  const hipGraphEdgeData* edgeData{reinterpret_cast<const hipGraphEdgeData*>(0x1)};
  size_t numDependencies{1};

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
 *    - Queries the same active capture through v2 and v3 and verifies they report identical
 * capture state and dependency lists, i.e. that v3 only adds edge data.
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamGetCaptureInfo_v3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipStreamGetCaptureInfo_v3_Positive_MatchesV2) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(float);

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);

  hipGraph_t graph{nullptr};
  hipGraph_t capInfoGraph_v2{nullptr};
  hipGraph_t capInfoGraph_v3{nullptr};
  hipStreamCaptureStatus status_v2{hipStreamCaptureStatusNone};
  hipStreamCaptureStatus status_v3{hipStreamCaptureStatusNone};
  unsigned long long id_v2{}, id_v3{};  // NOLINT
  const hipGraphNode_t* deps_v2{};
  const hipGraphNode_t* deps_v3{};
  const hipGraphEdgeData* edgeData{};
  size_t count_v2{}, count_v3{};

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(A_d.ptr(), 0, Nbytes, stream));

  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &status_v2, &id_v2, &capInfoGraph_v2, &deps_v2,
                                       &count_v2));
  HIP_CHECK(hipStreamGetCaptureInfo_v3(stream, &status_v3, &id_v3, &capInfoGraph_v3, &deps_v3,
                                       &edgeData, &count_v3));

  REQUIRE(status_v2 == status_v3);
  REQUIRE(id_v2 == id_v3);
  REQUIRE(capInfoGraph_v2 == capInfoGraph_v3);
  REQUIRE(count_v2 == count_v3);

  for (size_t i = 0; i < count_v3; i++) {
    REQUIRE(deps_v2[i] == deps_v3[i]);
    requireDefaultEdgeData(edgeData[i]);
  }

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
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

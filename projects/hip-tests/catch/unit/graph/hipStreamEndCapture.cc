/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include "stream_capture_common.hh"

static size_t captureN() { return isQuickLevel() ? 10000 : 1000000; }

/**
 * @addtogroup hipStreamEndCapture hipStreamEndCapture
 * @{
 * @ingroup GraphTest
 * `hipStreamEndCapture(hipStream_t stream, hipGraph_t *pGraph)` -
 * ends capture on a stream, returning the captured graph
 */

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# End capture on legacy/null stream
 *        -# End capture when graph is nullptr
 *        -# End capture on stream where capture has not yet started
 *        -# Destroy stream and try to end capture
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamEndCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipStreamEndCapture_Negative_Parameters) {
  hipGraph_t graph{nullptr};
  const auto stream_type = GENERATE(Streams::perThread, Streams::created);
  StreamGuard stream_guard(stream_type);
  hipStream_t stream = stream_guard.stream();

  SECTION("Pass stream as nullptr") {
    HIP_CHECK_ERROR(hipStreamEndCapture(nullptr, &graph), hipErrorIllegalState);
  }
#if HT_NVIDIA
  SECTION("Pass graph as nullptr") {
    HIP_CHECK_ERROR(hipStreamEndCapture(stream, nullptr), hipErrorIllegalState);
  }
#endif
  SECTION("End capture on stream where capture has not yet started") {
    HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph), hipErrorIllegalState);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify no error occurs when graph is destroyed before capture
 * ends
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamEndCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipStreamEndCapture_Positive_GraphDestroy) {
  hipGraph_t graph{nullptr};
  const size_t N = captureN();
  size_t Nbytes = N * sizeof(float);

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = hipStreamCaptureModeGlobal;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), N, stream);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
}

static void thread_func_neg(hipStream_t stream, hipGraph_t graph) {
  HIP_ASSERT(hipErrorStreamCaptureWrongThread == hipStreamEndCapture(stream, &graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify that when capture is initiated on a thread with mode
 * other than hipStreamCaptureModeRelaxed and try to end capture from different
 * thread, it is expected to return hipErrorStreamCaptureWrongThread
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamEndCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipStreamEndCapture_Negative_Thread) {
  const size_t N = captureN();
  size_t Nbytes = N * sizeof(float);

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);

  hipGraph_t graph{nullptr};
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = hipStreamCaptureModeGlobal;

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), N, stream);

  std::thread t(thread_func_neg, stream, graph);
  t.join();

#if HT_AMD
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
#endif

  HIP_CHECK(hipGraphDestroy(graph));
}

static void thread_func_pos(hipStream_t stream, hipGraph_t* graph) {
  HIP_CHECK(hipStreamEndCapture(stream, graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify that when capture is initiated on a thread with
 * hipStreamCaptureModeRelaxed mode, end capture in a different thread is
 * successful
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamEndCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipStreamEndCapture_Positive_Thread) {
  const size_t N = captureN();
  size_t Nbytes = N * sizeof(float);

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = hipStreamCaptureModeRelaxed;

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), N, stream);

  std::thread t(thread_func_pos, stream, &graph);
  t.join();
  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), N, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i), N);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify that after hipStreamEndCapture fails on an invalidated
 * stream, the capture state is reset to None (not left as Invalidated), and
 * subsequent stream operations succeed:
 *        -# Begin capture on a stream
 *        -# Perform an illegal hipStreamSynchronize during capture (invalidates stream)
 *        -# Verify capture state is Invalidated
 *        -# Call hipStreamEndCapture (expect hipErrorStreamCaptureInvalidated)
 *        -# Verify hipGetLastError returns the error once then clears
 *        -# Verify capture state is now None twice in a row (not still Invalidated)
 *        -# Verify hipStreamSynchronize now succeeds (stream is no longer capturing)
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamEndCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipStreamEndCapture_Negative_InvalidatedStateReset) {
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();
  hipGraph_t graph{nullptr};
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};

  // Begin capture
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, nullptr));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);

  // Illegal sync during capture — invalidates the stream
  HIP_CHECK_ERROR(hipStreamSynchronize(stream), hipErrorStreamCaptureUnsupported);
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, nullptr));
  REQUIRE(captureStatus == hipStreamCaptureStatusInvalidated);

  // EndCapture on invalidated stream must fail with the appropriate error
  HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph), hipErrorStreamCaptureInvalidated);
  REQUIRE(graph == nullptr);

  // GetLastError must return the error once, then clear it
  REQUIRE(hipGetLastError() == hipErrorStreamCaptureInvalidated);
  REQUIRE(hipGetLastError() == hipSuccess);

  // Querying capture state twice must both return None (not Invalidated).
  // Without the fix, the state stays Invalidated after the failed EndCapture.
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, nullptr));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, nullptr));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);

  // Stream must be usable again — sync should succeed
  HIP_CHECK(hipStreamSynchronize(stream));
}

/**
 * End doxygen group GraphTest.
 * @}
 */

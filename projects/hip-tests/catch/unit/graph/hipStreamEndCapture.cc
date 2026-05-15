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

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include "stream_capture_common.hh"

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
TEST_CASE("Unit_hipStreamEndCapture_Negative_Parameters") {
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
TEST_CASE("Unit_hipStreamEndCapture_Positive_GraphDestroy") {
  hipGraph_t graph{nullptr};
  constexpr size_t N = 1000000;
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
TEST_CASE("Unit_hipStreamEndCapture_Negative_Thread") {
  constexpr size_t N = 1000000;
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
TEST_CASE("Unit_hipStreamEndCapture_Positive_Thread") {
  constexpr size_t N = 1000000;
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
 * End doxygen group GraphTest.
 * @}
 */

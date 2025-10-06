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
#if HT_AMD
  SECTION("Destroy stream and try to end capture") {
    hipStream_t destroyed_stream;
    HIP_CHECK(hipStreamCreate(&destroyed_stream));
    HIP_CHECK(
        hipStreamBeginCapture(destroyed_stream, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipStreamDestroy(destroyed_stream));
    HIP_CHECK_ERROR(hipStreamEndCapture(destroyed_stream, &graph),
                    hipErrorContextIsDestroyed);
  }
#endif
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
}

static void thread_func_neg(hipStream_t stream, hipGraph_t graph) {
  HIP_ASSERT(hipErrorStreamCaptureWrongThread ==
             hipStreamEndCapture(stream, &graph));
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
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), N, stream);

  std::thread t(thread_func_neg, stream, graph);
  t.join();

#if HT_AMD
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
#endif

  HIP_CHECK(hipGraphDestroy(graph));
}

static void thread_func_pos(hipStream_t stream, hipGraph_t *graph) {
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
 * Test Description
 * ------------------------
 *    - Test to verify below scenario.
 * 1) Enable capture on stream s1
 * 2) Record event on stream s1
 * 3) wait event on another stream s2, stream s2 also becomes capture stream
 * 4) Enqueue work on capture stream s1
 * 5) Enqueue work on capture stream s2
 * 6) End capture on stream s1, returns hipErrorStreamCaptureUnjoined
 * 7) event wait on stream1. (s2 joined back to s1)
 * 8) End capture on stream s2, returns hipErrorStreamCaptureUnmatched
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamEndCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipStreamEndCapture_cross_dependencies") {
  size_t N = 10;
  hipGraph_t graph{nullptr};
  size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  // Stream and event create
  StreamsGuard streams(2);
  EventsGuard events(2);
  hipStreamCaptureMode mode =
      GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
               hipStreamCaptureModeRelaxed);
  HIP_CHECK(hipStreamBeginCapture(streams[0], mode));
  HIP_CHECK(hipEventRecord(events[0], streams[0]));
  HIP_CHECK(hipStreamWaitEvent(streams[1], events[0], 0));
  HIP_CHECK(
      hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, streams[0]));

  HIP_CHECK(
      hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, streams[1]));

  SECTION("hipErrorStreamCaptureUnjoined") {
    HIP_CHECK_ERROR(hipStreamEndCapture(streams[0], &graph),
                    hipErrorStreamCaptureUnjoined);
  }
  SECTION("hipErrorStreamCaptureUnmatched") {
    HIP_CHECK(hipEventRecord(events[1], streams[1]));
    HIP_CHECK(hipStreamWaitEvent(streams[0], events[1], 0));
    HIP_CHECK_ERROR(hipStreamEndCapture(streams[1], &graph),
                    hipErrorStreamCaptureUnmatched);
  }
}

static void thread_func_begin(hipStream_t stream) {
  constexpr size_t N = 10;
  size_t Nbytes = N * sizeof(float);

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);

  hipStreamCaptureMode captureMode =
      GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal);
  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), N, stream);
}

static void thread_func_end(hipStream_t stream, hipGraph_t *graph) {
#if HT_AMD
  HIP_CHECK_ERROR(hipStreamEndCapture(stream, graph),
                  hipErrorStreamCaptureWrongThread);
#endif
}
/**
 * Test Description
 * ------------------------
 *    - Test to verify bewlow scenario.
 * 1) Enable capture on stream s1 from thread1 with mode {Global, Threadlocal}
 * 2) Enqueue work on capture stream s1
 * 3) End capture on s1 from thread2, returns hipErrorStreamCaptureWrongThread
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipStreamEndCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipStreamEndCapture_cros_thread") {
  hipGraph_t graph{nullptr};
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();
  std::thread t1(thread_func_begin, stream);
  t1.join();
  std::thread t2(thread_func_end, stream, &graph);
  t2.join();
}

/**
 * End doxygen group GraphTest.
 * @}
 */

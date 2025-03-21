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
 * @addtogroup hipLaunchHostFunc hipLaunchHostFunc
 * @{
 * @ingroup GraphTest
 * `hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void *userData)` -
 * enqueues a host function call in a stream
 */

static void hostNodeCallbackDummy(void* data) { REQUIRE(data == nullptr); }

static void hostNodeCallback(void* data) {
  float** userData = static_cast<float**>(data);

  float input_data = *(userData[0]);
  float output_data = *(userData[1]);
  REQUIRE(input_data == output_data);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Stream is legacy/nullptr stream
 *        -# Function is nullptr
 *        -# Stream is uninitialized
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipLaunchHostFunc.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipLaunchHostFunc_Negative_Parameters") {
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
#if HT_NVIDIA  // EXSWHTEC-228
  SECTION("Pass stream as nullptr") {
    hipHostFn_t fn = hostNodeCallbackDummy;
    HIP_CHECK_ERROR(hipLaunchHostFunc(nullptr, fn, nullptr), hipErrorStreamCaptureImplicit);
  }
#endif
  SECTION("Pass functions as nullptr") {
    HIP_CHECK_ERROR(hipLaunchHostFunc(stream, nullptr, nullptr), hipErrorInvalidValue);
  }
#if HT_AMD
  SECTION("Pass uninitialized stream") {
    hipHostFn_t fn = hostNodeCallbackDummy;
    constexpr auto InvalidStream = [] {
      StreamGuard sg(Streams::created);
      return sg.stream();
    };
    HIP_CHECK_ERROR(hipLaunchHostFunc(InvalidStream(), fn, nullptr), hipErrorContextIsDestroyed);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify enquing a host function into a stream, which checks if
 * the captured computation result is correct
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipLaunchHostFunc.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipLaunchHostFunc_Positive_Functional") {
  LinearAllocGuard<float> A_h(LinearAllocs::malloc, sizeof(float));
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, sizeof(float));
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, sizeof(float));

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = hipStreamCaptureModeGlobal;

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), 1, stream);

  hipHostFn_t fn = hostNodeCallback;
  float* data[2] = {A_h.host_ptr(), B_h.host_ptr()};
  HIP_CHECK(hipLaunchHostFunc(stream, fn, static_cast<void*>(data)));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), 1, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i), 1);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

static void thread_func_pos(hipStream_t* stream, hipHostFn_t fn, float** data){

    HIP_CHECK(hipLaunchHostFunc(*stream, fn, static_cast<void*>(data)))}

/**
 * Test Description
 * ------------------------
 *    - Test to verify enquing a host function into a stream on a different
 * thread, which checks if the captured computation result is correct
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipLaunchHostFunc.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipLaunchHostFunc_Positive_Thread") {
  LinearAllocGuard<float> A_h(LinearAllocs::malloc, sizeof(float));
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, sizeof(float));
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, sizeof(float));

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureMode = hipStreamCaptureModeGlobal;

  HIP_CHECK(hipStreamBeginCapture(stream, captureMode));
  captureSequenceSimple(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), 1, stream);

  hipHostFn_t fn = hostNodeCallback;
  float* data[2] = {A_h.host_ptr(), B_h.host_ptr()};
  std::thread t(thread_func_pos, &stream, fn, data);
  t.join();

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Replay the recorded sequence multiple times
  for (size_t i = 0; i < kLaunchIters; i++) {
    std::fill_n(A_h.host_ptr(), 1, static_cast<float>(i));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    ArrayFindIfNot(B_h.host_ptr(), static_cast<float>(i), 1);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
* End doxygen group GraphTest.
* @}
*/

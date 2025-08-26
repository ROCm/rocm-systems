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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include "stream_capture_common.hh"


/**
 * @addtogroup hipThreadExchangeStreamCaptureMode
 * hipThreadExchangeStreamCaptureMode
 * @{
 * @ingroup GraphTest
 * `hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode *mode)` -
 * swaps the stream capture mode of a thread
 */

/* Local Function for swaping stream capture mode of a thread
 */
static void hipGraphLaunchWithMode(hipStream_t stream, hipStreamCaptureMode mode) {
  constexpr size_t N = 1024;
  size_t Nbytes = N * sizeof(float);
  constexpr float fill_value = 5.0f;

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};

  LinearAllocGuard<float> A_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> B_h(LinearAllocs::malloc, Nbytes);
  LinearAllocGuard<float> A_d(LinearAllocs::hipMalloc, Nbytes);
  LinearAllocGuard<float> B_d(LinearAllocs::hipMalloc, Nbytes);
  float* C_d;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&mode));

  HIP_CHECK(hipStreamBeginCapture(stream, mode));

  captureSequenceLinear(A_h.host_ptr(), A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream);
  captureSequenceCompute(A_d.ptr(), B_h.host_ptr(), B_d.ptr(), N, stream);

  if (mode == hipStreamCaptureModeRelaxed) {
    HIP_CHECK(hipMalloc(&C_d, Nbytes));
  }

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  std::fill_n(A_h.host_ptr(), N, fill_value);
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Validate the computation
  ArrayFindIfNot(B_h.host_ptr(), fill_value * fill_value, N);
  if (mode == hipStreamCaptureModeRelaxed) {
    HIP_CHECK(hipFree(C_d));
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

void threadFuncCaptureMode(hipStream_t stream, hipStreamCaptureMode mode) {
  hipGraphLaunchWithMode(stream, mode);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify basic functionality for API that swaps the stream capture
 * mode of a thread. All combinations for main and other thread capture modes
 * are tested
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipThreadExchangeStreamCaptureMode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_Positive_Functional") {
  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  const hipStreamCaptureMode captureModeMain = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);
  const hipStreamCaptureMode captureModeThread = GENERATE(
      hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);

  hipGraphLaunchWithMode(stream, captureModeMain);
  std::thread t(threadFuncCaptureMode, stream, captureModeThread);
  t.join();
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Mode as nullptr
 *        -# Mode as -1
 *        -# Mode as INT_MAX
 *        -# Mode other than existing 3 modes (hipStreamCaptureModeRelaxed + 1)
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipThreadExchangeStreamCaptureMode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.3
 */
#if HT_AMD  // getting error in Cuda Setup
TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_Negative_Parameters") {
  hipStreamCaptureMode mode;

  SECTION("Pass Mode as nullptr") {
    HIP_CHECK_ERROR(hipThreadExchangeStreamCaptureMode(nullptr), hipErrorInvalidValue);
  }
  SECTION("Pass Mode as -1") {
    mode = hipStreamCaptureMode(-1);
    HIP_CHECK_ERROR(hipThreadExchangeStreamCaptureMode(&mode), hipErrorInvalidValue);
  }
  SECTION("Pass Mode as INT_MAX") {
    mode = hipStreamCaptureMode(INT_MAX);
    HIP_CHECK_ERROR(hipThreadExchangeStreamCaptureMode(&mode), hipErrorInvalidValue);
  }
  SECTION("Pass Mode as hipStreamCaptureModeRelaxed + 1") {
    mode = hipStreamCaptureMode(hipStreamCaptureModeRelaxed + 1);
    HIP_CHECK_ERROR(hipThreadExchangeStreamCaptureMode(&mode), hipErrorInvalidValue);
  }
}
#endif

/**
 * End doxygen group GraphTest.
 * @}
 */

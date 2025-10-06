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
 * Kernel to add one for each element in array
 */
__global__ void addOneKernel(int *a, int size) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for ( int i = offset; i < size; i+=stride ) {
    a[i] += 1;
  }
}

TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_StreamCapture_1") {

  std::cout << "====================================================== " << std::endl;

  constexpr int N = 40;
  constexpr int Nbytes = N * sizeof(int);

  std::vector<int> hostMem(N);
  std::fill(hostMem.begin(), hostMem.end(), 5);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamCaptureMode modeAtStart = GENERATE(
                                     hipStreamCaptureModeGlobal,
                                     hipStreamCaptureModeThreadLocal,
                                     hipStreamCaptureModeRelaxed);
  std::cout << "modeAtStart = " << modeAtStart << std::endl;

  HIP_CHECK(hipStreamBeginCapture(stream, modeAtStart));

  hipStreamCaptureMode exchangeMode = GENERATE(
                                      hipStreamCaptureModeGlobal,
                                      hipStreamCaptureModeThreadLocal,
                                      hipStreamCaptureModeRelaxed);

  std::cout << "BEFORE : exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));
  
  std::cout << "AFTER s1 : exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));

  std::cout << "AFTER s2 : exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipMemcpyAsync(devMem, hostMem.data(), Nbytes,
                           hipMemcpyHostToDevice, stream));
  addOneKernel<<< 1, 1, 0, stream >>>(devMem , N);
  HIP_CHECK(hipMemcpyAsync(hostMem.data(), devMem, Nbytes,
                           hipMemcpyDeviceToHost, stream));

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < N; i++) {
    if (hostMem[i] != 6) {
      std::cout << "At index : " << i << " Got value : " << hostMem[i]
                << " Expected value : 6 \n"
                << std::endl;
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));

}

TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_StreamCapture_2") {

  std::cout << "====================================================== " << std::endl;

  constexpr int N = 40;
  constexpr int Nbytes = N * sizeof(int);

  std::vector<int> hostMem(N);
  std::fill(hostMem.begin(), hostMem.end(), 5);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamCaptureMode modeAtStart = hipStreamCaptureModeRelaxed;

  std::cout << "modeAtStart = " << modeAtStart << std::endl;

  HIP_CHECK(hipStreamBeginCapture(stream, modeAtStart));

  hipStreamCaptureMode exchangeMode = hipStreamCaptureModeGlobal;

  std::cout << "BEFORE : exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));
  
  std::cout << "AFTER s1 : exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));

  std::cout << "AFTER s2 : exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipMemcpyAsync(devMem, hostMem.data(), Nbytes,
                           hipMemcpyHostToDevice, stream));
  addOneKernel<<< 1, 1, 0, stream >>>(devMem , N);
  HIP_CHECK(hipMemcpyAsync(hostMem.data(), devMem, Nbytes,
                           hipMemcpyDeviceToHost, stream));

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < N; i++) {
    if (hostMem[i] != 6) {
      std::cout << "At index : " << i << " Got value : " << hostMem[i]
                << " Expected value : 6 \n"
                << std::endl;
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem));

}

TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_StreamCapture_3") {

  std::cout << "====================================================== " << std::endl;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamCaptureMode *modeAtStart = new hipStreamCaptureMode;
  *modeAtStart = hipStreamCaptureModeRelaxed;

  std::cout << "modeAtStart = " << *modeAtStart << std::endl;

  HIP_CHECK(hipStreamBeginCapture(stream, *modeAtStart));

  hipStreamCaptureMode *exchangeMode = new hipStreamCaptureMode;
  *exchangeMode = hipStreamCaptureModeGlobal;

  std::cout << "BEFORE : exchangeMode = " << *exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(exchangeMode));
  
  std::cout << "AFTER s1 : exchangeMode = " << *exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(exchangeMode));

  std::cout << "AFTER s2 : exchangeMode = " << *exchangeMode << std::endl;

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));

}

TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_StreamCapture_G_R_G_TL") {

  std::cout << "====================================================== " << std::endl;

  constexpr int N = 40;
  constexpr int Nbytes = N * sizeof(int);

  //h-> D -> D -> Kernel-> h
  //G   G    R    G        G 

  std::vector<int> hostMem(N);
  std::fill(hostMem.begin(), hostMem.end(), 5);

  int* devMem1 = nullptr;
  HIP_CHECK(hipMalloc(&devMem1, Nbytes));
  REQUIRE(devMem1 != nullptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamCaptureMode modeAtStart = hipStreamCaptureModeGlobal;

  std::cout << "modeAtStart = " << modeAtStart << std::endl;

  HIP_CHECK(hipStreamBeginCapture(stream, modeAtStart));

  HIP_CHECK(hipMemcpyAsync(devMem1, hostMem.data(), Nbytes,
                           hipMemcpyHostToDevice, stream));
						   
  hipStreamCaptureMode exchangeMode = hipStreamCaptureModeRelaxed;

  std::cout << "exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));

  int* devMem2 = nullptr;
  HIP_CHECK(hipMalloc(&devMem2, Nbytes));
  REQUIRE(devMem2 != nullptr);
  
  HIP_CHECK(hipMemcpyAsync(devMem2, devMem1, Nbytes,
                           hipMemcpyDeviceToDevice, stream));

  exchangeMode = hipStreamCaptureModeGlobal;

  std::cout << "exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));  

  addOneKernel<<< 1, 1, 0, stream >>>(devMem2 , N);
  
  exchangeMode = hipStreamCaptureModeThreadLocal;

  std::cout << "exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));  
  
  HIP_CHECK(hipMemcpyAsync(hostMem.data(), devMem2, Nbytes,
                           hipMemcpyDeviceToHost, stream));

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < N; i++) {
    if (hostMem[i] != 6) {
      std::cout << "At index : " << i << " Got value : " << hostMem[i]
                << " Expected value : 6 \n"
                << std::endl;
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem1));
  HIP_CHECK(hipFree(devMem2));
}

TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_StreamCapture_R_G_R_TL") {

  std::cout << "====================================================== " << std::endl;

  constexpr int N = 40;
  constexpr int Nbytes = N * sizeof(int);

  std::vector<int> hostMem(N);
  std::fill(hostMem.begin(), hostMem.end(), 5);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamCaptureMode modeAtStart = hipStreamCaptureModeRelaxed;

  std::cout << "modeAtStart = " << modeAtStart << std::endl;

  HIP_CHECK(hipStreamBeginCapture(stream, modeAtStart));

  int* devMem1 = nullptr;
  HIP_CHECK(hipMalloc(&devMem1, Nbytes));
  REQUIRE(devMem1 != nullptr);

  hipStreamCaptureMode exchangeMode = hipStreamCaptureModeGlobal;

  std::cout << "exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));

  HIP_CHECK(hipMemcpyAsync(devMem1, hostMem.data(), Nbytes,
                           hipMemcpyHostToDevice, stream));

 exchangeMode = hipStreamCaptureModeRelaxed;

  std::cout << "exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));

  int* devMem2 = nullptr;
  HIP_CHECK(hipMalloc(&devMem2, Nbytes));
  REQUIRE(devMem2 != nullptr);
  
  HIP_CHECK(hipMemcpyAsync(devMem2, devMem1, Nbytes,
                           hipMemcpyDeviceToDevice, stream));

  exchangeMode = hipStreamCaptureModeThreadLocal;

  std::cout << "exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));  

  addOneKernel<<< 1, 1, 0, stream >>>(devMem2 , N);  
  HIP_CHECK(hipMemcpyAsync(hostMem.data(), devMem2, Nbytes,
                           hipMemcpyDeviceToHost, stream));

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < N; i++) {
    if (hostMem[i] != 6) {
      std::cout << "At index : " << i << " Got value : " << hostMem[i]
                << " Expected value : 6 \n"
                << std::endl;
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem1));
  HIP_CHECK(hipFree(devMem2));
}

TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_StreamCapture_TL_R_TL_G") {

  std::cout << "====================================================== " << std::endl;

  constexpr int N = 40;
  constexpr int Nbytes = N * sizeof(int);

  //h-> D -> D -> Kernel-> h
  //G   G    R    G        G 

  std::vector<int> hostMem(N);
  std::fill(hostMem.begin(), hostMem.end(), 5);

  int* devMem1 = nullptr;
  HIP_CHECK(hipMalloc(&devMem1, Nbytes));
  REQUIRE(devMem1 != nullptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamCaptureMode modeAtStart = hipStreamCaptureModeThreadLocal;

  std::cout << "modeAtStart = " << modeAtStart << std::endl;

  HIP_CHECK(hipStreamBeginCapture(stream, modeAtStart));

  HIP_CHECK(hipMemcpyAsync(devMem1, hostMem.data(), Nbytes,
                           hipMemcpyHostToDevice, stream));
						   
  hipStreamCaptureMode exchangeMode = hipStreamCaptureModeRelaxed;

  std::cout << "exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));

  int* devMem2 = nullptr;
  HIP_CHECK(hipMalloc(&devMem2, Nbytes));
  REQUIRE(devMem2 != nullptr);
  
  HIP_CHECK(hipMemcpyAsync(devMem2, devMem1, Nbytes,
                           hipMemcpyDeviceToDevice, stream));

  exchangeMode = hipStreamCaptureModeThreadLocal;

  std::cout << "exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));  

  addOneKernel<<< 1, 1, 0, stream >>>(devMem2 , N);
  
  exchangeMode = hipStreamCaptureModeGlobal;

  std::cout << "exchangeMode = " << exchangeMode << std::endl;

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&exchangeMode));  
  
  HIP_CHECK(hipMemcpyAsync(hostMem.data(), devMem2, Nbytes,
                           hipMemcpyDeviceToHost, stream));

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < N; i++) {
    if (hostMem[i] != 6) {
      std::cout << "At index : " << i << " Got value : " << hostMem[i]
                << " Expected value : 6 \n"
                << std::endl;
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devMem1));
  HIP_CHECK(hipFree(devMem2));
}

/**
* End doxygen group GraphTest.
* @}
*/

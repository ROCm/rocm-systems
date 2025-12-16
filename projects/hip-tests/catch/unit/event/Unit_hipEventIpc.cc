/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

// Test hipEventIpc behavior.

#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include <hip_test_common.hh>

/**
 * @addtogroup hipEventCreateWithFlags hipEventCreateWithFlags
 * @{
 * @ingroup EventTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validate Event Management APIs when working with multiple processes.
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventIpc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipEventIpc") {
  size_t N = 4 * 1024 * 1024;
  unsigned threadsPerBlock = 256;
  int iterations = 1;

  unsigned blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  if (blocks > 1024) blocks = 1024;
  if (blocks == 0) blocks = 1;

  printf("N=%zu (A+B+C= %6.1f MB total) blocks=%u threadsPerBlock=%u iterations=%d\n", N,
         ((double)3 * N * sizeof(float)) / 1024 / 1024, blocks, threadsPerBlock, iterations);
  printf("iterations=%d\n", iterations);

  size_t Nbytes = N * sizeof(float);

  float *A_h, *B_h, *C_h;
  float *A_d, *B_d, *C_d;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  hipEvent_t start, stop;

  // NULL stream check:
  HIP_CHECK(hipEventCreateWithFlags(&start, hipEventDisableTiming | hipEventInterprocess));
  HIP_CHECK(hipEventCreateWithFlags(&stop, hipEventDisableTiming | hipEventInterprocess));

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));


  for (int i = 0; i < iterations; i++) {
    //--- START TIMED REGION
    long long hostStart = HipTest::get_time();
    // Record the start event
    HIP_CHECK(hipEventRecord(start, NULL));

    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       static_cast<const float*>(A_d), static_cast<const float*>(B_d), C_d, N);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventQuery(stop));
    long long hostStop = HipTest::get_time();
    //--- STOP TIMED REGION


    float eventMs = 1.0f;
    // should fail due to hipEventDisableTiming
    REQUIRE(hipSuccess != hipEventElapsedTime(&eventMs, start, stop));
    float hostMs = HipTest::elapsed_time(hostStart, hostStop);

    printf("host_time (chrono)                =%6.3fms\n", hostMs);
    printf("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);
    printf("\n");
  }

  hipIpcEventHandle_t ipc_handle;
  HIP_CHECK(hipIpcGetEventHandle(&ipc_handle, start));

  hipEvent_t ipc_event;
  hipError_t err = hipIpcOpenEventHandle(&ipc_event, ipc_handle);

#if HT_WIN
  // always different process Id on Windows
  HIP_CHECK(err);
#else
  // hipIpcOpenEventHandle() should be called in a different process, hence it should fail here
  REQUIRE(err == hipErrorInvalidContext);
#endif
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
#if HT_WIN
  HIP_CHECK(hipEventDestroy(ipc_event));
#endif
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  HipTest::checkVectorADD(A_h, B_h, C_h, N, true);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}
/**
 * Test Description
 * ------------------------
 *  -This test will validate the basic functionality of hipIpcGetEventHandle
 *  and hipIpcOpenEventHandle apis using stream capture apis in
 *  hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
 *  hipStreamCaptureModeRelaxed mode.
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventIpc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */

TEST_CASE("Unit_hipIpc_EventHandle_Stream_hipStreamCaptureModeThreadLocal") {
  int fd[2];
  REQUIRE(pipe(fd) == 0);
  hipStreamCaptureMode mode = GENERATE(
        hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);
  auto pid = fork();

  if (pid != 0) {  // parent process
  // Validating hipIpcGetEventHandle API
    hipEvent_t start = nullptr;
    HIP_CHECK(hipEventCreateWithFlags(&start, hipEventInterprocess |
                                                  hipEventDisableTiming));
    REQUIRE(start != nullptr);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamBeginCapture(stream, mode));

    hipIpcEventHandle_t handle;
    HIP_CHECK(hipIpcGetEventHandle(&handle, start));
    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graph_exec, stream));
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    REQUIRE(write(fd[1], &handle, sizeof(hipIpcEventHandle_t)) >= 0);
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);

    HIP_CHECK(hipEventDestroy(start));
  } else {  // child process
    // Validating hipIpcOpenMemHandle API
    hipIpcEventHandle_t handle;
    REQUIRE(read(fd[0], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[0]) == 0);
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamBeginCapture(stream, mode));

    hipEvent_t start = nullptr;
    HIP_CHECK(hipIpcOpenEventHandle(&start, handle));
    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graph_exec, stream));
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));
    REQUIRE(start != nullptr);
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&stop));
    REQUIRE(stop != nullptr);

    int N = 40;
    int Nbytes = N * sizeof(int);

    int *hostMem = reinterpret_cast<int *>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    fillHostArray(hostMem, N, 10);

    int *devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);

    hipStream_t stream1;
    HIP_CHECK(hipStreamCreate(&stream1));

    HIP_CHECK(hipEventRecord(start, stream1));

    HIP_CHECK(hipMemcpyAsync(devMem, hostMem, Nbytes, hipMemcpyHostToDevice,
                             stream1));
    addOneKernel<<<1, 1>>>(devMem, N);
    HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost,
                             stream1));

    HIP_CHECK(hipEventRecord(stop, stream1));
    HIP_CHECK(hipEventSynchronize(stop));

    REQUIRE(validateHostArray(hostMem, N, 11) == true);

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(stream1));
    free(hostMem);
    HIP_CHECK(hipFree(devMem));
  }
}
/**
 * End doxygen group EventTest.
 * @}
 */

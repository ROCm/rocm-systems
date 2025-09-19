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

/**
Negative Testcase Scenarios :
1) Pass stream as nullptr and verify there is no crash, api returns error code.
2) Pass graph as nullptr and verify there is no crash, api returns error code.
3) Pass graph as nullptr and  and stream as hipStreamPerThread verify there
   is no crash, api returns error code.
4) End capture on stream where capture has not yet started and verify
   error code is returned.
5) Destroy stream and try to end capture.
6) Destroy Graph and try to end capture.
7) Begin capture on a thread with mode other than hipStreamCaptureModeRelaxed
   and try to end capture from different thread. Expect to return
   hipErrorStreamCaptureWrongThread.
8) Start stream capture on stream1 using mode hipStreamCaptureModeRelaxed.
   In stream1 queue a memcpy operation, queue a kernel square of a number operation.
   Launch a thread. In the thread, queue a memcpy operation. End the capture on
   stream1 and return the captured graph. Wait for the thread in main function.
   Create an executable graph and launch the graph on input data and validate the
   output.
9) Create 2 streams s1 and s2. Begin stream capture in s1, spawn a
   captured fork stream on s2. Queue some operations
   (like increment kernel) on both s1 and s2. End the stream capture
   on s2 and verify the error returned by the End capture.
10)Create 2 streams s1 and s2. Begin stream capture in s1 and spawn a captured
   fork stream s2. In main thread, queue a memcpy operation on s1.
   Launch a thread, queue a memcpy operation on s2. Perform hipEventRecord on
   s2 and wait Event on S1.  Wait for the thread to complete. Queue operations
   kernel addition(Cd = Ad + Bd) operation and memcpy(Ch <- Cd) in s1. End the
   stream capture in s1. Create an executable graph and launch the graph on input
   data and validate the output.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

TEST_CASE("Unit_hipStreamEndCapture_Negative") {
  hipError_t ret;
  SECTION("Pass stream as nullptr") {
    hipGraph_t graph;
    ret = hipStreamEndCapture(nullptr, &graph);
    REQUIRE(hipErrorIllegalState == ret);
  }
#if HT_NVIDIA
  SECTION("Pass graph as nullptr") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    ret = hipStreamEndCapture(stream, nullptr);
    REQUIRE(hipErrorIllegalState == ret);
    HIP_CHECK(hipStreamDestroy(stream));
  }
  SECTION("Pass graph as nullptr and stream as hipStreamPerThread") {
    ret = hipStreamEndCapture(hipStreamPerThread, nullptr);
    REQUIRE(hipErrorIllegalState == ret);
  }
#endif
  SECTION("End capture on stream where capture has not yet started") {
    hipStream_t stream;
    hipGraph_t graph;
    HIP_CHECK(hipStreamCreate(&stream));
    ret = hipStreamEndCapture(stream, &graph);
    REQUIRE(hipErrorIllegalState == ret);
    HIP_CHECK(hipStreamDestroy(stream));
  }
  SECTION("Destroy graph and try to end capture in between") {
    hipStream_t stream{nullptr};
    hipGraph_t graph{nullptr};
    constexpr unsigned blocks = 512;
    constexpr unsigned threadsPerBlock = 256;
    constexpr size_t N = 100000;
    size_t Nbytes = N * sizeof(float);
    float *A_d, *C_d;
    float *A_h, *C_h;

    A_h = reinterpret_cast<float*>(malloc(Nbytes));
    C_h = reinterpret_cast<float*>(malloc(Nbytes));
    REQUIRE(A_h != nullptr);
    REQUIRE(C_h != nullptr);

    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
      A_h[i] = 1.618f + i;
    }

    HIP_CHECK(hipMalloc(&A_d, Nbytes));
    HIP_CHECK(hipMalloc(&C_d, Nbytes));
    REQUIRE(A_d != nullptr);
    REQUIRE(C_d != nullptr);

    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

    HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                       C_d, N);
    HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

    HIP_CHECK(hipGraphDestroy(graph));
    ret = hipStreamEndCapture(stream, &graph);
    REQUIRE(hipSuccess == ret);

    free(A_h);
    free(C_h);
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK(hipFree(C_d));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

static void thread_func(hipStream_t stream, hipGraph_t graph) {
  HIP_ASSERT(hipErrorStreamCaptureWrongThread == hipStreamEndCapture(stream, &graph));
}
static void StreamEndCaptureThreadNegative(float* A_d, float* A_h, float* C_d, float* C_h,
                                           hipStreamCaptureMode mode) {
  hipStream_t stream{nullptr};
  hipGraph_t graph{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  constexpr size_t N = 100000;
  size_t Nbytes = N * sizeof(float);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                     C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  std::thread t(thread_func, stream, graph);
  t.join();

#if HT_AMD
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
#endif
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphDestroy(graph));
}
TEST_CASE("Unit_hipStreamEndCapture_Thread_Negative") {
  constexpr size_t N = 100000;
  size_t Nbytes = N * sizeof(float);
  float *A_d, *C_d;
  float *A_h, *C_h;

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
  }

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  SECTION("Capture Mode:hipStreamCaptureModeGlobal") {
    StreamEndCaptureThreadNegative(A_d, A_h, C_d, C_h, hipStreamCaptureModeGlobal);
  }
  SECTION("Capture Mode:hipStreamCaptureModeThreadLocal") {
    StreamEndCaptureThreadNegative(A_d, A_h, C_d, C_h, hipStreamCaptureModeThreadLocal);
  }
  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}
// Thread function
static void thread_func1(hipStream_t stream, hipGraph_t* graph, size_t Nbytes, float* A_d,
                         float* B_h) {
  HIP_CHECK(hipMemcpyAsync(B_h, A_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, graph));
}
/*
 * Start stream capture on stream1 using mode hipStreamCaptureModeRelaxed.
 * In stream1 queue a memcpy operation, queue a kernel square of a number operation.
 * Launch a thread. In the thread, queue a memcpy operation. End the capture on
 * stream1 and return the captured graph. Wait for the thread in main function.
 * Create an executable graph and launch the graph on input data and validate the output.
 * */
TEST_CASE("Unit_hipStreamEndCapture_mode_hipStreamCaptureModeRelaxed") {
  hipStream_t stream{nullptr}, streamForGraph{nullptr};
  hipGraph_t graph{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  constexpr size_t N = 10;
  size_t Nbytes = N * sizeof(float);
  // Device Pointers
  float* A_d;
  // Host Pointers
  float *A_h, *B_h, *C_h;

  // Memory allocation to Host pointers
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  B_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(B_h != nullptr);
  REQUIRE(C_h != nullptr);

  // Initialize the Host data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.0f + i;
    C_h[i] = A_h[i];
  }
  // Memory allocation to Device pointers
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));
  REQUIRE(A_d != nullptr);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
  // Copy data from Host to Device
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                     A_d, N);
  // Thread Launch
  std::thread t(thread_func1, stream, &graph, Nbytes, A_d, B_h);
  t.join();

  // Launch the graph
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Output verification
  for (size_t i = 0; i < N; i++) {
    C_h[i] = C_h[i] * C_h[i];
    REQUIRE(B_h[i] == C_h[i]);
  }

  free(A_h);
  free(B_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
}

static __global__ void increment(int* A_d) { atomicAdd(A_d, 1); }
/*
 * Create 2 streams s1 and s2. Begin stream capture in s1, spawn a
 * captured fork stream on s2. Queue some operations
 * (like increment kernel) on both s1 and s2. End the stream capture
 * on s2 and verify the error returned by the End capture.
 */
TEST_CASE("Unit_hipStreamEndCapture_chkError_on_wrongStream") {
  int *A_d{nullptr}, *A_h{nullptr};
  hipStream_t stream1{nullptr}, stream2{nullptr};
  hipEvent_t forkStreamEvent{nullptr};
  hipEvent_t joinStreamEvent{nullptr};
  hipGraph_t graph{nullptr};
  hipError_t err;
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  size_t Nbytes = sizeof(int);

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  HIP_CHECK(hipEventCreate(&joinStreamEvent));
  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  // Initialize the Host data
  *A_h = 0;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));
  REQUIRE(A_d != nullptr);

  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));

  hipLaunchKernelGGL(increment, dim3(blocks), dim3(threadsPerBlock), 0, stream1, A_d);
  hipLaunchKernelGGL(increment, dim3(blocks), dim3(threadsPerBlock), 0, stream2, A_d);

  err = hipStreamEndCapture(stream2, &graph);
  REQUIRE(err == hipErrorStreamCaptureUnmatched);
  HIP_CHECK(hipEventRecord(joinStreamEvent, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, joinStreamEvent, 0));
  err = hipStreamEndCapture(stream1, &graph);

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(joinStreamEvent));
  HIP_CHECK(hipGraphDestroy(graph));
  free(A_h);
  HIP_CHECK(hipFree(A_d));
}
static void thread_func4(hipStream_t stream1, hipStream_t stream2, hipEvent_t event, size_t Nbytes,
                         int* B_d, int* B_h) {
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream2));
  HIP_CHECK(hipEventRecord(event, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, event, 0));
}
/*
 * Create 2 streams s1 and s2. Begin stream capture in s1 and spawn a captured
 * fork stream s2. In main thread, queue a memcpy operation on s1.
 * Launch a thread, queue a memcpy operation on s2. Perform hipEventRecord on
 * s2 and wait Event on S1.  Wait for the thread to complete. Queue operations
 * kernel addition(Cd = Ad + Bd) operation and memcpy(Ch <- Cd) in s1. End the
 * stream capture in s1. Create an executable graph and launch the graph on input
 * data and validate the output.
 * */
TEST_CASE("Unit_hipStreamEndCapture_streamMerge_in_thread") {
  // Device Pointers
  int *A_d, *B_d, *C_d;
  // Host Pointers
  int *A_h, *B_h, *C_h, *D_h;
  hipStream_t stream1{nullptr}, stream2{nullptr}, streamForGraph{nullptr};
  hipEvent_t forkStreamEvent{nullptr}, event{nullptr};
  hipGraph_t graph{nullptr};

  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  constexpr size_t N = 5;
  size_t Nbytes = N * sizeof(int);

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  HIP_CHECK(hipEventCreate(&event));
  // Memory allocation to Host Pointers
  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  B_h = reinterpret_cast<int*>(malloc(Nbytes));
  C_h = reinterpret_cast<int*>(malloc(Nbytes));
  D_h = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(B_h != nullptr);
  REQUIRE(C_h != nullptr);
  REQUIRE(D_h != nullptr);
  // Initialize the Host data
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1 + i;
    B_h[i] = 2 + i;
    C_h[i] = 0;
    D_h[i] = 0;
  }
  // Memory allocation to Device Pointers
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&B_d), Nbytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&C_d), Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(B_d != nullptr);
  REQUIRE(C_d != nullptr);

  // Begin Capture
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));

  HIP_CHECK(hipEventRecord(forkStreamEvent, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  // Thread Launch
  std::thread t(thread_func4, stream1, stream2, event, Nbytes, B_d, B_h);
  t.join();
  // Launch kernal
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream1, A_d, B_d,
                     C_d, N);

  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));

  // Launch graph
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify Output
  for (size_t i = 0; i < N; i++) {
    D_h[i] = A_h[i] + B_h[i];
    REQUIRE(C_h[i] == D_h[i]);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(streamForGraph));

  // Release the memory
  free(A_h);
  free(B_h);
  free(C_h);
  free(D_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}

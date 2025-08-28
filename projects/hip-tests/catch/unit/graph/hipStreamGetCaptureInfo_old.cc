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

Testcase Scenarios
------------------
Functional:
1) Start stream capture and get capture info. Verify api is success, capture status is
hipStreamCaptureStatusActive and identifier returned is valid/non-zero. 2) End stream capture and
get capture info. Verify api is success, capture status is hipStreamCaptureStatusNone and identifier
is not returned/updated by api. 3) Begin capture on hipStreamPerThread and get capture info. Verify
api is success, capture status is hipStreamCaptureStatusActive and identifier returned is
valid/non-zero. 4) End capture on hipStreamPerThread, get capture info. Verify api is success,
capture status is hipStreamCaptureStatusNone and identifier is not returned/updated by api. 5)
Perform multiple captures and verify the identifier returned is unique.

Argument Validation/Negative:
1) Pass pId as nullptr and verify api doesn’t crash and returns success.
2) Pass pCaptureStatus as nullptr and verify api doesn’t crash and returns error code.

Extended Scenarios
------------------
1.Create 2 streams s1 and s2. Start capturing s1. Record event e1 on s1 and wait for event e1 on s2.
Queue some operations in s1 and s2. Invoke hipStreamGetCaptureInfo on both s1 and s2. Verify that
the capture info (status and id) of both s1 and s2 are identical. Record event e2 on s2 and wait for
event e2 on s1. End the capture of stream s1. Verify that the capture info (status and id) of both
s1 and s2 are identical.

2.Create a stream s1. Start capturing s1. Get the capture info of s1. Launch a thread. In the thread
get the capture info of s1 using hipStreamGetCaptureInfo. Verify that it is in state
hipStreamCaptureStatusActive and capture id inside thread is same as capture id in main function.
Exit the thread and end the capture

3.Verify that the id remains same througout the capture. Create a stream s1. Start capturing s1. Get
the capture info of s1. Queue some oprations in s1. Again get the capture info. Queue different
operations in s1. Again get the capture info. Verify that all the capture info are identical.

4.Create a stream with default flag (hipStreamDefault). Start capturing the stream. Invoke
hipStreamGetCaptureInfo() on the null stream. Verify hipErrorStreamCaptureImplicit is returned by
hipStreamGetCaptureInfo(). Verify capture status of created stream. Do some operatoins. End the
capture on the created stream.Verify the capture status. Execute the graph and verify the output
from the operations.

5. Test scenario 1 using hipStreamGetCaptureInfo_v2.
6. Test scenario 2 using hipStreamGetCaptureInfo_v2.
7. Test scenario 3 using hipStreamGetCaptureInfo_v2.
8. Test scenario 4 using hipStreamGetCaptureInfo_v2.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

constexpr size_t N = 1000000;
constexpr unsigned blocks = 512;
constexpr unsigned threadsPerBlock = 256;
size_t Nbytes = N * sizeof(float);
constexpr int LAUNCH_ITERS = 1;

/**
 * Validates stream capture info, launches graph and verify results
 */
void validateStreamCaptureInfo(hipStream_t mstream) {
  hipStream_t stream1{nullptr}, stream2{nullptr}, streamForLaunch{nullptr};
  hipEvent_t memsetEvent1, memsetEvent2, forkStreamEvent;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  float *A_d, *C_d;
  float *A_h, *C_h;
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);
  HIP_CHECK(hipStreamCreate(&streamForLaunch));

  // Initialize input buffer
  for (size_t i = 0; i < N; ++i) {
    A_h[i] = 3.146f + i;  // Pi
  }

  // Create cross stream dependencies.
  // memset operations are done on stream1 and stream2
  // and they are joined back to mainstream
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&memsetEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));

  HIP_CHECK(hipStreamBeginCapture(mstream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, mstream));
  HIP_CHECK(hipStreamWaitEvent(stream1, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipMemsetAsync(A_d, 0, Nbytes, stream1));
  HIP_CHECK(hipEventRecord(memsetEvent1, stream1));
  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream2));
  HIP_CHECK(hipEventRecord(memsetEvent2, stream2));
  HIP_CHECK(hipStreamWaitEvent(mstream, memsetEvent1, 0));
  HIP_CHECK(hipStreamWaitEvent(mstream, memsetEvent2, 0));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mstream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, mstream, A_d,
                     C_d, N);

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  unsigned long long capSequenceID = 0;  // NOLINT
  HIP_CHECK(hipStreamGetCaptureInfo(mstream, &captureStatus, &capSequenceID));

  // verify capture status is active and sequence id is valid
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capSequenceID > 0);

  // End capture and verify graph is returned
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mstream));
  HIP_CHECK(hipStreamEndCapture(mstream, &graph));
  REQUIRE(graph != nullptr);

  // verify capture status is inactive and sequence id is not updated
  capSequenceID = 0;
  HIP_CHECK(hipStreamGetCaptureInfo(mstream, &captureStatus, &capSequenceID));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
  REQUIRE(capSequenceID == 0);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence multiple times
  for (int i = 0; i < LAUNCH_ITERS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForLaunch));
  }

  HIP_CHECK(hipStreamSynchronize(streamForLaunch));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForLaunch));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(memsetEvent1));
  HIP_CHECK(hipEventDestroy(memsetEvent2));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      INFO("A and C not matching at " << i << " C_h[i] " << C_h[i] << " A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }
  free(A_h);
  free(C_h);
}

/**
 * Basic Functional Test for stream capture and getting capture info.
 * Regular/custom stream is used for stream capture.
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_BasicFunctional") {
  hipStream_t streamForCapture;

  HIP_CHECK(hipStreamCreate(&streamForCapture));
  validateStreamCaptureInfo(streamForCapture);
  HIP_CHECK(hipStreamDestroy(streamForCapture));
}

/**
 * Test performs stream capture on hipStreamPerThread and validates
 * capture info.
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_hipStreamPerThread") {
  validateStreamCaptureInfo(hipStreamPerThread);
}

/**
 * Test starts stream capture on multiple streams and verifies uniqueness of
 * identifiers returned.
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_UniqueID") {
  constexpr int numStreams = 100;
  hipStream_t streams[numStreams]{};
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  std::vector<int> idlist;
  unsigned long long capSequenceID{};  // NOLINT
  hipGraph_t graph{nullptr};

  for (int i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
    HIP_CHECK(hipStreamBeginCapture(streams[i], hipStreamCaptureModeGlobal));
    HIP_CHECK(hipStreamGetCaptureInfo(streams[i], &captureStatus, &capSequenceID));
    REQUIRE(captureStatus == hipStreamCaptureStatusActive);
    REQUIRE(capSequenceID > 0);
    idlist.push_back(capSequenceID);
  }

  for (int i = 0; i < numStreams; i++) {
    for (int j = i + 1; j < numStreams; j++) {
      if (idlist[i] == idlist[j]) {
        INFO("Same identifier returned for stream " << i << " and stream " << j);
        REQUIRE(false);
      }
    }
  }

  for (int i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamEndCapture(streams[i], &graph));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

/**
 * Argument validation/Negative tests for api
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_ArgValidation") {
  hipError_t ret;
  hipStream_t stream;
  hipStreamCaptureStatus captureStatus;
  unsigned long long capSequenceID;  // NOLINT
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("Capture ID location as nullptr") {
    ret = hipStreamGetCaptureInfo(stream, &captureStatus, nullptr);
    // Capture ID is optional
    REQUIRE(ret == hipSuccess);
  }

  SECTION("Capture Status location as nullptr") {
    ret = hipStreamGetCaptureInfo(stream, nullptr, &capSequenceID);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  HIP_CHECK(hipStreamDestroy(stream));
}
/*
 * Create 2 streams s1 and s2. Start capturing s1. Record event e1 on s1 and
 * wait for event e1 on s2. Queue some operations in s1 and s2. Invoke
 * hipStreamGetCaptureInfo on both s1 and s2. Verify that the capture info
 * (status and id) of both s1 and s2 are identical. Record event e2 on s2
 * and wait for event e2 on s1. End the capture of stream s1. Verify that the
 * capture info (status and id) of both s1 and s2 are identical.
 * The above scenario using hipStreamGetCaptureInfo_v2 API
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_ParentAndForkedStrm_CaptureStatus") {
  hipStream_t stream1{nullptr}, stream2{nullptr};
  hipEvent_t event2{nullptr}, forkStreamEvent{nullptr};
  hipGraph_t graph{nullptr};
  float *A_d, *B_d, *C_d, *D_d;
  float *A_h, *B_h, *C_h, *D_h;
  // Memory allocation to Host pointers
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  B_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  D_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(B_h != nullptr);
  REQUIRE(C_h != nullptr);
  REQUIRE(D_h != nullptr);
  // Memory allocation to Device pointers
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMalloc(&D_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(B_d != nullptr);
  REQUIRE(C_d != nullptr);
  REQUIRE(D_d != nullptr);
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&event2));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  // Start capture on stream1
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  // Copy data to Device
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream2));
  // Kernal Operations
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream1, A_d,
                     C_d, N);
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream2, B_d,
                     D_d, N);
  // Copy data back to the Host
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipMemcpyAsync(D_h, D_d, Nbytes, hipMemcpyDeviceToHost, stream2));

  hipStreamCaptureStatus captureStatus1{hipStreamCaptureStatusNone},
      captureStatus2{hipStreamCaptureStatusNone}, captureStatus3{hipStreamCaptureStatusNone},
      captureStatus4{hipStreamCaptureStatusNone};
  unsigned long long capSequenceID1, capSequenceID2, capSequenceID3,  // NOLINT
      capSequenceID4;
  SECTION("hipStreamGetCaptureInfo verification before End capture") {
    // Capture info
    HIP_CHECK(hipStreamGetCaptureInfo(stream1, &captureStatus1, &capSequenceID1));
    HIP_CHECK(hipStreamGetCaptureInfo(stream2, &captureStatus2, &capSequenceID2));
    // Verfication of results
    REQUIRE(capSequenceID1 == capSequenceID2);
    REQUIRE(captureStatus1 == hipStreamCaptureStatusActive);
    REQUIRE(captureStatus2 == hipStreamCaptureStatusActive);
  }
  SECTION("hipStreamGetCaptureInfo_v2 verification before End capture") {
    // Capture info
    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream1, &captureStatus1, &capSequenceID1, nullptr,
                                         nullptr, nullptr));
    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream2, &captureStatus2, &capSequenceID2, nullptr,
                                         nullptr, nullptr));
    // Verfication of results
    REQUIRE(capSequenceID1 == capSequenceID2);
    REQUIRE(captureStatus1 == hipStreamCaptureStatusActive);
    REQUIRE(captureStatus2 == hipStreamCaptureStatusActive);
  }


  HIP_CHECK(hipEventRecord(event2, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, event2, 0));
  // End the capture
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  REQUIRE(graph != nullptr);
  SECTION("hipStreamGetCaptureInfo verification after  End capture") {
    // Capture Info
    HIP_CHECK(hipStreamGetCaptureInfo(stream1, &captureStatus3, &capSequenceID3));
    HIP_CHECK(hipStreamGetCaptureInfo(stream2, &captureStatus4, &capSequenceID4));
    // Verification of results
    REQUIRE(captureStatus3 == hipStreamCaptureStatusNone);
    REQUIRE(captureStatus4 == hipStreamCaptureStatusNone);
  }
  SECTION("hipStreamGetCaptureInfo_v2 verification after  End capture") {
    // Capture Info
    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream1, &captureStatus3, &capSequenceID3, nullptr,
                                         nullptr, nullptr));
    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream2, &captureStatus4, &capSequenceID4, nullptr,
                                         nullptr, nullptr));
    // Verification of results
    REQUIRE(captureStatus3 == hipStreamCaptureStatusNone);
    REQUIRE(captureStatus4 == hipStreamCaptureStatusNone);
  }
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(D_d));
  free(A_h);
  free(B_h);
  free(C_h);
  free(D_h);
}
// Thread Function
static void thread_func(hipStream_t stream, unsigned long long capSequenceID1,  // NOLINT
                        unsigned long long capSequenceID2) {                    // NOLINT
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  unsigned long long capSequenceID3, capSequenceID4;  // NOLINT
  SECTION("hipStreamGetCaptureInfo CaptureStatus in Thread") {
    HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, &capSequenceID3));
    REQUIRE(capSequenceID1 == capSequenceID3);
    REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  }
  SECTION("hipStreamGetCaptureInfo_v2 CaptureStatus in Thread") {
    HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, &capSequenceID4, nullptr, nullptr,
                                         nullptr));
    REQUIRE(capSequenceID2 == capSequenceID4);
    REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  }
}
/*
 * Create a stream s1. Start capturing s1. Get the capture info of s1. Launch
 * a thread. In the thread get the capture info of s1 using hipStreamGetCaptureInfo.
 * Verify that it is in state hipStreamCaptureStatusActive and capture id inside
 * thread is same as capture id in main function. Exit the thread and end the capture
 * The above scenario using hipStreamGetCaptureInfo_v2 API
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_CaptureStatus_InThread") {
  hipStream_t stream{nullptr};
  hipGraph_t graph{nullptr};

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  // Capture info
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  unsigned long long capSequenceID1, capSequenceID2;  // NOLINT
  // hipStreamGetCaptureInfo Capture status
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, &capSequenceID1));
  // hipStreamGetCaptureInfo_v2 Capture status
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, &capSequenceID2, nullptr, nullptr,
                                       nullptr));
  // Thread launch
  std::thread t(thread_func, stream, capSequenceID1, capSequenceID2);
  t.join();

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}
/*
 * Verify that the id remains same througout the capture. Create a stream s1.
 * Start capturing s1. Get the capture info of s1. Queue some oprations in s1.
 * Again get the capture info. Queue different operations in s1. Again get the
 * capture info. Verify that all the capture info are identical.
 * The above scenario using hipStreamGetCaptureInfo_v2 API
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_CaptureStatus_Througout_Capture") {
  hipStream_t stream{nullptr};
  hipGraph_t graph{nullptr};
  float *A_d, *B_d, *C_d, *D_d;
  float *A_h, *B_h, *C_h, *D_h;
  // Memory allocation to Host pointers
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  B_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  D_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(B_h != nullptr);
  REQUIRE(C_h != nullptr);
  REQUIRE(D_h != nullptr);
  // Memory allocation to Device pointers
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMalloc(&D_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(B_d != nullptr);
  REQUIRE(C_d != nullptr);
  REQUIRE(D_d != nullptr);
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  // Capture Info
  hipStreamCaptureStatus captureStatus1{hipStreamCaptureStatusNone},
      captureStatus2{hipStreamCaptureStatusNone}, captureStatus3{hipStreamCaptureStatusNone},
      captureStatus4{hipStreamCaptureStatusNone}, captureStatus5{hipStreamCaptureStatusNone},
      captureStatus6{hipStreamCaptureStatusNone};

  unsigned long long capSequenceID1, capSequenceID2, capSequenceID3,  // NOLINT
      capSequenceID4, capSequenceID5, capSequenceID6;

  // hipStreamGetCaptureInfo Capture status
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus1, &capSequenceID1));
  // hipStreamGetCaptureInfo_v2 Capture status
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus2, &capSequenceID2, nullptr, nullptr,
                                       nullptr));
  // Copy data to Device
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  // Kernal Operations
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                     C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  // hipStreamGetCaptureInfo Capture status
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus3, &capSequenceID3));
  REQUIRE(captureStatus1 == captureStatus3);
  REQUIRE(capSequenceID1 == capSequenceID3);
  // hipStreamGetCaptureInfo_v2 Capture status
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus4, &capSequenceID4, nullptr, nullptr,
                                       nullptr));
  REQUIRE(captureStatus2 == captureStatus4);
  REQUIRE(capSequenceID2 == capSequenceID4);

  // Kernal Operations
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d, B_d,
                     D_d, N);
  HIP_CHECK(hipMemcpyAsync(D_h, D_d, Nbytes, hipMemcpyDeviceToHost, stream));

  // hipStreamGetCaptureInfo Capture status
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus5, &capSequenceID5));
  REQUIRE(captureStatus3 == captureStatus5);
  REQUIRE(capSequenceID3 == capSequenceID5);
  // hipStreamGetCaptureInfo_v2 Capture status
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus6, &capSequenceID6, nullptr, nullptr,
                                       nullptr));
  REQUIRE(captureStatus4 == captureStatus6);
  REQUIRE(capSequenceID4 == capSequenceID6);

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(D_d));
  free(A_h);
  free(B_h);
  free(C_h);
  free(D_h);
}
/*
 * Create a stream with default flag (hipStreamDefault). Start capturing the stream.
 * Invoke hipStreamGetCaptureInfo() on the null stream. Verify hipErrorStreamCaptureImplicit
 * is returned by hipStreamGetCaptureInfo(). Verify capture status of created stream. Do some
 * operatoins. End the capture on the created stream.Verify the capture status. Execute the
 * graph and verify the output from the operations.
 * The above scenario using hipStreamGetCaptureInfo_v2 API
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_Nullstream_CaptureInfo") {
  hipStream_t stream{nullptr}, streamForGraph{nullptr};
  hipGraph_t graph{nullptr};
  hipError_t ret;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  float *A_d, *C_d;
  float *A_h, *C_h, *D_h;
  // Memory allocation to Host pointers
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  D_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);
  REQUIRE(D_h != nullptr);

  // Memory allocation to Device pointers
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  // Initialize input buffer
  for (size_t i = 0; i < N; ++i) {
    A_h[i] = 1.0f + i;
    D_h[i] = 0.0f;
  }
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone},
      captureStatus1{hipStreamCaptureStatusNone}, captureStatus2{hipStreamCaptureStatusNone};
  unsigned long long capSequenceID = 0,  // NOLINT
      capSequenceID1 = 0;

  // Verify the Error returned with null stream.
  SECTION("hipStreamGetCaptureInfo with null stream") {
    ret = hipStreamGetCaptureInfo(0, &captureStatus, &capSequenceID);
    REQUIRE(ret == hipErrorStreamCaptureImplicit);
  }
  SECTION("hipStreamGetCaptureInfo_v2 with null stream") {
    ret = hipStreamGetCaptureInfo_v2(0, &captureStatus, &capSequenceID, nullptr, nullptr, nullptr);
    REQUIRE(ret == hipErrorStreamCaptureImplicit);
  }


  // Check the capture status of the stream
  HIP_CHECK(hipStreamIsCapturing(stream, &captureStatus1));
  REQUIRE(captureStatus1 == hipStreamCaptureStatusActive);

  // Copy data to Device
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  // Kernal Operation
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                     C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  // End the capture
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  // Capture Status
  SECTION("hipStreamGetCaptureInfo with null stream after End capture") {
    ret = hipStreamGetCaptureInfo(0, &captureStatus2, &capSequenceID1);
    REQUIRE(ret == hipSuccess);
  }
  SECTION("hipStreamGetCaptureInfo_v2 with null stream after End capture") {
    ret =
        hipStreamGetCaptureInfo_v2(0, &captureStatus2, &capSequenceID1, nullptr, nullptr, nullptr);
    REQUIRE(ret == hipSuccess);
  }
  // Launch graph
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify Output
  for (size_t i = 0; i < N; i++) {
    D_h[i] = A_h[i] * A_h[i];
    REQUIRE(C_h[i] == D_h[i]);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
  free(A_h);
  free(C_h);
  free(D_h);
}

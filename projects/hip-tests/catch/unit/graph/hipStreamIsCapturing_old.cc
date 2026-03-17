/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

constexpr unsigned threadsPerBlock = 256;
constexpr size_t N = 100000;
constexpr int blocks =
    (N % threadsPerBlock == 0) ? (N / threadsPerBlock) : ((N / threadsPerBlock) + 1);
constexpr size_t Nbytes = N * sizeof(float);

/**
API - hipStreamIsCapturing
Negative Testcase Scenarios : Negative
  1) Check capture status with null pCaptureStatus.
  2) Check capture status with hipStreamPerThread and null pCaptureStatus.
Functional Testcase Scenarios :
  1) Check capture status with null stream.
  2) Check capture status with hipStreamPerThread.
  3) Functional : Create a stream, call api and check
     capture status is hipStreamCaptureStatusNone.
  4) Functional : Start capturing a stream and check
     capture status returned as hipStreamCaptureStatusActive.
  5) Functional : Stop capturing a stream and check
     status is returned as hipStreamCaptureStatusNone.
  6) Functional : Use hipStreamPerThread, call api and check
     capture status is hipStreamCaptureStatusNone.
  7) Functional : Start capturing using hipStreamPerThread and check
     capture status returned as hipStreamCaptureStatusActive.
  8) Functional : Stop capturing using hipStreamPerThread and check
     status is returned as hipStreamCaptureStatusNone.
  9) Functional : Create 2 streams s1 and s2. Start capturing s1. Record event e1
     on s1 and wait for event e1 on s2. Queue some operations in s1 and s2. Invoke
     hipStreamIsCapturing on both s1 and s2. Verify that the capture info (status)
     of both s1 and s2 are identical. Record event e2 on s2 and wait for event e2
     on s1. End the capture of stream s1. Invoke hipStreamIsCapturing on both streams.
     Verify that the capture info(status)of both s1 and s2 are identical
  10)Functional : Create a stream s1. Start capturing s1. Get the capture info using
     hipStreamIsCapturing of s1. Launch a thread. In the thread get the capture info
     of s1 using hipStreamIsCapturing. Verify that it is in state hipStreamCaptureStatusActive
     in thread. Exit the thread and end the capture.
  11)Functional : Create a stream with default flag (hipStreamDefault). Start capturing
     the stream. Invoke hipStreamIsCapturing() on the null stream. Verify
hipErrorStreamCaptureImplicit is returned by hipStreamIsCapturing(). Verify capture status of
created stream. Do some operatoins. End the capture on the created stream. Execute the graph and
verify the output from the operations.
*/

/*
 * Create 2 streams s1 and s2. Start capturing s1. Record event e1 on s1 and wait
 * for event e1 on s2. Queue some operations in s1 and s2. Invoke hipStreamIsCapturing
 * on both s1 and s2. Verify that the capture info (status) of both s1 and s2 are identical.
 * Record event e2 on s2 and wait for event e2 on s1. End the capture of stream s1.
 * Invoke hipStreamIsCapturing on both streams. Verify that the capture info(status)
 * of both s1 and s2 are identical.
 */
HIP_TEST_CASE(Unit_hipStreamIsCapturing_ParentAndForkedStream) {
  hipStream_t stream1{nullptr}, stream2{nullptr};
  hipEvent_t event2{nullptr}, forkStreamEvent{nullptr};
  hipGraph_t graph{nullptr};
  size_t Nbytes = N * sizeof(float);
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

  // Initialize input buffer
  for (size_t i = 0; i < N; ++i) {
    A_h[i] = 3.146f + i;  // Pi
    B_h[i] = A_h[i];
  }
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
  // Capturing info
  HIP_CHECK(hipStreamIsCapturing(stream1, &captureStatus1));
  HIP_CHECK(hipStreamIsCapturing(stream2, &captureStatus2));
  // Verfication of results
  REQUIRE(captureStatus1 == hipStreamCaptureStatusActive);
  REQUIRE(captureStatus2 == hipStreamCaptureStatusActive);

  HIP_CHECK(hipEventRecord(event2, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, event2, 0));
  // End the capture
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  REQUIRE(graph != nullptr);

  // Capture Info
  HIP_CHECK(hipStreamIsCapturing(stream1, &captureStatus3));
  HIP_CHECK(hipStreamIsCapturing(stream2, &captureStatus4));
  // Verification of results
  REQUIRE(captureStatus3 == hipStreamCaptureStatusNone);
  REQUIRE(captureStatus4 == hipStreamCaptureStatusNone);

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

/*
 * Create a stream with default flag (hipStreamDefault). Start capturing the stream.
 * Invoke hipStreamIsCapturing() on the null stream. Verify hipErrorStreamCaptureImplicit
 * is returned by hipStreamIsCapturing(). Verify capture status of created stream. Do some
 * operatoins. End the capture on the created stream. Execute the graph and verify the output from
 * the operations.
 */
HIP_TEST_CASE(Unit_hipStreamIsCapturing_ChkNullStrmStatus) {
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
  // Verify the Error returned if null stream is passed.
  ret = hipStreamIsCapturing(0, &captureStatus);
  REQUIRE(ret == hipErrorStreamCaptureImplicit);
  // Check the capture status of the stream
  HIP_CHECK(hipStreamIsCapturing(stream, &captureStatus1));
  REQUIRE(captureStatus1 == hipStreamCaptureStatusActive);
  // Copy data to Device
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  // Kernal Operations
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                     C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
  // End the capture
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  REQUIRE(graph != nullptr);

  ret = hipStreamIsCapturing(0, &captureStatus2);
  REQUIRE(ret == hipSuccess);

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

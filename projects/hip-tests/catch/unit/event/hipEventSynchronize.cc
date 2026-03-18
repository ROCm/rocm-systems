/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <kernels.hh>
#include <hip_test_checkers.hh>

/**
 * @addtogroup hipEventSynchronize hipEventSynchronize
 * @{
 * @ingroup EventTest
 * `hipEventSynchronize(hipEvent_t event)` -
 * Wait for an event to complete.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipEventIpc
 *  - @ref Unit_hipEventMGpuMThreads_1
 *  - @ref Unit_hipEventMGpuMThreads_2
 *  - @ref Unit_hipEventMGpuMThreads_3
 */

void testSynchronize(hipStream_t stream) {
  constexpr size_t N = 1024;

  constexpr int blocks = 1024;

  constexpr size_t Nbytes = N * sizeof(float);

  float *A_h, *B_h, *C_h;
  float *A_d, *B_d, *C_d;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  hipEvent_t end_event;
  HIP_CHECK(hipEventCreate(&end_event));

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  HipTest::launchKernel<float>(HipTest::vectorADD<float>, blocks, 1, 0, stream,
                               static_cast<const float*>(A_d), static_cast<const float*>(B_d), C_d,
                               N);

  if (stream != nullptr) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // Record the end_event
  HIP_CHECK(hipEventRecord(end_event, nullptr));
  // Wait for the end_event to complete
  HIP_CHECK(hipEventSynchronize(end_event));

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipEventDestroy(end_event));

  HipTest::checkVectorADD(A_h, B_h, C_h, N, true);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Test Description
 * ------------------------
 *  - Synchronization of an event that is completed after a simple kernel launch (on null/created
 * stream). Test source
 * ------------------------
 *  - unit/event/hipEventSynchronize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventSynchronize_Default_Positive) {
  hipStream_t stream{nullptr};

  SECTION("Kernel launched in null stream") { testSynchronize(stream); }

  SECTION("Kernel launched in created stream") {
    HIP_CHECK(hipStreamCreate(&stream));
    testSynchronize(stream);
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Synchronization of an event that has not been recorded.
 * Test source
 * ------------------------
 *  - unit/event/hipEventSynchronize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventSynchronize_NoEventRecord_Positive) {
  constexpr size_t N = 1024;

  constexpr int blocks = 1024;

  constexpr size_t Nbytes = N * sizeof(float);

  float *A_h, *B_h, *C_h;
  float *A_d, *B_d, *C_d;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  hipEvent_t dummy_event;
  HIP_CHECK(hipEventCreate(&dummy_event));

  hipEvent_t end_event;
  HIP_CHECK(hipEventCreate(&end_event));

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  HipTest::launchKernel<float>(HipTest::vectorADD<float>, blocks, 1, 0, 0,
                               static_cast<const float*>(A_d), static_cast<const float*>(B_d), C_d,
                               N);

  // Record the end_event
  HIP_CHECK(hipEventRecord(end_event, NULL));

  // When hipEventSynchronized is called on event that has not been recorded,
  // the function returns immediately
  HIP_CHECK(hipEventSynchronize(dummy_event));

  // Wait for end_event to complete
  HIP_CHECK(hipEventSynchronize(end_event));

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipEventDestroy(dummy_event));
  HIP_CHECK(hipEventDestroy(end_event));

  HipTest::checkVectorADD(A_h, B_h, C_h, N, true);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * End doxygen group EventTest.
 * @}
 */

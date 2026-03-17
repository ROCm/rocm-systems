/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <chrono>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include "hip/hip_runtime_api.h"
#include <utils.hh>
/**
 * @addtogroup hipEventDestroy hipEventDestroy
 * @{
 * @ingroup EventTest
 * `hipEventDestroy(hipEvent_t event)` -
 * Destroy the specified event.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipEventIpc
 */

static constexpr size_t vectorSize{1024};

/*
 * Launches vectorAdd kernel with a delay
 */
static inline void launchVectorAdd(float*& A_h, float*& B_h, float*& C_h,
                                   std::chrono::milliseconds delay, hipStream_t stream = nullptr) {
  float* A_d{nullptr};
  float* B_d{nullptr};
  float* C_d{nullptr};
  HipTest::initArraysForHost(&A_h, &B_h, &C_h, vectorSize, true);
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&B_d), B_h, 0));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&C_d), C_h, 0));
  LaunchDelayKernel(delay, stream);
  HipTest::vectorADD<<<1, 1, 0, stream>>>(A_d, B_d, C_d, vectorSize);
}

/**
 * Test Description
 * ------------------------
 *  - Destroys the event before launched kernel has finished running.
 * Test source
 * ------------------------
 *  - unit/event/hipEventDestroy.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventDestroy_Unfinished) {
  hipEvent_t event;

  HIP_CHECK(hipEventCreate(&event));

  float *A_h, *B_h, *C_h;
  launchVectorAdd(A_h, B_h, C_h, std::chrono::milliseconds(1000));

  HIP_CHECK(hipEventRecord(event));
  HIP_CHECK_ERROR(hipEventQuery(event), hipErrorNotReady);
  HIP_CHECK(hipEventDestroy(event));

  HIP_CHECK(hipDeviceSynchronize());
  HipTest::checkVectorADD(A_h, B_h, C_h, vectorSize);
  HipTest::freeArraysForHost(A_h, B_h, C_h, true);
}

/**
 * Test Description
 * ------------------------
 *  - Destroys the event that is enqueued into a stream.
 * Test source
 * ------------------------
 *  - unit/event/hipEventDestroy.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventDestroy_WithWaitingStream) {
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float *A_h, *B_h, *C_h;
  launchVectorAdd(A_h, B_h, C_h, std::chrono::milliseconds(1000), stream);

  HIP_CHECK(hipEventRecord(event, stream));
  HIP_CHECK_ERROR(hipEventQuery(event), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));

  HipTest::checkVectorADD(A_h, B_h, C_h, vectorSize);
  HipTest::freeArraysForHost(A_h, B_h, C_h, true);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the event is `nullptr`
 *      - Expected output: return `hipErrorInvalidResourceHandle`
 *    -# When event is destroyed twice
 *      - Expected output: return `hipErrorContextIsDestroyed`
 * Test source
 * ------------------------
 *  - unit/event/hipEventDestroy.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipEventDestroy_Negative) {
  SECTION("Invalid Event") {
    hipEvent_t event{nullptr};
    HIP_CHECK_ERROR(hipEventDestroy(event), hipErrorInvalidResourceHandle);
  }
}

TEST_CASE(Unit_hipEventDestroy_Verify_Capture) {
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipStreamCaptureMode mode = GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
                                       hipStreamCaptureModeRelaxed);
  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  HIP_CHECK(hipEventDestroy(event));
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group EventTest.
 * @}
 */

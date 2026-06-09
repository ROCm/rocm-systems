/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipExecutionCtxRecordEvent hipExecutionCtxRecordEvent
 * @{
 * @ingroup ExecutionContextTest
 * `hipExecutionCtxRecordEvent` and `hipExecutionCtxWaitEvent` APIs
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include "hip_executionctx_common.hh"


/**
 * Test Description
 * ------------------------
 *  - Records an event on a green context and waits on it
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxRecordWaitEvent_Sanity) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipEvent_t event = nullptr;
  hipExecutionCtx_t green_ctx = nullptr;
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);
  
  hipStream_t stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, green_ctx, hipStreamNonBlocking, 0x0));
  REQUIRE(stream != nullptr);

  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);

  HIP_CHECK(hipExecutionCtxRecordEvent(green_ctx, event));
  HIP_CHECK(hipExecutionCtxWaitEvent(green_ctx, event));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipExecutionCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Validates record/wait behavior with real work in execution context streams
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxRecordEventFunctional) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipEvent_t event = nullptr;
  hipExecutionCtx_t green_ctx = nullptr;
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);

  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);

  hipStream_t stream1 = nullptr;
  hipStream_t stream2 = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream1, green_ctx, hipStreamNonBlocking, 0x0));
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream2, green_ctx, hipStreamNonBlocking, 0x0));
  REQUIRE(stream1 != nullptr);
  REQUIRE(stream2 != nullptr);

  constexpr size_t kNumElements = 1 << 16;
  const size_t kBytes = kNumElements * sizeof(int);
  int* h_a = reinterpret_cast<int*>(malloc(kBytes));
  int* h_b = reinterpret_cast<int*>(malloc(kBytes));
  int* h_c = reinterpret_cast<int*>(malloc(kBytes));
  REQUIRE(h_a != nullptr);
  REQUIRE(h_b != nullptr);
  REQUIRE(h_c != nullptr);

  for (size_t i = 0; i < kNumElements; ++i) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(2 * i);
    h_c[i] = 0;
  }

  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_c, kBytes));

  HIP_CHECK(hipMemcpyAsync(d_a, h_a, kBytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b, kBytes, hipMemcpyHostToDevice, stream1));
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kNumElements + kThreads - 1) / kThreads);
  HipTest::vectorADD<<<blocks, kThreads, 0, stream1>>>(d_a, d_b, d_c, kNumElements);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipExecutionCtxRecordEvent(green_ctx, event));
  HIP_CHECK(hipExecutionCtxWaitEvent(green_ctx, event));

  HIP_CHECK(hipMemcpyAsync(h_c, d_c, kBytes, hipMemcpyDeviceToHost, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));

  for (size_t i = 0; i < kNumElements; ++i) {
    REQUIRE(h_c[i] == h_a[i] + h_b[i]);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipExecutionCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Validates that recordEvent captures work across all 3 streams in a green
 *    context, and waitEvent blocks subsequent work until that work completes.
 *    Each stream independently copies host data and doubles it via vectorADD.
 *    After record/wait, a final kernel on stream0 combines results from all 3
 *    streams.  If record/wait fails to synchronize any stream, the final result
 *    will read stale or un-doubled values and the verification will fail.
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipExecutionCtxRecordWait_Blocking) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipEvent_t event = nullptr;
  hipExecutionCtx_t green_ctx = nullptr;
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);

  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);

  hipStream_t stream0 = nullptr, stream1 = nullptr, stream2 = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream0, green_ctx, hipStreamNonBlocking, 0x0));
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream1, green_ctx, hipStreamNonBlocking, 0x0));
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream2, green_ctx, hipStreamNonBlocking, 0x0));
  REQUIRE(stream0 != nullptr);
  REQUIRE(stream1 != nullptr);
  REQUIRE(stream2 != nullptr);

  constexpr size_t kNumElements = 1 << 16;
  const size_t kBytes = kNumElements * sizeof(int);

  int* h_a = reinterpret_cast<int*>(malloc(kBytes));
  int* h_b = reinterpret_cast<int*>(malloc(kBytes));
  int* h_c = reinterpret_cast<int*>(malloc(kBytes));
  int* h_result = reinterpret_cast<int*>(malloc(kBytes));
  REQUIRE(h_a != nullptr);
  REQUIRE(h_b != nullptr);
  REQUIRE(h_c != nullptr);
  REQUIRE(h_result != nullptr);

  for (size_t i = 0; i < kNumElements; ++i) {
    h_a[i] = static_cast<int>(i + 1);
    h_b[i] = static_cast<int>(i + 2);
    h_c[i] = static_cast<int>(i + 3);
    h_result[i] = 0;
  }

  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c = nullptr;
  int* d_result = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_c, kBytes));
  HIP_CHECK(hipMalloc(&d_result, kBytes));

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kNumElements + kThreads - 1) / kThreads);

  // Stream 0: H2D copy, then d_a[i] = d_a[i] + d_a[i] = 2*(i+1)
  HIP_CHECK(hipMemcpyAsync(d_a, h_a, kBytes, hipMemcpyHostToDevice, stream0));
  HipTest::vectorADD<<<blocks, kThreads, 0, stream0>>>(d_a, d_a, d_a, kNumElements);
  HIP_CHECK(hipGetLastError());

  // Stream 1: H2D copy, then d_b[i] = d_b[i] + d_b[i] = 2*(i+2)
  HIP_CHECK(hipMemcpyAsync(d_b, h_b, kBytes, hipMemcpyHostToDevice, stream1));
  HipTest::vectorADD<<<blocks, kThreads, 0, stream1>>>(d_b, d_b, d_b, kNumElements);
  HIP_CHECK(hipGetLastError());

  // Stream 2: H2D copy, then d_c[i] = d_c[i] + d_c[i] = 2*(i+3)
  HIP_CHECK(hipMemcpyAsync(d_c, h_c, kBytes, hipMemcpyHostToDevice, stream2));
  HipTest::vectorADD<<<blocks, kThreads, 0, stream2>>>(d_c, d_c, d_c, kNumElements);
  HIP_CHECK(hipGetLastError());

  // Record captures all 3 streams; wait blocks all streams until recorded work completes
  HIP_CHECK(hipExecutionCtxRecordEvent(green_ctx, event));
  HIP_CHECK(hipExecutionCtxWaitEvent(green_ctx, event));

  // Final kernels on stream0 consume results from all 3 streams:
  //   d_result = d_a + d_b = 2*(i+1) + 2*(i+2) = 4i + 6
  //   d_result = d_result + d_c = (4i+6) + 2*(i+3) = 6i + 12
  HipTest::vectorADD<<<blocks, kThreads, 0, stream0>>>(d_a, d_b, d_result, kNumElements);
  HIP_CHECK(hipGetLastError());
  HipTest::vectorADD<<<blocks, kThreads, 0, stream0>>>(d_result, d_c, d_result, kNumElements);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpyAsync(h_result, d_result, kBytes, hipMemcpyDeviceToHost, stream0));
  HIP_CHECK(hipStreamSynchronize(stream0));

  for (size_t i = 0; i < kNumElements; ++i) {
    REQUIRE(h_result[i] == static_cast<int>(6 * i + 12));
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  HIP_CHECK(hipFree(d_result));
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_result);
  HIP_CHECK(hipStreamDestroy(stream0));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipExecutionCtxDestroy(green_ctx));
}

/**
 * End doxygen group hipExecutionCtxRecordEvent.
 * @}
 */

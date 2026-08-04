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

#include <chrono>
#include <cmath>
#include <vector>


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
 * Record a start/stop pair on a 1-stream ctx and verify hipEventElapsedTime
 * returns a positive finite value.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxRecordEvent_ElapsedTime_SingleStream) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
  REQUIRE(ctx != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, ctx, hipStreamNonBlocking, 0x0));
  REQUIRE(stream != nullptr);

  hipEvent_t start = nullptr, stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  REQUIRE(start != nullptr);
  REQUIRE(stop != nullptr);

  constexpr size_t kN = 1 << 18;
  const size_t kBytes = kN * sizeof(int);
  std::vector<int> h_a(kN), h_b(kN), h_c(kN, 0);
  for (size_t i = 0; i < kN; ++i) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(2 * i);
  }

  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_c, kBytes));

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kN + kThreads - 1) / kThreads);

  HIP_CHECK(hipExecutionCtxRecordEvent(ctx, start));
  HIP_CHECK(hipMemcpyAsync(d_a, h_a.data(), kBytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b.data(), kBytes, hipMemcpyHostToDevice, stream));
  HipTest::vectorADD<<<blocks, kThreads, 0, stream>>>(d_a, d_b, d_c, kN);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(h_c.data(), d_c, kBytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipExecutionCtxRecordEvent(ctx, stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = -1.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  REQUIRE(ms > 0.0f);
  REQUIRE(std::isfinite(ms));

  for (size_t i = 0; i < kN; ++i) {
    REQUIRE(h_c[i] == h_a[i] + h_b[i]);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * Record a start/stop pair across a 3-stream ctx and verify hipEventElapsedTime
 * tracks wall-time within tolerance, exercising the fan-in marker path.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxRecordEvent_ElapsedTime_MultiStream) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
  REQUIRE(ctx != nullptr);

  constexpr int kNumStreams = 3;
  hipStream_t streams[kNumStreams] = {nullptr, nullptr, nullptr};
  for (int i = 0; i < kNumStreams; ++i) {
    HIP_CHECK(hipExecutionCtxStreamCreate(&streams[i], ctx, hipStreamNonBlocking, 0x0));
    REQUIRE(streams[i] != nullptr);
  }

  hipEvent_t start = nullptr, stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  // Workload sized so the per-stream kernel loop dominates launch overhead on
  // a modern GPU without being so large that a slow CI host times out.
  constexpr size_t kN = 1 << 20;
  const size_t kBytes = kN * sizeof(int);
  constexpr int kIters = 16;

  std::vector<int> h_a(kN), h_b(kN);
  for (size_t i = 0; i < kN; ++i) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(2 * i);
  }

  int* d_a[kNumStreams] = {nullptr, nullptr, nullptr};
  int* d_b[kNumStreams] = {nullptr, nullptr, nullptr};
  int* d_c[kNumStreams] = {nullptr, nullptr, nullptr};
  for (int i = 0; i < kNumStreams; ++i) {
    HIP_CHECK(hipMalloc(&d_a[i], kBytes));
    HIP_CHECK(hipMalloc(&d_b[i], kBytes));
    HIP_CHECK(hipMalloc(&d_c[i], kBytes));
  }

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kN + kThreads - 1) / kThreads);

  // Pre-stage inputs so the timed window only covers the compute loop.
  for (int i = 0; i < kNumStreams; ++i) {
    HIP_CHECK(hipMemcpyAsync(d_a[i], h_a.data(), kBytes, hipMemcpyHostToDevice, streams[i]));
    HIP_CHECK(hipMemcpyAsync(d_b[i], h_b.data(), kBytes, hipMemcpyHostToDevice, streams[i]));
  }
  HIP_CHECK(hipExecutionCtxSynchronize(ctx));

  // Wall-time around the recorded window. The submit loop is host-side cheap;
  // the dominant contribution to walltime_ms is the slowest stream's
  // kIters-deep kernel chain on the partitioned device.
  auto t_wall_start = std::chrono::steady_clock::now();

  HIP_CHECK(hipExecutionCtxRecordEvent(ctx, start));
  for (int it = 0; it < kIters; ++it) {
    for (int i = 0; i < kNumStreams; ++i) {
      HipTest::vectorADD<<<blocks, kThreads, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], kN);
      HIP_CHECK(hipGetLastError());
    }
  }
  HIP_CHECK(hipExecutionCtxRecordEvent(ctx, stop));
  HIP_CHECK(hipEventSynchronize(stop));

  auto t_wall_end = std::chrono::steady_clock::now();
  const double walltime_ms =
      std::chrono::duration<double, std::milli>(t_wall_end - t_wall_start).count();

  float ms = -1.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  REQUIRE(ms > 0.0f);
  REQUIRE(std::isfinite(ms));
  // If the fan-in marker truly waits on every ctx stream, the GPU elapsed
  // time should track wall-time closely. A 0.5x floor leaves slack for
  // kernel-launch overhead and scheduler variance on busy hosts.
  REQUIRE(static_cast<double>(ms) >= 0.5 * walltime_ms);

  for (int i = 0; i < kNumStreams; ++i) {
    HIP_CHECK(hipFree(d_a[i]));
    HIP_CHECK(hipFree(d_b[i]));
    HIP_CHECK(hipFree(d_c[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * Negative-parameter validation for hipExecutionCtxRecordEvent / WaitEvent:
 * null ctx, null event, destroyed event.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxRecordWaitEvent_Negative) {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
  REQUIRE(ctx != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, ctx, hipStreamNonBlocking, 0x0));
  REQUIRE(stream != nullptr);

  hipEvent_t event = nullptr;
  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);

  SECTION("null ctx, record") {
    HIP_CHECK_ERROR(hipExecutionCtxRecordEvent(nullptr, event), hipErrorInvalidValue);
  }
  SECTION("null ctx, wait") {
    HIP_CHECK_ERROR(hipExecutionCtxWaitEvent(nullptr, event), hipErrorInvalidValue);
  }
  SECTION("null event, record") {
    HIP_CHECK_ERROR(hipExecutionCtxRecordEvent(ctx, nullptr), hipErrorInvalidHandle);
  }
  SECTION("null event, wait") {
    HIP_CHECK_ERROR(hipExecutionCtxWaitEvent(ctx, nullptr), hipErrorInvalidHandle);
  }
#if HT_AMD
  // CUDA's cudaExecutionCtxRecordEvent does not validate a destroyed/dangling
  // event handle (use-after-destroy is UB at the CUDA level and segfaults),
  // so these SECTIONs are AMD-only.
  SECTION("destroyed event, record") {
    hipEvent_t dead = nullptr;
    HIP_CHECK(hipEventCreate(&dead));
    HIP_CHECK(hipEventDestroy(dead));
    HIP_CHECK_ERROR(hipExecutionCtxRecordEvent(ctx, dead), hipErrorInvalidHandle);
  }
  SECTION("destroyed event, wait") {
    hipEvent_t dead = nullptr;
    HIP_CHECK(hipEventCreate(&dead));
    HIP_CHECK(hipEventDestroy(dead));
    HIP_CHECK_ERROR(hipExecutionCtxWaitEvent(ctx, dead), hipErrorInvalidHandle);
  }
#endif

  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * Record an event on ctx A, wait on ctx B, and verify data written on A is
 * visible to follow-up work on B.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxWaitEvent_CrossCtx) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  const unsigned int totalSMs = input.sm.smCount;
  const unsigned int alignment = input.sm.smCoscheduledAlignment;
  const unsigned int halfSMs = (totalSMs / 2 / alignment) * alignment;
  if (halfSMs < alignment) {
    SUCCEED("Device too small to split into two SM partitions");
    return;
  }

  hipDevSmResourceGroupParams params[2] = {};
  params[0].smCount = halfSMs;
  params[1].smCount = halfSMs;
  hipDevResource splits[2] = {};
  HIP_CHECK(hipDevSmResourceSplit(splits, 2, &input, nullptr, 0, params));

  hipDevResourceDesc_t descA{};
  hipDevResourceDesc_t descB{};
  HIP_CHECK(hipDevResourceGenerateDesc(&descA, &splits[0], 1));
  HIP_CHECK(hipDevResourceGenerateDesc(&descB, &splits[1], 1));

  hipExecutionCtx_t ctxA = nullptr;
  hipExecutionCtx_t ctxB = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctxA, descA, 0, 0));
  HIP_CHECK(hipGreenCtxCreate(&ctxB, descB, 0, 0));
  REQUIRE(ctxA != nullptr);
  REQUIRE(ctxB != nullptr);

  hipStream_t a0 = nullptr, a1 = nullptr, b0 = nullptr, b1 = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&a0, ctxA, hipStreamNonBlocking, 0x0));
  HIP_CHECK(hipExecutionCtxStreamCreate(&a1, ctxA, hipStreamNonBlocking, 0x0));
  HIP_CHECK(hipExecutionCtxStreamCreate(&b0, ctxB, hipStreamNonBlocking, 0x0));
  HIP_CHECK(hipExecutionCtxStreamCreate(&b1, ctxB, hipStreamNonBlocking, 0x0));
  REQUIRE(a0 != nullptr);
  REQUIRE(a1 != nullptr);
  REQUIRE(b0 != nullptr);
  REQUIRE(b1 != nullptr);

  hipEvent_t evt = nullptr;
  HIP_CHECK(hipEventCreate(&evt));
  REQUIRE(evt != nullptr);

  constexpr size_t kN = 1 << 16;
  const size_t kBytes = kN * sizeof(int);
  std::vector<int> h_x(kN), h_y(kN), h_out(kN, 0);
  for (size_t i = 0; i < kN; ++i) {
    h_x[i] = static_cast<int>(i);
    h_y[i] = 1;
  }

  int* d_x = nullptr;
  int* d_y = nullptr;
  int* d_x_doubled = nullptr;
  int* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_x, kBytes));
  HIP_CHECK(hipMalloc(&d_y, kBytes));
  HIP_CHECK(hipMalloc(&d_x_doubled, kBytes));
  HIP_CHECK(hipMalloc(&d_out, kBytes));

  // ctx A populates two device buffers across both of its streams. The
  // fan-in record on ctxA captures both streams' pending work.
  HIP_CHECK(hipMemcpyAsync(d_x, h_x.data(), kBytes, hipMemcpyHostToDevice, a0));
  HIP_CHECK(hipMemcpyAsync(d_y, h_y.data(), kBytes, hipMemcpyHostToDevice, a1));

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kN + kThreads - 1) / kThreads);
  // d_x_doubled[i] = d_x[i] + d_x[i] = 2 * i on stream a0
  HipTest::vectorADD<<<blocks, kThreads, 0, a0>>>(d_x, d_x, d_x_doubled, kN);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipExecutionCtxRecordEvent(ctxA, evt));

  // ctx B waits on ctx A's recorded event. Subsequent work on ctxB's streams
  // must observe the post-event state from ctxA.
  HIP_CHECK(hipExecutionCtxWaitEvent(ctxB, evt));

  // d_out[i] = d_x_doubled[i] + d_y[i] = 2 * i + 1 on stream b0.
  HipTest::vectorADD<<<blocks, kThreads, 0, b0>>>(d_x_doubled, d_y, d_out, kN);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(h_out.data(), d_out, kBytes, hipMemcpyDeviceToHost, b0));

  HIP_CHECK(hipExecutionCtxSynchronize(ctxB));

  for (size_t i = 0; i < kN; ++i) {
    REQUIRE(h_out[i] == 2 * static_cast<int>(i) + 1);
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_x_doubled));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipEventDestroy(evt));
  HIP_CHECK(hipStreamDestroy(a0));
  HIP_CHECK(hipStreamDestroy(a1));
  HIP_CHECK(hipStreamDestroy(b0));
  HIP_CHECK(hipStreamDestroy(b1));
  HIP_CHECK(hipExecutionCtxDestroy(ctxA));
  HIP_CHECK(hipExecutionCtxDestroy(ctxB));
}

/**
 * Record on a ctx, wait via hipStreamWaitEvent on a plain stream, and verify
 * the dependent work observes the ctx-side writes.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxRecordEvent_WaitFromPlainStream) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t cs = nullptr;
  MakeCtxAndStream(ctx, cs);

  hipStream_t ps = nullptr;
  HIP_CHECK(hipStreamCreate(&ps));
  REQUIRE(ps != nullptr);

  hipEvent_t evt = nullptr;
  HIP_CHECK(hipEventCreate(&evt));

  constexpr size_t kN = 1 << 16;
  const size_t kBytes = kN * sizeof(int);
  std::vector<int> h_a(kN), h_b(kN), h_out(kN, 0);
  for (size_t i = 0; i < kN; ++i) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(2 * i);
  }

  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_sum = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_sum, kBytes));

  // Producer side: stage inputs and compute the sum on the ctx-owned stream.
  HIP_CHECK(hipMemcpyAsync(d_a, h_a.data(), kBytes, hipMemcpyHostToDevice, cs));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b.data(), kBytes, hipMemcpyHostToDevice, cs));
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kN + kThreads - 1) / kThreads);
  HipTest::vectorADD<<<blocks, kThreads, 0, cs>>>(d_a, d_b, d_sum, kN);
  HIP_CHECK(hipGetLastError());

  // Record the ctx-wide event; have the plain stream wait on it via the
  // standard cross-stream wait API; then drain d_sum on the plain stream.
  HIP_CHECK(hipExecutionCtxRecordEvent(ctx, evt));
  HIP_CHECK(hipStreamWaitEvent(ps, evt, 0));
  HIP_CHECK(hipMemcpyAsync(h_out.data(), d_sum, kBytes, hipMemcpyDeviceToHost, ps));
  HIP_CHECK(hipStreamSynchronize(ps));

  for (size_t i = 0; i < kN; ++i) {
    REQUIRE(h_out[i] == h_a[i] + h_b[i]);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_sum));
  HIP_CHECK(hipEventDestroy(evt));
  HIP_CHECK(hipStreamDestroy(ps));
  HIP_CHECK(hipStreamDestroy(cs));
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * Record on a plain stream via hipEventRecord, wait via hipExecutionCtxWaitEvent
 * on a ctx, and verify ctx work observes the plain-side writes.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxWaitEvent_RecordedOnPlainStream) {
  hipExecutionCtx_t ctx = nullptr;
  hipStream_t cs = nullptr;
  MakeCtxAndStream(ctx, cs);

  hipStream_t ps = nullptr;
  HIP_CHECK(hipStreamCreate(&ps));
  REQUIRE(ps != nullptr);

  hipEvent_t evt = nullptr;
  HIP_CHECK(hipEventCreate(&evt));

  constexpr size_t kN = 1 << 16;
  const size_t kBytes = kN * sizeof(int);
  std::vector<int> h_a(kN), h_b(kN), h_out(kN, 0);
  for (size_t i = 0; i < kN; ++i) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(2 * i);
  }

  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_sum = nullptr;
  int* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_sum, kBytes));
  HIP_CHECK(hipMalloc(&d_out, kBytes));

  // Producer on the plain stream computes d_sum, then records the event.
  HIP_CHECK(hipMemcpyAsync(d_a, h_a.data(), kBytes, hipMemcpyHostToDevice, ps));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b.data(), kBytes, hipMemcpyHostToDevice, ps));
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kN + kThreads - 1) / kThreads);
  HipTest::vectorADD<<<blocks, kThreads, 0, ps>>>(d_a, d_b, d_sum, kN);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(evt, ps));

  // The ctx's wait fans the event out to every ctx-stream's per-stream
  // wait list; the dependent kernel on cs must observe d_sum.
  HIP_CHECK(hipExecutionCtxWaitEvent(ctx, evt));
  HipTest::vectorADD<<<blocks, kThreads, 0, cs>>>(d_sum, d_sum, d_out, kN);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(h_out.data(), d_out, kBytes, hipMemcpyDeviceToHost, cs));
  HIP_CHECK(hipExecutionCtxSynchronize(ctx));

  for (size_t i = 0; i < kN; ++i) {
    REQUIRE(h_out[i] == 2 * (h_a[i] + h_b[i]));
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_sum));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipEventDestroy(evt));
  HIP_CHECK(hipStreamDestroy(ps));
  HIP_CHECK(hipStreamDestroy(cs));
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * End doxygen group hipExecutionCtxRecordEvent.
 * @}
 */

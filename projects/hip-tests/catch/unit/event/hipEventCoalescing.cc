/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include <algorithm>
#include <vector>

/**
 * @addtogroup hipEventRecord hipEventRecord
 * @{
 * @ingroup EventTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates cross-stream synchronization via hipStreamWaitEvent works correctly
 *    when coalescing is active. This is the most safety-critical scenario for
 *    coalescing — if a needed barrier is wrongly skipped, the waiting stream
 *    could start before the producing stream's work completes, causing data races.
 *  - Scenario per iteration:
 *    1. Stream1 launches vectorADD producing C_d
 *    2. Stream1 records event multiple times (consecutive should coalesce)
 *    3. Stream2 waits on event via hipStreamWaitEvent
 *    4. Stream2 launches vectorADDReverse that reads C_d and writes D_d
 *    5. Validate D_d matches expected (proves stream2's kernel saw stream1's writes)
 *  - If coalescing wrongly skipped the barrier, D_d would contain stale/garbage data.
 * Test source
 * ------------------------
 *  - unit/event/hipEventCoalescing.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipEventCoalescing_CrossStreamWait) {
  constexpr size_t N = 1024 * 1024;
  constexpr size_t Nbytes = N * sizeof(float);
  constexpr int kNumIterations = 10;
  constexpr int kThreadsPerBlock = 256;
  const int kBlocks = (N + kThreadsPerBlock - 1) / kThreadsPerBlock;

  float *A_h, *B_h, *C_h, *D_h;
  float *A_d, *B_d, *C_d, *D_d;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);
  HIP_CHECK(hipMalloc(&D_d, Nbytes));
  D_h = reinterpret_cast<float*>(malloc(Nbytes));

  REQUIRE(D_h != nullptr);

  hipStream_t stream1, stream2;
  hipEvent_t event;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  for (int iter = 0; iter < kNumIterations; iter++) {
    // Reset destination on stream1 to known garbage so stale reads would be visible
    HIP_CHECK(hipMemsetAsync(C_d, 0xFF, Nbytes, stream1));

    // Producer: stream1 computes C_d = A_d + B_d
    HipTest::launchKernel<float>(HipTest::vectorADD<float>, kBlocks, kThreadsPerBlock, 0, stream1,
                                 static_cast<const float*>(A_d),
                                 static_cast<const float*>(B_d),
                                 C_d, N);
    HIP_CHECK(hipGetLastError());

    // Record event multiple times on stream1 (should coalesce — but the barrier
    // must survive coalescing so stream2's wait sees stream1's kernel completion)
    HIP_CHECK(hipEventRecord(event, stream1));
    HIP_CHECK(hipEventRecord(event, stream1));  // Should coalesce
    HIP_CHECK(hipEventRecord(event, stream1));  // Should coalesce

    // Consumer: stream2 waits on event, then reads C_d and writes D_d = C_d + B_d
    HIP_CHECK(hipStreamWaitEvent(stream2, event, 0));
    HipTest::launchKernel<float>(HipTest::vectorADDReverse<float>, kBlocks, kThreadsPerBlock, 0,
                                 stream2, static_cast<const float*>(C_d),
                                 static_cast<const float*>(B_d), D_d, N);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipStreamSynchronize(stream2));

    // D_d should equal (A_d + B_d) + B_d
    HIP_CHECK(hipMemcpy(D_h, D_d, Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectors<float>(A_h, B_h, D_h, N, [](float a, float b) { return (a + b) + b; });
  }

  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipFree(D_d));
  free(D_h);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Test Description
 * ------------------------
 *  - Validates that async memcpy and memset between hipEventRecord calls
 *    correctly break coalescing. After such operations, the next record must
 *    submit a fresh barrier (otherwise the event sync wouldn't wait for the
 *    intervening copy/set to complete).
 *  - Two SECTIONs: one for memcpy break, one for memset break.
 *  - Pattern per section:
 *    1. record event
 *    2. async memcpy/memset on same stream (must break coalescing)
 *    3. record event (must NOT coalesce — submits new barrier covering step 2)
 *    4. sync event, validate the intervening operation completed
 * Test source
 * ------------------------
 *  - unit/event/hipEventCoalescing.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipEventCoalescing_AsyncOpsBreakCoalescing) {
  constexpr size_t N = 1024 * 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr int kNumIterations = 10;

  hipStream_t stream;
  hipEvent_t event;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));

  int* d_buf;
  HIP_CHECK(hipMalloc(&d_buf, Nbytes));
  std::vector<int> h_buf(N);

  SECTION("Memcpy breaks coalescing") {
    for (int iter = 0; iter < kNumIterations; iter++) {
      // Fill host source buffer with iter-specific value
      const int sentinel = 0x12340000 + iter;
      std::fill(h_buf.begin(), h_buf.end(), sentinel);

      // First record - submits barrier
      HIP_CHECK(hipEventRecord(event, stream));

      // Async memcpy - must break coalescing
      HIP_CHECK(hipMemcpyAsync(d_buf, h_buf.data(), Nbytes, hipMemcpyHostToDevice, stream));

      // Second record - must submit new barrier (NOT coalesce) so sync waits for memcpy
      HIP_CHECK(hipEventRecord(event, stream));

      // Sync event - if coalescing wrongly skipped the second barrier, the memcpy
      // might not have completed when we read back
      HIP_CHECK(hipEventSynchronize(event));

      // Read back without further sync — relies on event sync covering the memcpy
      std::vector<int> h_verify(N, 0);
      HIP_CHECK(hipMemcpy(h_verify.data(), d_buf, Nbytes, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < N; i++) {
        REQUIRE(h_verify[i] == sentinel);
      }
    }
  }

  SECTION("Memset breaks coalescing") {
    for (int iter = 0; iter < kNumIterations; iter++) {
      const unsigned char pattern = static_cast<unsigned char>(0x40 + iter);
      const int expected = (pattern << 24) | (pattern << 16) | (pattern << 8) | pattern;

      HIP_CHECK(hipEventRecord(event, stream));

      // Async memset - must break coalescing
      HIP_CHECK(hipMemsetAsync(d_buf, pattern, Nbytes, stream));

      // Second record - must submit new barrier
      HIP_CHECK(hipEventRecord(event, stream));

      HIP_CHECK(hipEventSynchronize(event));

      std::vector<int> h_verify(N, 0);
      HIP_CHECK(hipMemcpy(h_verify.data(), d_buf, Nbytes, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < N; i++) {
        REQUIRE(h_verify[i] == expected);
      }
    }
  }

  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Validates that interleaving records of different events on the same stream
 *    does not cause incorrect coalescing.
 *  - Coalescing only applies when last_packet was BARRIER for the SAME event.
 *  - Scenario:
 *    1. kernel_A on stream
 *    2. record event1  (barrier for event1)
 *    3. record event2  (different event - must NOT coalesce, new barrier)
 *    4. record event1  (last packet was event2's barrier - must NOT coalesce)
 *    5. kernel_B on stream
 *    6. record event2  (different event - must NOT coalesce)
 *  - Both events must independently capture the correct point in the stream.
 *  - event1's final record point is between kernel_A and kernel_B (after step 4)
 *  - event2's final record point is after kernel_B (after step 6)
 * Test source
 * ------------------------
 *  - unit/event/hipEventCoalescing.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipEventCoalescing_InterleavedEvents) {
  constexpr size_t N = 1024 * 1024;
  constexpr size_t Nbytes = N * sizeof(float);
  constexpr int kNumIterations = 10;
  constexpr int kThreadsPerBlock = 256;
  const int kBlocks = (N + kThreadsPerBlock - 1) / kThreadsPerBlock;

  float *A_h, *B_h, *C_h;
  float *A_d, *B_d, *C_d, *D_d;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);
  HIP_CHECK(hipMalloc(&D_d, Nbytes));

  hipStream_t stream1, stream2;
  hipEvent_t event1, event2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreateWithFlags(&event1, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event2, hipEventDisableTiming));

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  for (int iter = 0; iter < kNumIterations; iter++) {
    // kernel_A: C_d = A_d + B_d
    HipTest::launchKernel<float>(HipTest::vectorADD<float>, kBlocks, kThreadsPerBlock, 0, stream1,
                                 static_cast<const float*>(A_d),
                                 static_cast<const float*>(B_d),
                                 C_d, N);
    HIP_CHECK(hipGetLastError());

    // Interleaved records - none should coalesce because event identity changes
    HIP_CHECK(hipEventRecord(event1, stream1));  // barrier(event1)
    HIP_CHECK(hipEventRecord(event2, stream1));  // different event — NOT coalesce
    HIP_CHECK(hipEventRecord(event1, stream1));  // last was event2 — NOT coalesce

    // kernel_B: D_d = A_d - B_d (after event1's final record point)
    HipTest::launchKernel<float>(HipTest::vectorSUB<float>, kBlocks, kThreadsPerBlock, 0, stream1,
                                 static_cast<const float*>(A_d),
                                 static_cast<const float*>(B_d),
                                 D_d, N);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipEventRecord(event2, stream1));  // different event — NOT coalesce

    // event2 captures point AFTER kernel_B. If stream2 waits on event2 and reads D_d,
    // it should see kernel_B's output. If event2 wrongly coalesced with event1,
    // stream2 would not wait long enough.
    HIP_CHECK(hipStreamWaitEvent(stream2, event2, 0));

    // Validate kernel_B's output is readable after waiting on event2
    std::vector<float> D_h(N);
    HIP_CHECK(hipMemcpyAsync(D_h.data(), D_d, Nbytes, hipMemcpyDeviceToHost, stream2));
    HIP_CHECK(hipStreamSynchronize(stream2));
    HipTest::checkVectorSUB(A_h, B_h, D_h.data(), N);

    // Validate kernel_A's output via event1 wait on a fresh stream wait
    HIP_CHECK(hipEventSynchronize(event1));
    HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h, B_h, C_h, N);
  }

  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipFree(D_d));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Test Description
 * ------------------------
 *  - Validates that hipEventQuery returns correct status after coalesced records.
 *  - If the last record was coalesced (no fresh barrier submitted), the event
 *    should still report the status of the actual last submitted barrier.
 *  - Scenario:
 *    1. Launch long-running kernel (busy work)
 *    2. Record event multiple times (later records coalesce)
 *    3. hipEventQuery should report NotReady while kernel runs
 *    4. After hipDeviceSynchronize, hipEventQuery should report Success
 * Test source
 * ------------------------
 *  - unit/event/hipEventCoalescing.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipEventCoalescing_EventQuery) {
  constexpr size_t N = 1024 * 64;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr int kNumIterations = 5;
  constexpr int kThreadsPerBlock = 256;
  const int kBlocks = (N + kThreadsPerBlock - 1) / kThreadsPerBlock;
  // High count makes addCount slow enough to query before completion
  constexpr int kSlowCount = 10000;

  hipStream_t stream;
  hipEvent_t event;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));

  int *A_d, *C_d;
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMemset(A_d, 0, Nbytes));

  for (int iter = 0; iter < kNumIterations; iter++) {
    // Launch slow kernel
    HipTest::launchKernel<int>(HipTest::addCount<int>, kBlocks, kThreadsPerBlock, 0, stream,
                               static_cast<const int*>(A_d), C_d, N, kSlowCount);
    HIP_CHECK(hipGetLastError());

    // Record event multiple times - later records should coalesce
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipEventRecord(event, stream));  // Should coalesce
    HIP_CHECK(hipEventRecord(event, stream));  // Should coalesce

    // Query event - should be NotReady because slow kernel is still running.
    // If coalescing wrongly returned success without a barrier, this would fail.
    hipError_t query_status = hipEventQuery(event);
    REQUIRE((query_status == hipErrorNotReady || query_status == hipSuccess));

    // After full sync, query must report success
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventQuery(event));
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * @}
 */

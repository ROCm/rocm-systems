/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipSharedQueueAnyOrderOverlap hipSharedQueueAnyOrderOverlap
 * @{
 * @ingroup StreamTest
 * Correctness tests for the shared-queue any-order overlap optimization
 * (DEBUG_HIP_SHARED_QUEUE_ANYORDER). When more streams are created than the HW
 * queue pool (GPU_MAX_HW_QUEUES, default 4), the extra streams are recycled onto
 * shared HW queues. With the flag ON, the first kernel of an oversubscribed stream
 * has its head barrier bit cleared so it can overlap the prior tenant on the ring.
 *
 * These tests assert properties that must hold regardless of the flag:
 *   1. Independent oversubscribed streams always produce correct results.
 *   2. Explicit cross-stream dependencies (hipStreamWaitEvent) are always honored,
 *      because the optimization skips any launch that carries an event wait.
 *
 * To exercise the ON path, run the process with DEBUG_HIP_SHARED_QUEUE_ANYORDER=1
 * (optionally GPU_MAX_HW_QUEUES=1 to force a single shared ring); both tests must
 * still pass.
 */

#include <hip_test_common.hh>

#include <vector>

namespace {

// Spins to widen the execution window, then increments its own slot.
__global__ void SlotIncrementKernel(int* out, int slot, int busy_iters) {
  volatile float acc = 0.0f;
  for (int i = 0; i < busy_iters; ++i) {
    acc += __sinf(static_cast<float>(i) * 0.001f);
  }
  if (acc == 123456.789f) {  // never true; defeats dead-code elimination
    out[slot] += 1;
  }
  out[slot] += 1;
}

// Producer writes 'val' after a spin (late write); consumer reads buf into seen (early read).
__global__ void ProducerKernel(int* buf, int val, int busy_iters) {
  volatile float acc = 0.0f;
  for (int i = 0; i < busy_iters; ++i) acc += __sinf(static_cast<float>(i) * 0.001f);
  if (acc == 123456.789f) buf[0] += 1;  // defeat DCE
  buf[0] = val;
}
__global__ void ConsumerKernel(const int* buf, int* seen) { seen[0] = buf[0]; }

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Create far more streams than the HW queue pool so they oversubscribe shared
 *    rings, run an independent chain of increment kernels per stream on its own
 *    buffer slot, and verify every slot has the exact expected count.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipSharedQueueAnyOrderOverlap_Independence") {
  constexpr int kStreams = 32;  // >> GPU_MAX_HW_QUEUES (default 4) -> forced oversubscription
  constexpr int kChain = 4;
  constexpr int kReps = 20;
  constexpr int kBusyIters = 4000;

  std::vector<hipStream_t> streams(kStreams);
  for (auto& s : streams) HIP_CHECK(hipStreamCreate(&s));

  int* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, kStreams * sizeof(int)));
  HIP_CHECK(hipMemset(d_out, 0, kStreams * sizeof(int)));

  for (int r = 0; r < kReps; ++r) {
    for (int k = 0; k < kStreams; ++k) {
      for (int c = 0; c < kChain; ++c) {
        SlotIncrementKernel<<<dim3(1), dim3(1), 0, streams[k]>>>(d_out, k, kBusyIters);
      }
    }
  }
  for (auto& s : streams) HIP_CHECK(hipStreamSynchronize(s));

  std::vector<int> h(kStreams);
  HIP_CHECK(hipMemcpy(h.data(), d_out, kStreams * sizeof(int), hipMemcpyDeviceToHost));

  const int expected = kReps * kChain;
  for (int k = 0; k < kStreams; ++k) {
    INFO("stream slot " << k);
    REQUIRE(h[k] == expected);
  }

  HIP_CHECK(hipFree(d_out));
  for (auto& s : streams) HIP_CHECK(hipStreamDestroy(s));
}

/**
 * Test Description
 * ------------------------
 *  - Set up producer/consumer stream pairs that oversubscribe the HW queue pool,
 *    linking each consumer to its producer with hipEventRecord/hipStreamWaitEvent.
 *    The producer writes late and the consumer reads early, so a missed dependency
 *    would surface as a stale read. Verify the consumer always observes the
 *    producer's write, proving the optimization never drops an explicit wait.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipSharedQueueAnyOrderOverlap_EventDependencyHonored") {
  constexpr int kPairs = 8;  // 16 streams -> oversubscribes the default pool
  constexpr int kReps = 50;
  constexpr int kBusyIters = 8000;

  std::vector<hipStream_t> prod(kPairs), cons(kPairs);
  std::vector<hipEvent_t> ev(kPairs);
  std::vector<int*> buf(kPairs), seen(kPairs);
  for (int p = 0; p < kPairs; ++p) {
    HIP_CHECK(hipStreamCreate(&prod[p]));
    HIP_CHECK(hipStreamCreate(&cons[p]));
    HIP_CHECK(hipEventCreateWithFlags(&ev[p], hipEventDisableTiming));
    HIP_CHECK(hipMalloc(&buf[p], sizeof(int)));
    HIP_CHECK(hipMalloc(&seen[p], sizeof(int)));
    HIP_CHECK(hipMemset(buf[p], 0, sizeof(int)));
  }

  std::vector<int> h_seen(kPairs);
  for (int it = 1; it <= kReps; ++it) {
    for (int p = 0; p < kPairs; ++p) {
      ProducerKernel<<<dim3(1), dim3(1), 0, prod[p]>>>(buf[p], it, kBusyIters);
      HIP_CHECK(hipEventRecord(ev[p], prod[p]));
      HIP_CHECK(hipStreamWaitEvent(cons[p], ev[p], 0));
      ConsumerKernel<<<dim3(1), dim3(1), 0, cons[p]>>>(buf[p], seen[p]);
    }
    for (int p = 0; p < kPairs; ++p) {
      HIP_CHECK(hipStreamSynchronize(cons[p]));
      HIP_CHECK(hipMemcpy(&h_seen[p], seen[p], sizeof(int), hipMemcpyDeviceToHost));
      INFO("pair " << p << " iteration " << it);
      REQUIRE(h_seen[p] == it);  // consumer must observe producer's write, never a stale value
    }
  }

  for (int p = 0; p < kPairs; ++p) {
    HIP_CHECK(hipFree(buf[p]));
    HIP_CHECK(hipFree(seen[p]));
    HIP_CHECK(hipEventDestroy(ev[p]));
    HIP_CHECK(hipStreamDestroy(prod[p]));
    HIP_CHECK(hipStreamDestroy(cons[p]));
  }
}

/**
 * End doxygen group StreamTest.
 * @}
 */

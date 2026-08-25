/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipSharedQueueAnyOrderOverlap hipSharedQueueAnyOrderOverlap
 * @{
 * @ingroup StreamTest
 * Correctness tests for DEBUG_CLR_AQL_BARRIER_OPT. Each test asserts a property that
 * must hold with the flag OFF and ON; run with the flag set (optionally
 * GPU_MAX_HW_QUEUES=1 to force one shared ring) to exercise the ON path. Correctness is
 * checked via observable ordering, not by inspecting packets.
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

// Long-running write then a dependent add on the same stream (intra-stream RAW hazard).
__global__ void SpinSetKernel(int* buf, int val, long long spin) {
#if HT_NVIDIA
  long long s = clock64();
  while ((clock64() - s) < spin) {}
#else
  long long s = clock_function();
  while (clock_function() - s < spin) {}
#endif
  buf[0] = val;
}
__global__ void AddDeltaKernel(int* buf, int delta) { buf[0] += delta; }
__global__ void TouchKernel(int* buf) { buf[0] += 1; }

// Reads the last byte of a buffer (written last by a large memset/copy predecessor).
__global__ void ReadLastByteKernel(const unsigned char* buf, size_t n, int* seen) {
  *seen = static_cast<int>(buf[n - 1]);
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Oversubscribe the HW queue pool with independent per-stream increment chains
 *    (each on its own buffer slot); every slot must have the exact expected count.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipSharedQueueAnyOrderOverlap_Independence) {
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
 *  - Oversubscribed producer/consumer pairs linked by hipEventRecord/
 *    hipStreamWaitEvent (producer writes late, consumer reads early). The consumer
 *    must never see a stale value.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipSharedQueueAnyOrderOverlap_EventDependencyHonored) {
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
 * Test Description
 * ------------------------
 *  - Guards intra-stream order across the sole-tenant -> shared transition: stream A
 *    writes BASE (long kernel), stream B joins the ring, then A adds DELTA to the same
 *    buffer. If A's second dispatch were misread as a "first dispatch" and cleared it
 *    would race A0, so the sum must equal BASE+DELTA. Fresh streams each iteration.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipSharedQueueAnyOrderOverlap_IntraStreamOrderAcrossActivation) {
  constexpr int kIters = 200;
  constexpr int kBase = 5, kDelta = 7, kExpect = kBase + kDelta;

  int ticks_per_ms = 0;  // rate attribute is in kHz, i.e. ticks per millisecond
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeClockRate, 0));
#else
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
#endif
  if (ticks_per_ms == 0) ticks_per_ms = 1000;
  const long long kSpin = 2LL * ticks_per_ms;  // long first kernel so a racing second is observable

  for (int it = 0; it < kIters; ++it) {
    hipStream_t a, b;
    int *bufA = nullptr, *bufB = nullptr;
    HIP_CHECK(hipStreamCreate(&a));
    HIP_CHECK(hipStreamCreate(&b));
    HIP_CHECK(hipMalloc(&bufA, sizeof(int)));
    HIP_CHECK(hipMalloc(&bufB, sizeof(int)));
    HIP_CHECK(hipMemset(bufA, 0, sizeof(int)));
    HIP_CHECK(hipMemset(bufB, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());

    SpinSetKernel<<<dim3(1), dim3(1), 0, a>>>(bufA, kBase, kSpin);  // A0: sole tenant, long
    TouchKernel<<<dim3(1), dim3(1), 0, b>>>(bufB);                   // B0: joins the shared ring
    AddDeltaKernel<<<dim3(1), dim3(1), 0, a>>>(bufA, kDelta);        // A1: depends on A0
    HIP_CHECK(hipDeviceSynchronize());

    int h = 0;
    HIP_CHECK(hipMemcpy(&h, bufA, sizeof(int), hipMemcpyDeviceToHost));
    INFO("iteration " << it);
    REQUIRE(h == kExpect);

    HIP_CHECK(hipFree(bufA));
    HIP_CHECK(hipFree(bufB));
    HIP_CHECK(hipStreamDestroy(a));
    HIP_CHECK(hipStreamDestroy(b));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Guards ordering against an internal same-stream predecessor: on a shared ring, a
 *    large hipMemsetAsync (its device fillBuffer is a compute dispatch) is followed by
 *    a kernel reading the last byte written. A wrongly cleared barrier would race the
 *    fill and read a stale byte.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipSharedQueueAnyOrderOverlap_InternalPredecessorOrdering) {
  constexpr int kStreams = 16, kReps = 40;
  const size_t n = 32u << 20;  // large fill to widen the race window
  const unsigned char kVal = 0xAB;

  std::vector<hipStream_t> s(kStreams);
  std::vector<unsigned char*> buf(kStreams);
  std::vector<int*> seen(kStreams), warm(kStreams);
  for (int i = 0; i < kStreams; ++i) {
    HIP_CHECK(hipStreamCreate(&s[i]));
    HIP_CHECK(hipMalloc(&buf[i], n));
    HIP_CHECK(hipMalloc(&seen[i], sizeof(int)));
    HIP_CHECK(hipMalloc(&warm[i], sizeof(int)));
    HIP_CHECK(hipMemset(warm[i], 0, sizeof(int)));
  }

  for (int r = 0; r < kReps; ++r) {
    for (int i = 0; i < kStreams; ++i) TouchKernel<<<dim3(1), dim3(1), 0, s[i]>>>(warm[i]);
    HIP_CHECK(hipDeviceSynchronize());  // all streams now co-reside -> ring is shared
    for (int i = 0; i < kStreams; ++i) {
      HIP_CHECK(hipMemsetAsync(buf[i], kVal, n, s[i]));                 // internal predecessor
      ReadLastByteKernel<<<dim3(1), dim3(1), 0, s[i]>>>(buf[i], n, seen[i]);
    }
    for (int i = 0; i < kStreams; ++i) HIP_CHECK(hipStreamSynchronize(s[i]));
    for (int i = 0; i < kStreams; ++i) {
      int h = 0;
      HIP_CHECK(hipMemcpy(&h, seen[i], sizeof(int), hipMemcpyDeviceToHost));
      INFO("rep " << r << " stream " << i);
      REQUIRE(h == static_cast<int>(kVal));
    }
  }

  for (int i = 0; i < kStreams; ++i) {
    HIP_CHECK(hipFree(buf[i]));
    HIP_CHECK(hipFree(seen[i]));
    HIP_CHECK(hipFree(warm[i]));
    HIP_CHECK(hipStreamDestroy(s[i]));
  }
}

/**
 * Test Description
 * ------------------------
 *  - The null/blocking stream (dedicated queue, excluded from the opt) keeps its chain
 *    ordered while many async streams hammer the shared ring; async streams stay correct.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipSharedQueueAnyOrderOverlap_NullStreamOrdering) {
  constexpr int kAsync = 16, kChain = 12;
  std::vector<hipStream_t> a(kAsync);
  for (auto& x : a) HIP_CHECK(hipStreamCreate(&x));
  std::vector<int*> abuf(kAsync);
  int* nbuf = nullptr;
  for (auto& p : abuf) {
    HIP_CHECK(hipMalloc(&p, sizeof(int)));
    HIP_CHECK(hipMemset(p, 0, sizeof(int)));
  }
  HIP_CHECK(hipMalloc(&nbuf, sizeof(int)));
  HIP_CHECK(hipMemset(nbuf, 0, sizeof(int)));

  for (int c = 0; c < kChain; ++c) {
    TouchKernel<<<dim3(1), dim3(1), 0, 0>>>(nbuf);  // null (blocking) stream chain
    for (int i = 0; i < kAsync; ++i) TouchKernel<<<dim3(1), dim3(1), 0, a[i]>>>(abuf[i]);
  }
  HIP_CHECK(hipDeviceSynchronize());

  int hn = 0;
  HIP_CHECK(hipMemcpy(&hn, nbuf, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(hn == kChain);
  for (int i = 0; i < kAsync; ++i) {
    int h = 0;
    HIP_CHECK(hipMemcpy(&h, abuf[i], sizeof(int), hipMemcpyDeviceToHost));
    INFO("async stream " << i);
    REQUIRE(h == kChain);
  }

  HIP_CHECK(hipFree(nbuf));
  for (auto& p : abuf) HIP_CHECK(hipFree(p));
  for (auto& x : a) HIP_CHECK(hipStreamDestroy(x));
}

/**
 * Test Description
 * ------------------------
 *  - HIP graphs must be unaffected: independent same-buffer dependency chains
 *    (oversubscribed), instantiated once and launched many times; the chained sum
 *    proves every intra-graph edge survived.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipSharedQueueAnyOrderOverlap_GraphUnaffected) {
  constexpr int kChains = 16, kDepth = 6, kLaunches = 50;
  std::vector<int*> buf(kChains);
  for (auto& b : buf) {
    HIP_CHECK(hipMalloc(&b, sizeof(int)));
    HIP_CHECK(hipMemset(b, 0, sizeof(int)));
  }

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int one = 1;  // outlives graph instantiation/launch; args are captured at node add time
  for (int c = 0; c < kChains; ++c) {
    hipGraphNode_t prev = nullptr;
    for (int d = 0; d < kDepth; ++d) {
      void* args[] = {&buf[c], &one};
      hipKernelNodeParams np{};
      np.func = reinterpret_cast<void*>(AddDeltaKernel);
      np.gridDim = dim3(1);
      np.blockDim = dim3(1);
      np.sharedMemBytes = 0;
      np.kernelParams = args;
      np.extra = nullptr;
      hipGraphNode_t node;
      hipGraphNode_t* deps = prev ? &prev : nullptr;
      HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps, prev ? 1 : 0, &np));
      prev = node;
    }
  }

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t s;
  HIP_CHECK(hipStreamCreate(&s));
  for (int i = 0; i < kLaunches; ++i) HIP_CHECK(hipGraphLaunch(exec, s));
  HIP_CHECK(hipStreamSynchronize(s));

  for (int c = 0; c < kChains; ++c) {
    int h = 0;
    HIP_CHECK(hipMemcpy(&h, buf[c], sizeof(int), hipMemcpyDeviceToHost));
    INFO("graph chain " << c);
    REQUIRE(h == kLaunches * kDepth);
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(s));
  for (auto& b : buf) HIP_CHECK(hipFree(b));
}

/**
 * Test Description
 * ------------------------
 *  - Barrier-value fence on a shared ring: a hipStreamWaitValue32 consumer must
 *    always observe its hipStreamWriteValue32 producer's payload. Must hold with
 *    DEBUG_CLR_AQL_BARRIER_OPT OFF, ON, and ON with GPU_MAX_HW_QUEUES=1.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 *  - Stream wait-value support (hipDeviceAttributeCanUseStreamWaitValue)
 */
HIP_TEST_CASE(Unit_hipSharedQueueAnyOrderOverlap_StreamWaitValueDependencyHonored) {
  int waitValueSupported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&waitValueSupported, hipDeviceAttributeCanUseStreamWaitValue, 0));
  if (waitValueSupported == 0) {
    HIP_SKIP_TEST("hipStreamWaitValue not supported on this device");
    return;
  }

  // A few pairs oversubscribe the 4-queue pool so fences land on shared rings (functional, not stress).
  constexpr int kPairs = 3;  // 6 streams > default pool (4) -> shared rings
  constexpr int kReps = 5;
  constexpr int kBusyIters = 1000;

  std::vector<hipStream_t> prod(kPairs), cons(kPairs);
  std::vector<uint32_t*> flag(kPairs);
  std::vector<int*> payload(kPairs), seen(kPairs);
  for (int p = 0; p < kPairs; ++p) {
    HIP_CHECK(hipStreamCreate(&prod[p]));
    HIP_CHECK(hipStreamCreate(&cons[p]));
    HIP_CHECK(hipMalloc(&flag[p], sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&payload[p], sizeof(int)));
    HIP_CHECK(hipMalloc(&seen[p], sizeof(int)));
    HIP_CHECK(hipMemset(flag[p], 0, sizeof(uint32_t)));
  }

  // Monotonic flag + Gte wait: a stale prior value can't fire early, so payload != it is a real ordering bug.
  std::vector<int> hSeen(kPairs);
  for (int it = 1; it <= kReps; ++it) {
    for (int p = 0; p < kPairs; ++p) {
      // Producer publishes the payload late (after a spin), then raises the flag to it.
      ProducerKernel<<<dim3(1), dim3(1), 0, prod[p]>>>(payload[p], it, kBusyIters);
      HIP_CHECK(hipStreamWriteValue32(prod[p], flag[p], static_cast<uint32_t>(it), 0));
      // Consumer waits on the flag via a barrier-value fence, then reads the payload.
      HIP_CHECK(hipStreamWaitValue32(cons[p], flag[p], static_cast<uint32_t>(it),
                                     hipStreamWaitValueGte, 0xFFFFFFFF));
      ConsumerKernel<<<dim3(1), dim3(1), 0, cons[p]>>>(payload[p], seen[p]);
    }
    for (int p = 0; p < kPairs; ++p) {
      HIP_CHECK(hipStreamSynchronize(cons[p]));
      HIP_CHECK(hipMemcpy(&hSeen[p], seen[p], sizeof(int), hipMemcpyDeviceToHost));
      INFO("pair " << p << " iteration " << it);
      REQUIRE(hSeen[p] == it);  // consumer must observe the producer's payload
    }
  }

  for (int p = 0; p < kPairs; ++p) {
    HIP_CHECK(hipFree(flag[p]));
    HIP_CHECK(hipFree(payload[p]));
    HIP_CHECK(hipFree(seen[p]));
    HIP_CHECK(hipStreamDestroy(prod[p]));
    HIP_CHECK(hipStreamDestroy(cons[p]));
  }
}

/**
 * End doxygen group StreamTest.
 * @}
 */

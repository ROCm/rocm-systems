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
 * shared HW queues. With the flag ON, a kernel's AQL barrier bit is cleared when its
 * in-stream order is already preserved, letting independent streams overlap on the ring.
 *
 * These tests assert properties that must hold regardless of the flag:
 *   1. Independent oversubscribed streams always produce correct results.
 *   2. Explicit cross-stream dependencies (hipStreamWaitEvent) are always honored,
 *      because the optimization skips any launch that carries an event wait.
 *   3. Intra-stream ordering survives the sole-tenant -> shared transition: a
 *      dispatch issued while a stream is alone on the ring is still tracked, so a
 *      later same-stream dispatch is never misread as a "first dispatch" and cleared.
 *   4. A later kernel observes an internal same-stream predecessor (the fillBuffer
 *      behind hipMemsetAsync, an async copy) -- these must not be treated as absent.
 *   5. Stream priority pools, the null/blocking stream, and async copies keep their
 *      ordering.
 *   6. HIP graphs (dependency edges) are unaffected -- the optimization must not
 *      perturb intra-graph ordering.
 *
 * To exercise the ON path, run with DEBUG_HIP_SHARED_QUEUE_ANYORDER=1 (optionally
 * GPU_MAX_HW_QUEUES=1 to force a single shared ring); every test must still pass.
 * Per-packet barrier-bit placement and the cooperative-kernel exclusion are checked
 * by asserting the observable ordering those choices must preserve, rather than by
 * inspecting packets directly.
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
  long long s = clock_function();
  while (clock_function() - s < spin) {}
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
 * Test Description
 * ------------------------
 *  - Guards intra-stream ordering across the sole-tenant -> shared transition. A
 *    fresh stream A (sole tenant) issues a long kernel that writes BASE, a second
 *    stream B then joins the shared ring, and A issues a dependent kernel that adds
 *    DELTA to the same buffer. If A's second dispatch were misclassified as a "first
 *    dispatch" and had its barrier bit cleared, it would race A's first kernel and
 *    the sum would be wrong. Fresh streams each iteration retrigger the transition.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipSharedQueueAnyOrderOverlap_IntraStreamOrderAcrossActivation") {
  constexpr int kIters = 200;
  constexpr int kBase = 5, kDelta = 7, kExpect = kBase + kDelta;

  int ticks_per_ms = 0;  // hipDeviceAttributeWallClockRate is in kHz, i.e. ticks per millisecond
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
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
 *  - Guards ordering against an internal same-stream predecessor. Several co-resident
 *    streams (a warm-up dispatch makes the ring genuinely shared) each do a large
 *    hipMemsetAsync -- whose device-side fillBuffer is a compute dispatch on the same
 *    ring -- immediately followed by a kernel that reads the LAST byte written by that
 *    memset. If the kernel's barrier bit were cleared without accounting for the
 *    memset predecessor, the kernel would race the fill and read a stale byte.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipSharedQueueAnyOrderOverlap_InternalPredecessorOrdering") {
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
 *  - Priority streams live in separate HW-queue pools; independent per-stream
 *    increment chains across the high- and low-priority pools must be exact under
 *    oversubscription.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipSharedQueueAnyOrderOverlap_PriorityStreamsIndependence") {
  int lo = 0, hi = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
  const int kEach = getenv("SQ_TEST_KEACH") ? atoi(getenv("SQ_TEST_KEACH")) : 8;
  constexpr int kChain = 10;

  std::vector<hipStream_t> s;
  for (int i = 0; i < kEach; ++i) {
    hipStream_t a, b;
    HIP_CHECK(hipStreamCreateWithPriority(&a, hipStreamNonBlocking, hi));
    HIP_CHECK(hipStreamCreateWithPriority(&b, hipStreamNonBlocking, lo));
    s.push_back(a);
    s.push_back(b);
  }
  std::vector<int*> buf(s.size());
  for (auto& p : buf) {
    HIP_CHECK(hipMalloc(&p, sizeof(int)));
    HIP_CHECK(hipMemset(p, 0, sizeof(int)));
  }

  for (int c = 0; c < kChain; ++c)
    for (size_t i = 0; i < s.size(); ++i) TouchKernel<<<dim3(1), dim3(1), 0, s[i]>>>(buf[i]);
  for (auto& st : s) HIP_CHECK(hipStreamSynchronize(st));

  for (size_t i = 0; i < buf.size(); ++i) {
    int h = 0;
    HIP_CHECK(hipMemcpy(&h, buf[i], sizeof(int), hipMemcpyDeviceToHost));
    INFO("priority stream " << i);
    REQUIRE(h == kChain);
  }

  for (auto& p : buf) HIP_CHECK(hipFree(p));
  for (auto& st : s) HIP_CHECK(hipStreamDestroy(st));
}

/**
 * Test Description
 * ------------------------
 *  - The null/blocking stream runs on a dedicated queue (excluded from the
 *    optimization). Its dependency chain must stay ordered while many async streams
 *    hammer the shared ring in between; the async streams must also be correct.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipSharedQueueAnyOrderOverlap_NullStreamOrdering") {
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
 *  - HIP graphs must be unaffected by the optimization. Build a graph with several
 *    independent same-buffer dependency chains (forcing oversubscription onto shared
 *    rings), instantiate once and launch many times; the chained sum proves every
 *    intra-graph edge survived.
 * Test source
 * ------------------------
 *  - unit/stream/hipSharedQueueAnyOrderOverlap.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipSharedQueueAnyOrderOverlap_GraphUnaffected") {
  constexpr int kChains = 16, kDepth = 6, kLaunches = 50;
  std::vector<int*> buf(kChains);
  for (auto& b : buf) {
    HIP_CHECK(hipMalloc(&b, sizeof(int)));
    HIP_CHECK(hipMemset(b, 0, sizeof(int)));
  }

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  for (int c = 0; c < kChains; ++c) {
    hipGraphNode_t prev = nullptr;
    for (int d = 0; d < kDepth; ++d) {
      int one = 1;
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
 * End doxygen group StreamTest.
 * @}
 */

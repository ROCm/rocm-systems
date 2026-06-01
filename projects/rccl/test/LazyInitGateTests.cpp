/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Regression coverage for the lazy kernel-init CAS gate that lives in
 * src/enqueue.cc::lazyKernelInitOnce. We cannot link the production helper
 * directly (it pulls in HIP runtime + the rest of enqueue.cc), so this test
 * mirrors the gate's state machine and memory ordering inline in
 * testGateRun() below. If the production gate's contract changes, this
 * mirror MUST be updated in lockstep -- otherwise these tests still pass
 * while the real code drifts away from the spec they pin.
 *
 * Contract under test (must match src/enqueue.cc::lazyKernelInitOnce):
 *   - Per-(process,device) std::atomic<int> gate with four states:
 *       LAZY_UNINIT (0), LAZY_INITIALIZING (1), LAZY_DONE (2), LAZY_FAILED (3).
 *   - Exactly one caller wins the UNINIT->INITIALIZING CAS and runs initFn();
 *     all other callers wait until LAZY_DONE / LAZY_FAILED is observed.
 *   - On owner failure, the owner first writes the failing ncclResult_t to a
 *     per-device err side-table and THEN publishes LAZY_FAILED on the gate
 *     (release). Waiters acquire-load LAZY_FAILED and return the same code,
 *     so every rank sees the first-cause error instead of a generic
 *     ncclSystemError.
 *   - DONE / FAILED short-circuit fresh callers without re-running initFn
 *     (this is the common-case fast path after first launch).
 *   - A would-be owner that cannot legally run init right now (in production:
 *     inside a stream capture) is detected AFTER winning the CAS. It then
 *     releases ownership back to LAZY_UNINIT and returns the precondition
 *     error. Any concurrent waiter sees that release and retries the CAS
 *     itself, so a legal caller still completes init. Threads that never
 *     became the owner never observe a precondition-driven error.
 ************************************************************************/

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <sched.h>

#include <gtest/gtest.h>
#include <rccl/rccl.h>

namespace RcclUnitTesting {
namespace {

enum class LazyInitState : int {
  LAZY_UNINIT       = 0,
  LAZY_INITIALIZING = 1,
  LAZY_DONE         = 2,
  LAZY_FAILED       = 3,
};

// Inline mirror of src/enqueue.cc::lazyKernelInitOnce minus the HIP-specific
// pieces (INFO/WARN logging, clockNano). Behaviorally identical for the CAS
// race, owner/waiter handoff, post-CAS precondition with owner-abort retry,
// and failure propagation this file pins.
//
// initFn  - test-supplied stand-in for ncclInitKernelsForDevice.
// errSlot - mirror of g_rcclLazyKernelInitErr.
// ownerPrecondition(err_out) - mirror of the stream-capture check; the
//   thread invokes it ONLY AFTER it has won the UNINIT->INITIALIZING CAS.
//   Return false to abort ownership: the gate is released back to
//   LAZY_UNINIT and the caller returns err_out. Threads that did not win
//   the CAS never invoke it -- that's the property we pin.
// waiterReachedWaitLoop - optional instrumentation hook. When non-null, a
//   thread that did NOT win the CAS sets it to 1 on entering the spin
//   wait. Lets multi-thread tests deterministically synchronize on "the
//   waiter is parked" without resorting to sleep_for-based handshakes.
static ncclResult_t testGateRun(
    std::atomic<LazyInitState>& gate,
    std::atomic<int>& errSlot,
    std::function<ncclResult_t()> initFn,
    std::function<bool(ncclResult_t&)> ownerPrecondition =
        [](ncclResult_t&){ return true; },
    std::atomic<int>* waiterReachedWaitLoop = nullptr) {
  for (;;) {
    LazyInitState s = gate.load(std::memory_order_acquire);
    if (s == LazyInitState::LAZY_DONE) return ncclSuccess;
    if (s == LazyInitState::LAZY_FAILED) {
      return (ncclResult_t)errSlot.load(std::memory_order_relaxed);
    }

    if (s == LazyInitState::LAZY_UNINIT) {
      LazyInitState expected = LazyInitState::LAZY_UNINIT;
      if (gate.compare_exchange_strong(expected, LazyInitState::LAZY_INITIALIZING,
                                       std::memory_order_acq_rel,
                                       std::memory_order_acquire)) {
        ncclResult_t pre = ncclSuccess;
        if (!ownerPrecondition(pre)) {
          gate.store(LazyInitState::LAZY_UNINIT, std::memory_order_release);
          return pre;
        }
        ncclResult_t r = initFn();
        if (r != ncclSuccess) {
          errSlot.store((int)r, std::memory_order_relaxed);
          gate.store(LazyInitState::LAZY_FAILED, std::memory_order_release);
          return r;
        }
        gate.store(LazyInitState::LAZY_DONE, std::memory_order_release);
        return ncclSuccess;
      }
    }

    LazyInitState w;
    if (waiterReachedWaitLoop) {
      waiterReachedWaitLoop->store(1, std::memory_order_release);
    }
    while ((w = gate.load(std::memory_order_acquire)) == LazyInitState::LAZY_INITIALIZING) {
      sched_yield();
    }
    if (w == LazyInitState::LAZY_DONE)   return ncclSuccess;
    if (w == LazyInitState::LAZY_FAILED) return (ncclResult_t)errSlot.load(std::memory_order_relaxed);
    // w == LAZY_UNINIT: owner aborted, retry the CAS ourselves.
  }
}

// ---------------------------------------------------------------------------
// Fine-grained lazy init mirrors (src/enqueue.cc fine-grained path).
//
// The fine-grained path keeps the SAME per-slot gate contract pinned by
// testGateRun above, but (a) uses a separate gate per generic-kernel index
// instead of one per device, and (b) raises cudaLimitStackSize incrementally
// as each kernel's stack size is discovered. The two helpers below mirror the
// only NEW logic those add on top of the gate: fn->index lookup and the
// serialized running-max stack update.
// ---------------------------------------------------------------------------

// Mirror of src/enqueue.cc::lazyKernelGenericIndex: linear search of fn in the
// generic ncclKerns[] table; -1 when fn is not a generic kernel (in production
// a symmetric-memory kernel, which falls back to the device-wide gate).
static int testGenericIndex(void* const* kerns, int count, void* fn) {
  for (int k = 0; k < count; k++) {
    if (kerns[k] == fn) return k;
  }
  return -1;
}

// Mirror of the incremental stack-limit update in lazyKernelInitFnOnce: under a
// lock, raise the running max and only then "apply" the limit. lastApplied
// records the value handed to cudaDeviceSetLimit so tests can assert the limit
// is monotonic and is never lowered by a racing smaller value.
struct StackLimitMirror {
  std::mutex mu;
  size_t maxStack = 0;
  size_t lastApplied = 0;
  std::atomic<int> applyCalls{0};
  void offer(size_t stack) {
    std::lock_guard<std::mutex> lk(mu);
    if (stack > maxStack) {
      maxStack = stack;
      lastApplied = stack;
      applyCalls.fetch_add(1, std::memory_order_relaxed);
    }
  }
};

} // namespace

// Single-threaded happy path: UNINIT -> INITIALIZING -> DONE, initFn fires
// exactly once, gate ends in LAZY_DONE.
TEST(LazyInitGateTests, SingleThread_HappyPath_RunsInitOnce) {
  std::atomic<LazyInitState> gate{LazyInitState::LAZY_UNINIT};
  std::atomic<int> err{(int)ncclSuccess};
  int calls = 0;
  ncclResult_t r = testGateRun(gate, err, [&]{ ++calls; return ncclSuccess; });
  EXPECT_EQ(r, ncclSuccess);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(gate.load(), LazyInitState::LAZY_DONE);
}

// Re-entry after success must NOT run initFn again -- this is the steady-state
// fast path that fires on every launch past the first.
TEST(LazyInitGateTests, AlreadyDone_IsFastPath_NoReinit) {
  std::atomic<LazyInitState> gate{LazyInitState::LAZY_DONE};
  std::atomic<int> err{(int)ncclSuccess};
  int calls = 0;
  ncclResult_t r = testGateRun(gate, err, [&]{ ++calls; return ncclSuccess; });
  EXPECT_EQ(r, ncclSuccess);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(gate.load(), LazyInitState::LAZY_DONE);
}

// Once the gate has been poisoned (owner failed), every later caller must
// receive the owner's exact ncclResult_t WITHOUT re-running initFn. This is
// the contract that keeps a misconfigured device from hanging every
// subsequent launch and from losing the first-cause error code.
TEST(LazyInitGateTests, AlreadyFailed_IsFastPath_NoReinit) {
  std::atomic<LazyInitState> gate{LazyInitState::LAZY_FAILED};
  std::atomic<int> err{(int)ncclUnhandledCudaError};
  int calls = 0;
  ncclResult_t r = testGateRun(gate, err, [&]{ ++calls; return ncclSuccess; });
  EXPECT_EQ(r, ncclUnhandledCudaError);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(gate.load(), LazyInitState::LAZY_FAILED);
}

// Owner failure must publish LAZY_FAILED, store the failing code in errSlot,
// and surface initFn's exact error to the owner and any later caller.
TEST(LazyInitGateTests, OwnerFailure_PoisonsGate_PropagatesError) {
  std::atomic<LazyInitState> gate{LazyInitState::LAZY_UNINIT};
  std::atomic<int> err{(int)ncclSuccess};
  int calls = 0;
  ncclResult_t r = testGateRun(gate, err, [&]{
    ++calls;
    return ncclInternalError;
  });
  EXPECT_EQ(r, ncclInternalError);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(gate.load(), LazyInitState::LAZY_FAILED);
  EXPECT_EQ((ncclResult_t)err.load(), ncclInternalError);

  // A second caller must not retry the failed init -- it gets the cached
  // first-cause code from the err side-table.
  ncclResult_t r2 = testGateRun(gate, err, [&]{ ++calls; return ncclSuccess; });
  EXPECT_EQ(r2, ncclInternalError);
  EXPECT_EQ(calls, 1);
}

// The headline race: N threads hit a fresh gate in parallel; exactly one
// must win the CAS and run initFn, and every thread must return ncclSuccess.
// We hold initFn artificially long (10ms) so contenders are still racing on
// the gate when the owner publishes LAZY_DONE -- this exercises the waiter
// path, not just the steady-state fast path.
TEST(LazyInitGateTests, MultiThread_OnlyOneOwnerRunsInit_AllSucceed) {
  constexpr int kThreads = 16;
  std::atomic<LazyInitState> gate{LazyInitState::LAZY_UNINIT};
  std::atomic<int> err{(int)ncclSuccess};
  std::atomic<int> initCalls{0};
  std::atomic<int> startGate{0};
  std::vector<ncclResult_t> results(kThreads, ncclInternalError);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i]{
      while (startGate.load(std::memory_order_acquire) == 0) sched_yield();
      results[i] = testGateRun(gate, err, [&]{
        initCalls.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return ncclSuccess;
      });
    });
  }

  startGate.store(1, std::memory_order_release);
  for (auto& t : threads) t.join();

  EXPECT_EQ(initCalls.load(), 1) << "initFn must fire exactly once across all racing threads";
  EXPECT_EQ(gate.load(), LazyInitState::LAZY_DONE);
  for (int i = 0; i < kThreads; ++i) {
    EXPECT_EQ(results[i], ncclSuccess) << "thread " << i << " saw a non-success result";
  }
}

// Same multi-thread race, but the owner FAILS. Every thread (owner and
// waiters) must return the SAME ncclResult_t the owner produced -- no
// generic ncclSystemError, no hangs, no silent re-run of initFn.
TEST(LazyInitGateTests, MultiThread_OwnerFails_AllThreadsGetSameError) {
  constexpr int kThreads = 16;
  constexpr ncclResult_t kOwnerErr = ncclUnhandledCudaError;
  std::atomic<LazyInitState> gate{LazyInitState::LAZY_UNINIT};
  std::atomic<int> err{(int)ncclSuccess};
  std::atomic<int> initCalls{0};
  std::atomic<int> startGate{0};
  std::vector<ncclResult_t> results(kThreads, ncclInternalError);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i]{
      while (startGate.load(std::memory_order_acquire) == 0) sched_yield();
      results[i] = testGateRun(gate, err, [&]{
        initCalls.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return kOwnerErr;
      });
    });
  }

  startGate.store(1, std::memory_order_release);
  for (auto& t : threads) t.join();

  EXPECT_EQ(initCalls.load(), 1) << "initFn must fire exactly once even when it fails";
  EXPECT_EQ(gate.load(), LazyInitState::LAZY_FAILED);
  EXPECT_EQ((ncclResult_t)err.load(), kOwnerErr);
  for (int i = 0; i < kThreads; ++i) {
    EXPECT_EQ(results[i], kOwnerErr) << "thread " << i << " did not receive the owner's first-cause error";
  }
}

// Post-CAS precondition (in production: stream capture) must not poison
// the gate on failure. A later caller for whom the precondition holds wins
// a fresh CAS, runs init, and succeeds -- proving the precondition path is
// retryable.
TEST(LazyInitGateTests, OwnerPrecondition_FailsRetryable_NoGatePoison) {
  std::atomic<LazyInitState> gate{LazyInitState::LAZY_UNINIT};
  std::atomic<int> err{(int)ncclSuccess};
  int calls = 0;

  ncclResult_t r1 = testGateRun(gate, err,
      [&]{ ++calls; return ncclSuccess; },
      [](ncclResult_t& out){ out = ncclInvalidUsage; return false; });
  EXPECT_EQ(r1, ncclInvalidUsage);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(gate.load(), LazyInitState::LAZY_UNINIT)
      << "owner-abort must release the gate back to LAZY_UNINIT for retry";

  ncclResult_t r2 = testGateRun(gate, err,
      [&]{ ++calls; return ncclSuccess; });
  EXPECT_EQ(r2, ncclSuccess);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(gate.load(), LazyInitState::LAZY_DONE);
}

// Owner aborts the precondition WHILE a second thread is already in the
// waiter loop. The waiter must observe the LAZY_UNINIT release, retry the
// CAS itself, run init, and return ncclSuccess. This pins the "waiter
// retries on owner abort" branch -- a hang or a generic error here would
// mean the state machine cannot recover from a legitimately-failing
// precondition without dropping concurrent callers.
TEST(LazyInitGateTests, MultiThread_OwnerAborts_WaiterRetriesAndSucceeds) {
  std::atomic<LazyInitState> gate{LazyInitState::LAZY_UNINIT};
  std::atomic<int> err{(int)ncclSuccess};
  std::atomic<int> initCalls{0};
  std::atomic<int> ownerEnteredPrecondition{0};
  std::atomic<int> releaseOwner{0};
  std::atomic<int> retrierInWaitLoop{0};

  std::thread owner([&]{
    ncclResult_t r = testGateRun(gate, err,
        [&]{ initCalls.fetch_add(1); return ncclSuccess; },
        [&](ncclResult_t& out){
          // Park inside the precondition long enough for the second thread
          // to enter the waiter loop with s == LAZY_INITIALIZING. Then
          // abort -- this releases the gate back to LAZY_UNINIT.
          ownerEnteredPrecondition.store(1, std::memory_order_release);
          while (releaseOwner.load(std::memory_order_acquire) == 0) sched_yield();
          out = ncclInvalidUsage;
          return false;
        });
    EXPECT_EQ(r, ncclInvalidUsage);
  });

  while (ownerEnteredPrecondition.load(std::memory_order_acquire) == 0) sched_yield();

  // Second thread: precondition passes, but it arrives while the first
  // thread already owns the gate. It must wait, observe the owner's
  // LAZY_UNINIT release, retry the CAS itself, run init, and succeed.
  std::thread retrier([&]{
    ncclResult_t r = testGateRun(gate, err,
        [&]{ initCalls.fetch_add(1); return ncclSuccess; },
        [](ncclResult_t&){ return true; },
        &retrierInWaitLoop);
    EXPECT_EQ(r, ncclSuccess) << "waiter must retry CAS on owner-abort and succeed";
  });

  // Deterministically wait until the retrier is actually parked in the
  // wait loop, then unblock the owner. Avoids relying on sleep_for, which
  // is flaky under CI load.
  while (retrierInWaitLoop.load(std::memory_order_acquire) == 0) sched_yield();
  releaseOwner.store(1, std::memory_order_release);

  owner.join();
  retrier.join();

  EXPECT_EQ(initCalls.load(), 1) << "init must fire exactly once across owner+retrier";
  EXPECT_EQ(gate.load(), LazyInitState::LAZY_DONE);
}

// fn->index lookup: known generic kernels map to their slot, everything else
// (symmetric kernels, null) maps to -1 so the caller falls back to the
// device-wide gate.
TEST(LazyInitFineGrainedTests, GenericIndex_MapsKnownFns_RejectsUnknown) {
  int a, b, c, other;
  void* kerns[3] = {&a, &b, &c};
  EXPECT_EQ(testGenericIndex(kerns, 3, &a), 0);
  EXPECT_EQ(testGenericIndex(kerns, 3, &b), 1);
  EXPECT_EQ(testGenericIndex(kerns, 3, &c), 2);
  EXPECT_EQ(testGenericIndex(kerns, 3, &other), -1) << "unknown (e.g. symmetric) fn must map to -1";
  EXPECT_EQ(testGenericIndex(kerns, 3, nullptr), -1);
}

// Per-fn gates are independent: initializing the kernel one comm uses must not
// flip the gates for the kernels it never launches, and the inited slot still
// has the steady-state no-reinit fast path.
TEST(LazyInitFineGrainedTests, PerFnGates_AreIndependent_EachInitsOnce) {
  constexpr int K = 3;
  std::array<std::atomic<LazyInitState>, K> gate;
  std::array<std::atomic<int>, K> err;
  for (int k = 0; k < K; ++k) {
    gate[k].store(LazyInitState::LAZY_UNINIT);
    err[k].store((int)ncclSuccess);
  }
  int calls[K] = {0, 0, 0};

  // Only the kernel at index 1 is launched first.
  EXPECT_EQ(testGateRun(gate[1], err[1], [&]{ ++calls[1]; return ncclSuccess; }), ncclSuccess);
  EXPECT_EQ(gate[1].load(), LazyInitState::LAZY_DONE);
  EXPECT_EQ(gate[0].load(), LazyInitState::LAZY_UNINIT) << "untouched kernel must stay UNINIT";
  EXPECT_EQ(gate[2].load(), LazyInitState::LAZY_UNINIT) << "untouched kernel must stay UNINIT";
  EXPECT_EQ(calls[0], 0);
  EXPECT_EQ(calls[2], 0);

  // Re-launch of the same kernel takes the fast path, no second init.
  EXPECT_EQ(testGateRun(gate[1], err[1], [&]{ ++calls[1]; return ncclSuccess; }), ncclSuccess);
  EXPECT_EQ(calls[1], 1) << "already-inited kernel must not reinit";

  // A different kernel inits independently exactly once.
  EXPECT_EQ(testGateRun(gate[0], err[0], [&]{ ++calls[0]; return ncclSuccess; }), ncclSuccess);
  EXPECT_EQ(calls[0], 1);
  EXPECT_EQ(gate[0].load(), LazyInitState::LAZY_DONE);
}

// Incremental stack-limit update raises the device limit monotonically and
// never re-applies for a smaller value.
TEST(LazyInitFineGrainedTests, StackLimit_RaisesMonotonically_NeverLowers) {
  StackLimitMirror m;
  m.offer(512);
  EXPECT_EQ(m.maxStack, 512u);
  EXPECT_EQ(m.lastApplied, 512u);
  m.offer(256); // smaller: must not lower or re-apply
  EXPECT_EQ(m.maxStack, 512u);
  EXPECT_EQ(m.lastApplied, 512u);
  m.offer(1024);
  EXPECT_EQ(m.maxStack, 1024u);
  EXPECT_EQ(m.lastApplied, 1024u);
  EXPECT_EQ(m.applyCalls.load(), 2) << "limit applied only when it actually grows (512, 1024)";
}

// Concurrent stack offers in arbitrary order must converge to the true max as
// the last-applied limit -- the property that keeps a racing smaller value
// from being the final cudaLimitStackSize.
TEST(LazyInitFineGrainedTests, StackLimit_ConcurrentOffers_FinalIsTrueMax) {
  StackLimitMirror m;
  const std::vector<size_t> vals = {128, 2048, 512, 1024, 256, 2048, 64};
  std::atomic<int> start{0};
  std::vector<std::thread> ts;
  ts.reserve(vals.size());
  for (size_t v : vals) {
    ts.emplace_back([&, v]{
      while (start.load(std::memory_order_acquire) == 0) sched_yield();
      m.offer(v);
    });
  }
  start.store(1, std::memory_order_release);
  for (auto& t : ts) t.join();

  EXPECT_EQ(m.maxStack, 2048u);
  EXPECT_EQ(m.lastApplied, 2048u) << "final applied limit must equal the true max regardless of order";
}

} // namespace RcclUnitTesting

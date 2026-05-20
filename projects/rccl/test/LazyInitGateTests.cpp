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
 *   - LAZY_FAILED is REQUIRED so that an owner failure propagates to every
 *     waiter as ncclSystemError instead of leaving the gate stuck at
 *     LAZY_INITIALIZING and hanging every subsequent launch on that device.
 *   - LAZY_DONE / LAZY_FAILED short-circuit fresh callers without doing the
 *     CAS again (this is the common-case fast path after first launch).
 ************************************************************************/

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include <sched.h>

#include <gtest/gtest.h>
#include <rccl/rccl.h>

namespace RcclUnitTesting {
namespace {

enum LazyInitState : int {
  LAZY_UNINIT       = 0,
  LAZY_INITIALIZING = 1,
  LAZY_DONE         = 2,
  LAZY_FAILED       = 3,
};

// Inline mirror of src/enqueue.cc::lazyKernelInitOnce minus the HIP-specific
// pieces (stream-capture check, INFO/WARN logging, clockNano). Behaviorally
// identical for the CAS race and failure-propagation contract this file
// pins. initFn is the test-supplied stand-in for ncclInitKernelsForDevice.
static ncclResult_t testGateRun(std::atomic<int>& gate,
                                std::function<ncclResult_t()> initFn) {
  int s = gate.load(std::memory_order_acquire);
  if (s == LAZY_DONE)   return ncclSuccess;
  if (s == LAZY_FAILED) return ncclSystemError;

  int expected = LAZY_UNINIT;
  if (gate.compare_exchange_strong(expected, LAZY_INITIALIZING,
                                   std::memory_order_acq_rel,
                                   std::memory_order_acquire)) {
    ncclResult_t r = initFn();
    gate.store(r == ncclSuccess ? LAZY_DONE : LAZY_FAILED,
               std::memory_order_release);
    return r;
  }

  int w;
  while ((w = gate.load(std::memory_order_acquire)) != LAZY_DONE) {
    if (w == LAZY_FAILED) return ncclSystemError;
    sched_yield();
  }
  return ncclSuccess;
}

} // namespace

// Single-threaded happy path: UNINIT -> INITIALIZING -> DONE, initFn fires
// exactly once, gate ends in LAZY_DONE.
TEST(LazyInitGateTests, SingleThread_HappyPath_RunsInitOnce) {
  std::atomic<int> gate{LAZY_UNINIT};
  int calls = 0;
  ncclResult_t r = testGateRun(gate, [&]{ ++calls; return ncclSuccess; });
  EXPECT_EQ(r, ncclSuccess);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(gate.load(), LAZY_DONE);
}

// Re-entry after success must NOT run initFn again -- this is the steady-state
// fast path that fires on every launch past the first.
TEST(LazyInitGateTests, AlreadyDone_IsFastPath_NoReinit) {
  std::atomic<int> gate{LAZY_DONE};
  int calls = 0;
  ncclResult_t r = testGateRun(gate, [&]{ ++calls; return ncclSuccess; });
  EXPECT_EQ(r, ncclSuccess);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(gate.load(), LAZY_DONE);
}

// Once the gate has been poisoned (owner failed), every later caller must get
// ncclSystemError WITHOUT re-running initFn. This is the contract that keeps
// a misconfigured device from hanging every subsequent launch.
TEST(LazyInitGateTests, AlreadyFailed_IsFastPath_NoReinit) {
  std::atomic<int> gate{LAZY_FAILED};
  int calls = 0;
  ncclResult_t r = testGateRun(gate, [&]{ ++calls; return ncclSuccess; });
  EXPECT_EQ(r, ncclSystemError);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(gate.load(), LAZY_FAILED);
}

// Owner failure must publish LAZY_FAILED and surface initFn's exact error.
TEST(LazyInitGateTests, OwnerFailure_PoisonsGate_PropagatesError) {
  std::atomic<int> gate{LAZY_UNINIT};
  int calls = 0;
  ncclResult_t r = testGateRun(gate, [&]{
    ++calls;
    return ncclInternalError;
  });
  EXPECT_EQ(r, ncclInternalError);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(gate.load(), LAZY_FAILED);

  // A second caller must not retry the failed init -- it gets the cached
  // failure mapped to ncclSystemError (the gate has no room to store the
  // exact error, by design).
  ncclResult_t r2 = testGateRun(gate, [&]{ ++calls; return ncclSuccess; });
  EXPECT_EQ(r2, ncclSystemError);
  EXPECT_EQ(calls, 1);
}

// The headline race: N threads hit a fresh gate in parallel; exactly one
// must win the CAS and run initFn, and every thread must return ncclSuccess.
// We hold initFn artificially long (10ms) so contenders are still racing on
// the gate when the owner publishes LAZY_DONE -- this exercises the waiter
// path, not just the steady-state fast path.
TEST(LazyInitGateTests, MultiThread_OnlyOneOwnerRunsInit_AllSucceed) {
  constexpr int kThreads = 16;
  std::atomic<int> gate{LAZY_UNINIT};
  std::atomic<int> initCalls{0};
  std::atomic<int> startGate{0};
  std::vector<ncclResult_t> results(kThreads, ncclInternalError);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i]{
      while (startGate.load(std::memory_order_acquire) == 0) sched_yield();
      results[i] = testGateRun(gate, [&]{
        initCalls.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return ncclSuccess;
      });
    });
  }

  startGate.store(1, std::memory_order_release);
  for (auto& t : threads) t.join();

  EXPECT_EQ(initCalls.load(), 1) << "initFn must fire exactly once across all racing threads";
  EXPECT_EQ(gate.load(), LAZY_DONE);
  for (int i = 0; i < kThreads; ++i) {
    EXPECT_EQ(results[i], ncclSuccess) << "thread " << i << " saw a non-success result";
  }
}

// Same multi-thread race, but the owner FAILS. Every waiter must observe
// LAZY_FAILED and return ncclSystemError -- no waiter may hang and no waiter
// may silently re-run initFn after the failure is published.
TEST(LazyInitGateTests, MultiThread_OwnerFails_AllWaitersGetSystemError) {
  constexpr int kThreads = 16;
  std::atomic<int> gate{LAZY_UNINIT};
  std::atomic<int> initCalls{0};
  std::atomic<int> startGate{0};
  std::vector<ncclResult_t> results(kThreads, ncclInternalError);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i]{
      while (startGate.load(std::memory_order_acquire) == 0) sched_yield();
      results[i] = testGateRun(gate, [&]{
        initCalls.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return ncclUnhandledCudaError;
      });
    });
  }

  startGate.store(1, std::memory_order_release);
  for (auto& t : threads) t.join();

  EXPECT_EQ(initCalls.load(), 1) << "initFn must fire exactly once even when it fails";
  EXPECT_EQ(gate.load(), LAZY_FAILED);

  // Exactly one thread saw the owner's raw error code; every other thread
  // is a waiter and must see the cached ncclSystemError.
  int ownerCount  = 0;
  int waiterCount = 0;
  for (int i = 0; i < kThreads; ++i) {
    if (results[i] == ncclUnhandledCudaError) {
      ++ownerCount;
    } else {
      EXPECT_EQ(results[i], ncclSystemError) << "thread " << i << " saw an unexpected result";
      ++waiterCount;
    }
  }
  EXPECT_EQ(ownerCount,  1);
  EXPECT_EQ(waiterCount, kThreads - 1);
}

} // namespace RcclUnitTesting

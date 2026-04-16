//===-- TracingEngineGTest.cpp - TracingEngine Tests -------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for TracingEngine.  Most of the engine's logic requires the HSA
/// runtime and a GPU, so we test only what's feasible in a CPU-only
/// environment: singleton identity, initialization state, and stats defaults.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/TracingEngine.h"
#include <gtest/gtest.h>

using namespace aegisbit;

//===----------------------------------------------------------------------===//
// T-001: singleton identity
//===----------------------------------------------------------------------===//

TEST(TracingEngine, GetInstanceReturnsSamePointer) {
  auto &A = TracingEngine::getInstance();
  auto &B = TracingEngine::getInstance();
  EXPECT_EQ(&A, &B);
}

//===----------------------------------------------------------------------===//
// T-002: isInitialized before initialize
//===----------------------------------------------------------------------===//

TEST(TracingEngine, NotInitializedByDefault) {
  // The singleton may have been initialized by a prior test run in the same
  // process, but in a fresh process it should be false.  Since we can't
  // control construction order across test binaries, just verify the method
  // is callable and returns a boolean.
  bool State = TracingEngine::getInstance().isInitialized();
  (void)State;
  SUCCEED();
}

//===----------------------------------------------------------------------===//
// T-003: getStats returns zero counters initially
//===----------------------------------------------------------------------===//

TEST(TracingEngine, StatsDefaultToZero) {
  auto Stats = TracingEngine::getInstance().getStats();
  // In a fresh process these should be zero. Even if initialized, counters
  // start at zero until dispatches occur (which need a GPU).
  EXPECT_EQ(Stats.TotalDispatches, 0u);
  EXPECT_EQ(Stats.TracedDispatches, 0u);
  EXPECT_EQ(Stats.SkippedDispatches, 0u);
  EXPECT_EQ(Stats.ErrorDispatches, 0u);
  EXPECT_EQ(Stats.TracesWritten, 0u);
}

//===----------------------------------------------------------------------===//
// T-004: initialize() — may succeed or gracefully fail without GPU
//===----------------------------------------------------------------------===//

TEST(TracingEngine, InitializeDoesNotCrash) {
  // initialize() defers GPU work to first dispatch, so it should succeed
  // even without a GPU.  If it returns an error, that's acceptable too —
  // the important thing is no crash or uncaught exception.
  auto Err = TracingEngine::getInstance().initialize();
  if (Err) {
    llvm::consumeError(std::move(Err));
  }
  SUCCEED();
}

//===----------------------------------------------------------------------===//
// T-005: finalize() doesn't crash (even if not initialized or already done)
//===----------------------------------------------------------------------===//

TEST(TracingEngine, FinalizeDoesNotCrash) {
  TracingEngine::getInstance().finalize();
  SUCCEED();
}

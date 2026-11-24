/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_SIGNAL_EVENT_EXHAUSTION_H_
#define ROCRTST_SUITES_FUNCTIONAL_SIGNAL_EVENT_EXHAUSTION_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// Latency and CPU regression tests for interrupt signals created after the
// process has used up its KFD signal events (KFD_SIGNAL_EVENT_LIMIT, 4096).
//
// Such signals used to get no event at all. An async handler on one of them
// made the async-events thread either spin a full core or poll the batch with
// 20-200 us naps, and a blocked waiter on one of them spun. These tests use up
// the events, then check that handlers and waiters on the signals created
// afterwards are still woken promptly while the process stays idle.
class SignalEventExhaustionTest : public TestBase {
 public:
  SignalEventExhaustionTest();
  virtual ~SignalEventExhaustionTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  // A CPU store to a post-exhaustion signal runs its async handler within
  // tens of microseconds, not after a polling nap.
  void TestAsyncHandlerLatency(void);

  // While such a handler is pending, the process (async-events thread
  // included) is idle instead of polling.
  void TestAsyncHandlerIdleCpu(void);

  // Threads blocked in hsa_signal_wait_scacquire on post-exhaustion signals
  // sleep, and every completion wakes the right one.
  void TestBlockedWaiters(void);

  // GPU (CP end-of-pipe) completions of post-exhaustion signals reach their
  // async handler.
  void TestGpuCompletion(void);

  // Per-signal state shared with the async handler.
  struct Target {
    hsa_signal_t signal{};
    std::atomic<int64_t> start_ns{0};
    std::atomic<int64_t> seen_ns{0};  // 0 while a completion is pending
    std::atomic<bool> stop{false};
  };

 private:
  // Creates interrupt signals until one no longer gets an event of its own
  // (event_id 0, or the event of the previous signal). Returns false if the
  // limit was not reached.
  bool ExhaustEvents();

  // Registers an async handler (LT 1) on a new post-exhaustion signal.
  Target* NewHandlerTarget();

  // Completes target (CPU store, or a GPU barrier packet) and returns the
  // latency in us until its handler or waiter saw it, or -1 after 1 s.
  double Complete(Target* t, bool gpu);

  // Process CPU time as a percentage of wall time over the given interval.
  static double ProcessCpuPercent(int ms);

  std::vector<hsa_signal_t> held_;  // only hold KFD event slots
  std::vector<std::unique_ptr<Target>> targets_;
  std::vector<Target*> handlers_;  // targets with a registered async handler
  Target helper_;                  // created before exhaustion, never completed
  hsa_queue_t* queue_;
  uint32_t first_event_id_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SIGNAL_EVENT_EXHAUSTION_H_

/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_SIGNAL_YIELDING_POLL_H_
#define ROCRTST_SUITES_FUNCTIONAL_SIGNAL_YIELDING_POLL_H_

#include <cstdint>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// Functional tests for BusyWaitSignal::WaitRelaxed with
// HSA_ENABLE_YIELDING_POLL. They wait on a GPU-only signal (a BusyWaitSignal,
// which has no interrupt event) and measure the waiting thread's CPU time and
// wake-up timing, so they exercise the real wait loop rather than only the
// poll_backoff.h helpers.
class SignalYieldingPollTest : public TestBase {
 public:
  // yielding_poll selects the value of HSA_ENABLE_YIELDING_POLL that is set
  // before hsa_init(): true -> "1", false -> unset (the default spin path).
  explicit SignalYieldingPollTest(bool yielding_poll);

  virtual ~SignalYieldingPollTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  // A long wait that never completes spends most of its time asleep, with both
  // wait hints.
  void TestLongWaitYieldsCpu(void);

  // A wait that outlasts the hot window returns close to its timeout instead
  // of up to one nap (kPollNapCeilingUs) late.
  void TestNapHonorsTimeout(void);

  // A completion that lands while the waiter naps is still observed promptly.
  void TestCompletionDuringNap(void);

  // HSA_WAIT_STATE_ACTIVE keeps spinning through its wider hot window while
  // HSA_WAIT_STATE_BLOCKED starts napping after the short one.
  void TestActiveHintWidensHotWindow(void);

  // Without HSA_ENABLE_YIELDING_POLL the loop keeps spinning as before.
  void TestDefaultKeepsSpinning(void);

 private:
  struct WaitSample {
    hsa_signal_value_t value;
    double wall_ms;
    double cpu_ms;
  };

  // Waits on signal_ for "value == 0" with the given timeout and hint and
  // reports the returned value, the wall time and the waiting thread's CPU time.
  WaitSample TimedWait(double timeout_ms, hsa_wait_state_t hint);

  bool yielding_poll_;
  uint64_t timestamp_frequency_;
  hsa_signal_t signal_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SIGNAL_YIELDING_POLL_H_

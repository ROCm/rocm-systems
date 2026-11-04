/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_SIGNAL_HINTS_H_
#define ROCRTST_SUITES_FUNCTIONAL_SIGNAL_HINTS_H_

#include <string>
#include <vector>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// Checks that the CP honors each completion signal handle hint (amd_hsa_signal.h) in the AQL
// packets it processes. Skipped on an agent that does not report
// HSA_AMD_AGENT_INFO_SIGNAL_HINTS_SUPPORTED.
class SignalHintsTest : public TestBase {
 public:
  SignalHintsTest(void);

  virtual ~SignalHintsTest(void);

  virtual void SetUp(void);

  virtual void Run(void);

  virtual void Close(void);

  virtual void DisplayResults(void) const;

  virtual void DisplayTestInfo(void);

  // VALUE_BINARY stores 0, in host and device signal memory; without it the CP decrements
  void ValueBinary(void);

  // TS_ON and TS_OFF override the queue's profiling property; TS_INHERIT follows it
  void Timestamps(void);

  // INTR_NONE leaves a waiter sleeping on the signal's event asleep; INTR_ON_ZERO wakes it only
  // on the update to 0
  void Interrupts(void);

 private:
  // Microseconds from the first of num_updates barrier completions of an interrupt signal,
  // started at num_updates and hinted with hints, until a blocked waiter for a value below
  // num_updates returns
  double WakeDelayUs(uint64_t hints, int num_updates);

  uint64_t TimeoutTicks(uint64_t milliseconds) const;

  hsa_queue_t* profiled_queue_ = nullptr;
  hsa_queue_t* unprofiled_queue_ = nullptr;
  uint64_t timestamp_frequency_ = 0;
  std::vector<std::string> resultList_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SIGNAL_HINTS_H_

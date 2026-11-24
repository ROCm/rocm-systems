/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/
#ifndef ROCRTST_SUITES_PERFORMANCE_SIGNAL_HINT_COST_H_
#define ROCRTST_SUITES_PERFORMANCE_SIGNAL_HINT_COST_H_

#include <string>
#include <vector>

#include "suites/test_common/test_base.h"
#include "common/base_rocr.h"
#include "common/common.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

// Measures the CP time per barrier packet that completes a signal, with each completion signal
// handle hint and with all of them, for signals in host memory, device memory and with
// interrupt events. Skipped on an agent that does not report
// HSA_AMD_AQL_PROPERTY_SIGNAL_HINTS in HSA_AMD_AGENT_INFO_AQL_EXTENSIONS.
class SignalHintCost : public TestBase {
 public:
  SignalHintCost(void);

  virtual ~SignalHintCost(void);

  virtual void SetUp(void);

  virtual void Run(void);

  virtual void DisplayResults(void) const;

  virtual void DisplayTestInfo(void);

  virtual void Close(void);

 private:
  enum class SignalKind { kNone, kHost, kDevice, kInterrupt };

  struct BarrierCost {
    std::string config;
    double us_per_barrier;
    double us_saved;
  };

  // Median CP time per barrier over chains of barriers that each complete their own signal of
  // signal_kind, hinted with hints
  double TimeBarrierChain(SignalKind signal_kind, uint64_t hints);

  hsa_queue_t* queue_ = nullptr;
  hsa_amd_memory_pool_t device_signal_pool_ = {0};
  uint64_t gpu_timestamp_frequency_ = 0;
  std::vector<BarrierCost> barrier_costs_;
};

#endif  // ROCRTST_SUITES_PERFORMANCE_SIGNAL_HINT_COST_H_

/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/
#include "suites/performance/signal_hint_cost.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include "common/base_rocr_utils.h"
#include "gtest/gtest.h"
#include "hsa/amd_hsa_signal.h"

namespace {

constexpr uint32_t kQueueSize = 1024;
constexpr int kBarriersPerChain = 64;
constexpr int kTimedChains = 100;
// Long enough for the CP to reach the chain's first barrier and wait on the trigger
constexpr auto kTriggerDelay = std::chrono::microseconds(100);

constexpr uint64_t kAllHints =
    AMD_SIGNAL_HINT_INTR_NONE | AMD_SIGNAL_HINT_VALUE_BINARY | AMD_SIGNAL_HINT_TS_OFF;

const std::pair<const char*, uint64_t> kHintSets[] = {
    {"no hints", 0},
    {"VALUE_BINARY", AMD_SIGNAL_HINT_VALUE_BINARY},
    {"TS_OFF", AMD_SIGNAL_HINT_TS_OFF},
    {"INTR_NONE", AMD_SIGNAL_HINT_INTR_NONE},
    {"VALUE_BINARY + TS_OFF", AMD_SIGNAL_HINT_VALUE_BINARY | AMD_SIGNAL_HINT_TS_OFF},
    {"VALUE_BINARY + INTR_NONE", AMD_SIGNAL_HINT_VALUE_BINARY | AMD_SIGNAL_HINT_INTR_NONE},
    {"TS_OFF + INTR_NONE", AMD_SIGNAL_HINT_TS_OFF | AMD_SIGNAL_HINT_INTR_NONE},
    {"all three", kAllHints},
};

hsa_signal_t CreateHostSignal(hsa_signal_value_t initial_value) {
  hsa_signal_t signal = {0};
  EXPECT_EQ(hsa_amd_signal_create(initial_value, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal),
            HSA_STATUS_SUCCESS);
  return signal;
}

}  // namespace

SignalHintCost::SignalHintCost(void) : TestBase() {
  set_title("Signal Hint Cost");
  set_description(
      "Measures the CP time per barrier packet that completes a signal, with each completion "
      "signal handle hint and with all of them. Each chain of barriers is held behind a host "
      "trigger and timed from its first to its last barrier's end timestamp.");
}

SignalHintCost::~SignalHintCost(void) {}

void SignalHintCost::SetUp(void) {
  // Interrupt signals need HSA_ENABLE_INTERRUPT, which TestBase::SetUp sets from this
  set_enable_interrupt(true);
  TestBase::SetUp();
  if (test_skipped_) return;

  ASSERT_EQ(rocrtst::SetDefaultAgents(this), HSA_STATUS_SUCCESS);
  uint8_t aql_extensions[8] = {};
  ASSERT_EQ(hsa_agent_get_info(*gpu_device1(),
                               static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_AQL_EXTENSIONS),
                               aql_extensions),
            HSA_STATUS_SUCCESS);
  if (!hsa_flag_isset64(aql_extensions, HSA_AMD_AQL_PROPERTY_SIGNAL_HINTS)) {
    markAsSkip();
    std::cout << "Test not applicable as the GPU's CP does not support signal hints. Skipping."
              << std::endl;
    return;
  }

  ASSERT_EQ(
      hsa_agent_get_info(*gpu_device1(),
                         static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY),
                         &gpu_timestamp_frequency_),
      HSA_STATUS_SUCCESS);
  device_signal_pool_ = rocrtst::FindExtendedFineGrainedPool(*gpu_device1());
  // A profiling queue, like the ones HIP creates, so TS_OFF has timestamps to remove
  ASSERT_EQ(rocrtst::CreateQueue(*gpu_device1(), &queue_, kQueueSize), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_profiling_set_profiler_enabled(queue_, 1), HSA_STATUS_SUCCESS);
}

double SignalHintCost::TimeBarrierChain(SignalKind signal_kind, uint64_t hints) {
  std::vector<hsa_signal_t> completions(signal_kind == SignalKind::kNone ? 0 : kBarriersPerChain);
  if (signal_kind == SignalKind::kDevice) {
    const std::vector<hsa_signal_value_t> initial_values(completions.size(), 1);
    hsa_amd_signal_batch_t batch;
    EXPECT_EQ(hsa_amd_signal_batch_create(completions.size(), initial_values.data(), 0, nullptr,
                                          HSA_AMD_SIGNAL_AMD_GPU_ONLY, device_signal_pool_,
                                          &batch, completions.data()),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_amd_signal_batch_destroy(batch), HSA_STATUS_SUCCESS);
  } else {
    for (hsa_signal_t& completion : completions) {
      if (signal_kind == SignalKind::kInterrupt) {
        EXPECT_EQ(hsa_signal_create(1, 0, nullptr, &completion), HSA_STATUS_SUCCESS);
      } else {
        completion = CreateHostSignal(1);
      }
    }
  }
  const hsa_signal_t trigger = CreateHostSignal(1);
  const hsa_signal_t chain_begin = CreateHostSignal(1);
  const hsa_signal_t chain_end = CreateHostSignal(1);
  const amd_signal_t* begin_record = amd_signal_from_handle(chain_begin.handle);
  const amd_signal_t* end_record = amd_signal_from_handle(chain_end.handle);

  std::vector<double> us_per_barrier;
  for (int chain = 0; chain < kTimedChains; ++chain) {
    for (const hsa_signal_t signal : {trigger, chain_begin, chain_end}) {
      hsa_signal_store_relaxed(signal, 1);
    }
    for (const hsa_signal_t completion : completions) hsa_signal_store_relaxed(completion, 1);

    rocrtst::WriteBarrierAndToQueue(queue_, trigger,
                                    hsa_amd_signal_set_hints(chain_begin, AMD_SIGNAL_HINT_TS_ON));
    for (int barrier = 0; barrier < kBarriersPerChain; ++barrier) {
      const hsa_signal_t completion =
          completions.empty() ? hsa_signal_t{0}
                              : hsa_amd_signal_set_hints(completions[barrier], hints);
      rocrtst::WriteBarrierAndToQueue(queue_, hsa_signal_t{0}, completion);
    }
    const uint64_t end_index = rocrtst::WriteBarrierAndToQueue(
        queue_, hsa_signal_t{0}, hsa_amd_signal_set_hints(chain_end, AMD_SIGNAL_HINT_TS_ON));
    hsa_signal_store_screlease(queue_->doorbell_signal, end_index);

    std::this_thread::sleep_for(kTriggerDelay);
    hsa_signal_store_screlease(trigger, 0);
    hsa_signal_wait_scacquire(chain_end, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_ACTIVE);
    const double chain_ticks = static_cast<double>(end_record->end_ts - begin_record->end_ts);
    us_per_barrier.push_back(chain_ticks * 1e6 / gpu_timestamp_frequency_ /
                             (kBarriersPerChain + 1));
  }

  for (const hsa_signal_t signal : {trigger, chain_begin, chain_end}) {
    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  }
  for (const hsa_signal_t completion : completions) {
    EXPECT_EQ(hsa_signal_destroy(completion), HSA_STATUS_SUCCESS);
  }
  std::nth_element(us_per_barrier.begin(), us_per_barrier.begin() + us_per_barrier.size() / 2,
                   us_per_barrier.end());
  return us_per_barrier[us_per_barrier.size() / 2];
}

void SignalHintCost::Run(void) {
  if (!rocrtst::CheckProfile(this)) return;
  if (Skip()) return;
  TestBase::Run();

  // Untimed, so the first timed configuration doesn't run while the GPU clocks ramp up
  TimeBarrierChain(SignalKind::kHost, 0);
  barrier_costs_.push_back(
      {"no completion signal", TimeBarrierChain(SignalKind::kNone, 0), 0});

  const std::pair<const char*, SignalKind> signal_kinds[] = {
      {"host signal", SignalKind::kHost},
      {"device signal", SignalKind::kDevice},
      {"interrupt signal", SignalKind::kInterrupt},
  };
  for (const auto& [kind_name, signal_kind] : signal_kinds) {
    if (signal_kind == SignalKind::kDevice && device_signal_pool_.handle == 0) continue;
    double unhinted_us = 0;
    for (const auto& [hints_name, hints] : kHintSets) {
      const double us_per_barrier = TimeBarrierChain(signal_kind, hints);
      if (hints == 0) unhinted_us = us_per_barrier;
      barrier_costs_.push_back({std::string(kind_name) + ", " + hints_name, us_per_barrier,
                                unhinted_us - us_per_barrier});
    }
  }
}

void SignalHintCost::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void SignalHintCost::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this) || Skip()) return;
  TestBase::DisplayResults();
  std::printf("  %-44s %12s %10s\n", "completion signal", "us/barrier", "us saved");
  for (const BarrierCost& cost : barrier_costs_) {
    std::printf("  %-44s %12.3f %10.3f\n", cost.config.c_str(), cost.us_per_barrier,
                cost.us_saved);
  }
}

void SignalHintCost::Close(void) {
  if (queue_ != nullptr) {
    EXPECT_EQ(hsa_queue_destroy(queue_), HSA_STATUS_SUCCESS);
  }
  TestBase::Close();
}

/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/signal_hints.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/amd_hsa_queue.h"
#include "hsa/amd_hsa_signal.h"
#include "hsa/hsa_ext_amd.h"

namespace {

constexpr uint32_t kQueueSize = 64;
constexpr uint64_t kCompletionTimeoutMs = 5000;
// A blocked InterruptSignal wait spins for 200 us, then sleeps on its event; release the
// packets well after that
constexpr auto kWaiterSleepDelay = std::chrono::milliseconds(5);
constexpr auto kUpdateGap = std::chrono::milliseconds(20);
constexpr uint64_t kWaiterTimeoutMs = 100;
constexpr double kInterruptWakeUs = 5000;

void PrintSubtestHeader(const char* header) {
  std::cout << "  *** Signal Hints Subtest: " << header << " ***" << std::endl;
}

void RingDoorbell(hsa_queue_t* queue, uint64_t index) {
  hsa_signal_store_screlease(queue->doorbell_signal, index);
}

hsa_signal_t CreateHostSignal(hsa_signal_value_t initial_value) {
  hsa_signal_t signal = {0};
  EXPECT_EQ(hsa_amd_signal_create(initial_value, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal),
            HSA_STATUS_SUCCESS);
  return signal;
}

// Whether the CP reports AMD_QUEUE_CAPS_CP_SIGNAL_HINTS in a queue's caps, which it writes once it
// has processed a packet on the queue
bool CpReportsSignalHints(hsa_agent_t agent) {
  hsa_queue_t* queue = nullptr;
  EXPECT_EQ(rocrtst::CreateQueue(agent, &queue, kQueueSize), HSA_STATUS_SUCCESS);
  if (queue == nullptr) return false;
  const hsa_signal_t completion = CreateHostSignal(1);
  RingDoorbell(queue, rocrtst::WriteBarrierAndToQueue(queue, hsa_signal_t{0}, completion));
  EXPECT_EQ(hsa_signal_wait_scacquire(completion, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                      HSA_WAIT_STATE_ACTIVE),
            0);
  // The queue is the hsa_queue_t at the start of the runtime's amd_queue_t
  const uint32_t caps = reinterpret_cast<const amd_queue_t*>(queue)->caps;
  EXPECT_EQ(hsa_signal_destroy(completion), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  return (caps & AMD_QUEUE_CAPS_CP_SIGNAL_HINTS) != 0;
}

hsa_signal_t CreateDeviceSignal(hsa_amd_memory_pool_t pool, hsa_signal_value_t initial_value) {
  hsa_signal_t signal = {0};
  hsa_amd_signal_batch_t batch;
  EXPECT_EQ(hsa_amd_signal_batch_create(1, &initial_value, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY,
                                        pool, &batch, &signal),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_signal_batch_destroy(batch), HSA_STATUS_SUCCESS);
  return signal;
}

}  // namespace

SignalHintsTest::SignalHintsTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("  *** Signal Hints Tests ***");
  set_description(
      "  *** Submits barrier packets with hinted completion signals and checks the CP's value "
      "update, timestamps and interrupts follow each hint ***");
}

SignalHintsTest::~SignalHintsTest(void) {}

void SignalHintsTest::SetUp(void) {
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
  const bool hints_supported = hsa_flag_isset64(aql_extensions, HSA_AMD_AQL_PROPERTY_SIGNAL_HINTS);
  // ROCr sets the bit from the same caps, at the index hsa_flag_isset64 reads
  EXPECT_EQ(hints_supported, CpReportsSignalHints(*gpu_device1()))
      << "AQL_EXTENSIONS bytes " << std::hex << static_cast<uint32_t>(aql_extensions[0]);
  if (!hints_supported) {
    markAsSkip();
    std::cout << "Test not applicable as the GPU's CP does not support signal hints. Skipping."
              << std::endl;
    return;
  }

  ASSERT_EQ(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_frequency_),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(rocrtst::CreateQueue(*gpu_device1(), &profiled_queue_, kQueueSize),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_profiling_set_profiler_enabled(profiled_queue_, 1), HSA_STATUS_SUCCESS);
  ASSERT_EQ(rocrtst::CreateQueue(*gpu_device1(), &unprofiled_queue_, kQueueSize),
            HSA_STATUS_SUCCESS);
}

void SignalHintsTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) return;
  TestBase::Run();
}

void SignalHintsTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void SignalHintsTest::DisplayResults(void) const {
  TestBase::DisplayResults();
  std::cout << std::endl;
  for (const auto& result : resultList_) std::cout << result << std::endl;
}

void SignalHintsTest::Close(void) {
  if (profiled_queue_ != nullptr) {
    EXPECT_EQ(hsa_queue_destroy(profiled_queue_), HSA_STATUS_SUCCESS);
  }
  if (unprofiled_queue_ != nullptr) {
    EXPECT_EQ(hsa_queue_destroy(unprofiled_queue_), HSA_STATUS_SUCCESS);
  }
  TestBase::Close();
}

uint64_t SignalHintsTest::TimeoutTicks(uint64_t milliseconds) const {
  return timestamp_frequency_ * milliseconds / 1000;
}

void SignalHintsTest::ValueBinary(void) {
  if (Skip()) return;
  PrintSubtestHeader("VALUE_BINARY");
  const hsa_amd_memory_pool_t device_pool = rocrtst::FindExtendedFineGrainedPool(*gpu_device1());
  const bool has_device_pool = device_pool.handle != 0;

  for (const bool device_memory : {false, true}) {
    if (device_memory && !has_device_pool) {
      resultList_.push_back("    VALUE_BINARY in device memory: no extended fine-grained pool");
      continue;
    }
    const char* memory_name = device_memory ? "device" : "host";
    for (const bool binary : {false, true}) {
      constexpr hsa_signal_value_t kInitialValue = 3;
      const hsa_signal_t signal = device_memory ? CreateDeviceSignal(device_pool, kInitialValue)
                                                : CreateHostSignal(kInitialValue);
      const uint64_t hints = binary ? AMD_SIGNAL_HINT_VALUE_BINARY : 0;
      RingDoorbell(profiled_queue_,
                   rocrtst::WriteBarrierAndToQueue(profiled_queue_, hsa_signal_t{0},
                                                   hsa_amd_signal_set_hints(signal, hints)));
      const hsa_signal_value_t value =
          hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_NE, kInitialValue,
                                    TimeoutTicks(kCompletionTimeoutMs), HSA_WAIT_STATE_ACTIVE);
      EXPECT_EQ(value, binary ? 0 : kInitialValue - 1)
          << memory_name << " signal, VALUE_BINARY " << binary;
      EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
    }
    resultList_.push_back(std::string("    VALUE_BINARY stores 0 in ") + memory_name +
                          " memory; no hint decrements");
  }
}

void SignalHintsTest::Timestamps(void) {
  if (Skip()) return;
  PrintSubtestHeader("TS_INHERIT, TS_ON, TS_OFF");
  struct TimestampCase {
    const char* name;
    hsa_queue_t* queue;
    uint64_t hint;
    bool expect_timestamps;
  };
  const TimestampCase cases[] = {
      {"TS_INHERIT on a profiling queue", profiled_queue_, AMD_SIGNAL_HINT_TS_INHERIT, true},
      {"TS_OFF on a profiling queue", profiled_queue_, AMD_SIGNAL_HINT_TS_OFF, false},
      {"TS_INHERIT on a non-profiling queue", unprofiled_queue_, AMD_SIGNAL_HINT_TS_INHERIT, false},
      {"TS_ON on a non-profiling queue", unprofiled_queue_, AMD_SIGNAL_HINT_TS_ON, true},
  };

  for (const TimestampCase& timestamp_case : cases) {
    const hsa_signal_t signal = CreateHostSignal(1);
    RingDoorbell(timestamp_case.queue,
                 rocrtst::WriteBarrierAndToQueue(timestamp_case.queue, hsa_signal_t{0},
                                                 hsa_amd_signal_set_hints(signal,
                                                                          timestamp_case.hint)));
    ASSERT_EQ(hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                        TimeoutTicks(kCompletionTimeoutMs), HSA_WAIT_STATE_ACTIVE),
              0)
        << timestamp_case.name;
    const amd_signal_t* amd_signal = amd_signal_from_handle(signal.handle);
    EXPECT_EQ(amd_signal->start_ts != 0, timestamp_case.expect_timestamps) << timestamp_case.name;
    EXPECT_EQ(amd_signal->end_ts != 0, timestamp_case.expect_timestamps) << timestamp_case.name;
    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
    resultList_.push_back(std::string("    ") + timestamp_case.name + ": timestamps " +
                          (timestamp_case.expect_timestamps ? "written" : "not written"));
  }
}

double SignalHintsTest::WakeDelayUs(uint64_t hints, int num_updates) {
  hsa_signal_t signal = {0};
  EXPECT_EQ(hsa_signal_create(num_updates, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  std::vector<hsa_signal_t> triggers;
  uint64_t last_index = 0;
  for (int update = 0; update < num_updates; ++update) {
    triggers.push_back(CreateHostSignal(1));
    last_index = rocrtst::WriteBarrierAndToQueue(unprofiled_queue_, triggers.back(),
                                                 hsa_amd_signal_set_hints(signal, hints));
  }
  RingDoorbell(unprofiled_queue_, last_index);

  std::chrono::steady_clock::time_point woke_at;
  std::thread waiter([&] {
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, num_updates,
                              TimeoutTicks(kWaiterTimeoutMs), HSA_WAIT_STATE_BLOCKED);
    woke_at = std::chrono::steady_clock::now();
  });
  std::this_thread::sleep_for(kWaiterSleepDelay);
  const auto first_release = std::chrono::steady_clock::now();
  for (size_t update = 0; update < triggers.size(); ++update) {
    if (update != 0) std::this_thread::sleep_for(kUpdateGap);
    hsa_signal_store_screlease(triggers[update], 0);
  }
  waiter.join();

  EXPECT_EQ(hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                      TimeoutTicks(kCompletionTimeoutMs), HSA_WAIT_STATE_ACTIVE),
            0);
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  for (const hsa_signal_t trigger : triggers) {
    EXPECT_EQ(hsa_signal_destroy(trigger), HSA_STATUS_SUCCESS);
  }
  return std::chrono::duration<double, std::micro>(woke_at - first_release).count();
}

void SignalHintsTest::Interrupts(void) {
  if (Skip()) return;
  PrintSubtestHeader("INTR_DEFAULT, INTR_NONE, INTR_ON_ZERO");
  const double gap_us = std::chrono::duration<double, std::micro>(kUpdateGap).count();
  auto record = [this](const char* name, double delay_us) {
    resultList_.push_back(std::string("    ") + name + ": waiter woke " +
                          std::to_string(static_cast<int64_t>(delay_us)) +
                          " us after the first update");
  };

  const double default_delay_us = WakeDelayUs(AMD_SIGNAL_HINT_INTR_DEFAULT, 1);
  EXPECT_LT(default_delay_us, kInterruptWakeUs) << "INTR_DEFAULT did not wake the waiter";
  record("INTR_DEFAULT, 1 -> 0", default_delay_us);

  const double none_delay_us = WakeDelayUs(AMD_SIGNAL_HINT_INTR_NONE, 1);
  EXPECT_GT(none_delay_us, gap_us) << "INTR_NONE woke the waiter";
  record("INTR_NONE, 1 -> 0 (wakes at its timeout)", none_delay_us);

  const double default_nonzero_delay_us = WakeDelayUs(AMD_SIGNAL_HINT_INTR_DEFAULT, 2);
  EXPECT_LT(default_nonzero_delay_us, kInterruptWakeUs)
      << "INTR_DEFAULT did not wake the waiter on 2 -> 1";
  record("INTR_DEFAULT, 2 -> 1 -> 0", default_nonzero_delay_us);

  const double on_zero_delay_us = WakeDelayUs(AMD_SIGNAL_HINT_INTR_ON_ZERO, 2);
  EXPECT_GT(on_zero_delay_us, gap_us) << "INTR_ON_ZERO woke the waiter on 2 -> 1";
  EXPECT_LT(on_zero_delay_us, gap_us + kInterruptWakeUs)
      << "INTR_ON_ZERO did not wake the waiter on 1 -> 0";
  record("INTR_ON_ZERO, 2 -> 1 -> 0 (wakes at 0)", on_zero_delay_us);
}

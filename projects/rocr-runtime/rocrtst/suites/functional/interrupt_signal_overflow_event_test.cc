/*
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifdef __linux__
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/amd_hsa_signal.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "kfd_ioctl.h"

namespace {

class InterruptSignalOverflowEventTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto status = hsa_init();
    initialized_ = status == HSA_STATUS_SUCCESS;
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    ASSERT_EQ(HSA_STATUS_INFO_BREAK, hsa_iterate_agents(FindGpu, &gpu_));
  }

  void TearDown() override {
    if (!initialized_) return;
    for (auto queue : queues_) EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_queue_destroy(queue));
    for (auto signal : signals_) EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(signal));
    EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_shut_down());
  }

  static hsa_status_t FindGpu(hsa_agent_t agent, void* data) {
    hsa_device_type_t type;
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;
    if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
    *static_cast<hsa_agent_t*>(data) = agent;
    return HSA_STATUS_INFO_BREAK;
  }

  static const amd_signal_t* Amd(hsa_signal_t signal) {
    return reinterpret_cast<const amd_signal_t*>(signal.handle);
  }

  // ROCr clears the mailbox of a signal without an event; event id 0 is a valid event.
  static bool HasEvent(hsa_signal_t signal) { return Amd(signal)->event_mailbox_ptr != 0; }

  // The overflow event is only used with KFD event age, the same check ROCr makes.
  static bool KfdHasEventAge() {
    const int fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (fd < 0) return false;
    kfd_ioctl_get_version_args args{};
    const int rc = ioctl(fd, AMDKFD_IOC_GET_VERSION, &args);
    close(fd);
    return rc == 0 && args.major_version == 1 && args.minor_version >= 14;
  }

  hsa_signal_t NewSignal(hsa_signal_value_t value) {
    hsa_signal_t signal{};
    EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(value, 0, nullptr, &signal));
    signals_.push_back(signal);
    return signal;
  }

  hsa_queue_t* NewQueue() {
    hsa_queue_t* queue = nullptr;
    EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_queue_create(gpu_, 64, HSA_QUEUE_TYPE_SINGLE, nullptr,
                                                   nullptr, UINT32_MAX, UINT32_MAX, &queue));
    if (queue != nullptr) queues_.push_back(queue);
    return queue;
  }

  enum class Exhaustion { kNoEvents, kNotReached, kShared, kLost };

  // Creates interrupt signals until the KFD events run out. Once they do, new
  // signals must share one event (two in a row report the same event); a signal
  // without an event after earlier ones had events means the overflow event is
  // missing.
  Exhaustion Exhaust() {
    uint64_t previous = 0;
    for (unsigned i = 0; i <= KFD_SIGNAL_EVENT_LIMIT + 16; ++i) {
      const hsa_signal_t signal = NewSignal(0);
      if (!HasEvent(signal)) return i == 0 ? Exhaustion::kNoEvents : Exhaustion::kLost;
      if (i != 0 && Amd(signal)->event_id == previous) return Exhaustion::kShared;
      previous = Amd(signal)->event_id;
    }
    return Exhaustion::kNotReached;
  }

  // Skips (returns false) where the case cannot be set up; fails if a signal
  // created after exhaustion has no event.
  bool SetUpExhaustion() {
    if (!KfdHasEventAge()) {
      rocrtst::SkipCurrentTest("KFD without event age: ROCr does not share an event.");
      return false;
    }
    const Exhaustion exhaustion = Exhaust();
    if (exhaustion == Exhaustion::kNoEvents || exhaustion == Exhaustion::kNotReached) {
      rocrtst::SkipCurrentTest("No KFD events, or they could not be used up.");
      return false;
    }
    EXPECT_EQ(Exhaustion::kShared, exhaustion)
        << "an interrupt signal created after the KFD events ran out has no event";
    return exhaustion == Exhaustion::kShared;
  }

  // Barrier-AND packet that waits for gate == 0, then decrements completion.
  static void SubmitBarrier(hsa_queue_t* queue, hsa_signal_t completion, hsa_signal_t gate) {
    const uint64_t index = hsa_queue_add_write_index_relaxed(queue, 1);
    auto* packet =
        reinterpret_cast<hsa_barrier_and_packet_t*>(queue->base_address) + (index % queue->size);
    memset(reinterpret_cast<uint8_t*>(packet) + 4, 0, sizeof(*packet) - 4);
    packet->dep_signal[0] = gate;
    packet->completion_signal = completion;
    const uint16_t header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n(reinterpret_cast<uint16_t*>(packet), header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(queue->doorbell_signal, index);
  }

  // Opens the gate after a delay longer than the runtime's polling window, so the
  // GPU completes the signal while the waiter is asleep in KFD.
  static std::thread OpenLater(hsa_signal_t gate, int delay_ms = 20) {
    return std::thread([gate, delay_ms] {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      hsa_signal_store_screlease(gate, 0);
    });
  }

  static double ProcessCpuSeconds() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
        (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6;
  }

  hsa_agent_t gpu_{};
  std::vector<hsa_queue_t*> queues_;
  std::vector<hsa_signal_t> signals_;

 private:
  bool initialized_ = false;
};

TEST_F(InterruptSignalOverflowEventTest, SignalsAfterExhaustionWakeWaiters) {
  // Queues need KFD events too, so create one before using them up.
  hsa_queue_t* queue = NewQueue();
  ASSERT_NE(nullptr, queue);
  if (!SetUpExhaustion()) return;
  uint64_t frequency = 0;
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency));

  // A blocked wait sleeps in KFD and the GPU completion wakes it.
  const hsa_signal_t waited = NewSignal(1);
  const hsa_signal_t gate = NewSignal(1);
  EXPECT_TRUE(HasEvent(waited));
  SubmitBarrier(queue, waited, gate);
  struct rusage before {};
  struct rusage after {};
  ASSERT_EQ(0, getrusage(RUSAGE_THREAD, &before));
  std::thread opener = OpenLater(gate);
  const auto value = hsa_signal_wait_scacquire(waited, HSA_SIGNAL_CONDITION_EQ, 0, frequency,
                                               HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(0, getrusage(RUSAGE_THREAD, &after));
  opener.join();
  EXPECT_EQ(0, value);
  // A busy loop may be preempted, but only a blocking wait yields voluntarily.
  EXPECT_GT(after.ru_nvcsw, before.ru_nvcsw);

  // An async handler on a signal created after exhaustion runs on the GPU completion.
  static std::atomic<bool> fired;
  fired = false;
  const hsa_signal_t handled = NewSignal(1);
  hsa_signal_store_screlease(gate, 1);
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_signal_async_handler(
                handled, HSA_SIGNAL_CONDITION_EQ, 0,
                [](hsa_signal_value_t, void*) {
                  fired = true;
                  return false;
                },
                nullptr));
  SubmitBarrier(queue, handled, gate);
  opener = OpenLater(gate);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!fired && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  opener.join();
  EXPECT_TRUE(fired.load());

  // With a handler parked on a signal created after exhaustion, the async-events
  // thread sleeps in KFD instead of polling: the idle process stays well below a core.
  const hsa_signal_t parked = NewSignal(1);
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_signal_async_handler(
                parked, HSA_SIGNAL_CONDITION_EQ, 0,
                [](hsa_signal_value_t, void*) { return false; }, nullptr));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const double cpu0 = ProcessCpuSeconds();
  const auto wall0 = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  const double cpu = ProcessCpuSeconds() - cpu0;
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  EXPECT_LT(cpu / wall, 0.5) << "process CPU while idle: " << cpu / wall * 100 << "%";
  hsa_signal_store_screlease(parked, 0);  // let the handler run and unregister
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

// Wait-all over a signal with its own event and a signal sharing the overflow event,
// while an unrelated signal raises the shared event first. ROCr must not end up
// waiting in KFD for the own event of the already satisfied signal.
TEST_F(InterruptSignalOverflowEventTest, WaitAllWithSharedEvent) {
  hsa_queue_t* queue = NewQueue();
  ASSERT_NE(nullptr, queue);
  const hsa_signal_t own = NewSignal(1);  // created before exhaustion: own event
  if (!SetUpExhaustion()) return;
  uint64_t frequency = 0;
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency));

  const hsa_signal_t shared = NewSignal(1);
  const hsa_signal_t unrelated = NewSignal(1);
  const hsa_signal_t gate_unrelated = NewSignal(1);
  const hsa_signal_t gate_own = NewSignal(1);
  const hsa_signal_t gate_shared = NewSignal(1);
  ASSERT_TRUE(HasEvent(own) && HasEvent(shared) && HasEvent(unrelated));
  ASSERT_NE(Amd(own)->event_id, Amd(shared)->event_id);
  ASSERT_EQ(Amd(shared)->event_id, Amd(unrelated)->event_id);

  // Completion order: unrelated, then own, then shared.
  SubmitBarrier(queue, unrelated, gate_unrelated);
  SubmitBarrier(queue, own, gate_own);
  SubmitBarrier(queue, shared, gate_shared);
  std::thread opener([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    hsa_signal_store_screlease(gate_unrelated, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    hsa_signal_store_screlease(gate_own, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    hsa_signal_store_screlease(gate_shared, 0);
  });

  hsa_signal_t waited[] = {own, shared};
  hsa_signal_condition_t conds[] = {HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ};
  hsa_signal_value_t values[] = {0, 0};
  hsa_signal_value_t satisfying[2] = {};
  // A wait stuck in KFD only returns when its timeout expires, so give it a long one
  // and require it to finish soon after the last completion (~60 ms).
  const auto start = std::chrono::steady_clock::now();
  const uint32_t ret = hsa_amd_signal_wait_all(2, waited, conds, values, frequency * 10,
                                               HSA_WAIT_STATE_BLOCKED, satisfying);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  opener.join();
  EXPECT_EQ(0u, ret) << "wait-all timed out";
  EXPECT_LT(elapsed, std::chrono::seconds(2)) << "wait-all returned only at its timeout";
  EXPECT_EQ(0, hsa_signal_load_scacquire(own));
  EXPECT_EQ(0, hsa_signal_load_scacquire(shared));
}

}  // namespace
#endif  // __linux__

/*
 * SPDX-License-Identifier: MIT
 */

#ifdef __linux__
#include <sys/resource.h>

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "hsa/amd_hsa_signal.h"
#include "hsa/hsa.h"
#include "kfd_ioctl.h"

namespace {

class InterruptSignalWaitTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto status = hsa_init();
    initialized_ = status == HSA_STATUS_SUCCESS;
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  }

  void TearDown() override {
    if (!initialized_) return;
    for (auto signal : signals_) EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(signal));
    EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_shut_down());
  }

  std::vector<hsa_signal_t> signals_;

 private:
  bool initialized_ = false;
};

TEST_F(InterruptSignalWaitTest, BlockedWaitYieldsWithoutAnEvent) {
  hsa_signal_t signal{};
  bool missing_event = false;
  for (unsigned i = 0; i <= KFD_SIGNAL_EVENT_LIMIT; ++i) {
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));
    signals_.push_back(signal);
    const auto* record = reinterpret_cast<const amd_signal_t*>(signal.handle);
    if (record->event_id == 0) {
      missing_event = true;
      break;
    }
  }

  // Require at least one interrupt-backed signal to distinguish exhaustion
  // from a platform or configuration without interrupt-backed waits.
  if (!missing_event || signals_.size() == 1) {
    std::puts("[ SKIPPED ] Could not reproduce KFD event exhaustion.");
    return;
  }

  uint64_t frequency = 0;
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency));
  ASSERT_GT(frequency, 20u);
  struct rusage before {};
  struct rusage after {};
  const auto start = std::chrono::steady_clock::now();
  ASSERT_EQ(0, getrusage(RUSAGE_THREAD, &before));
  // Wait 50 ms without a producer so the wait must reach the fallback.
  const auto pending = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, frequency / 20,
                                                 HSA_WAIT_STATE_BLOCKED);
  ASSERT_EQ(0, getrusage(RUSAGE_THREAD, &after));
  EXPECT_EQ(1, pending);
  EXPECT_TRUE(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(25));
  // A busy loop may be preempted, but only a blocking wait yields voluntarily.
  EXPECT_GT(after.ru_nvcsw, before.ru_nvcsw);

  std::thread completer([signal] {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    hsa_signal_store_screlease(signal, 0);
  });
  const auto completed = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, frequency,
                                                   HSA_WAIT_STATE_BLOCKED);
  completer.join();
  EXPECT_EQ(0, completed);
}

}  // namespace
#endif  // __linux__

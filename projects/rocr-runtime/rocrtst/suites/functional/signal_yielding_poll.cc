/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/signal_yielding_poll.h"

#include <time.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>

#include "common/os.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

constexpr const char* kYieldingPollEnv = "HSA_ENABLE_YIELDING_POLL";

double ThreadCpuMs() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

SignalYieldingPollTest::SignalYieldingPollTest(bool yielding_poll)
    : TestBase(), yielding_poll_(yielding_poll), timestamp_frequency_(0), signal_({0}) {
  set_num_iteration(1);
  set_title(yielding_poll ? "RocR Signal Yielding Poll Test"
                          : "RocR Signal Default Busy-Wait Test");
  set_description(
      "Waits on a GPU-only (busy-wait) signal and checks the waiting "
      "thread's CPU time and wake-up timing with and without "
      "HSA_ENABLE_YIELDING_POLL=1.");
}

SignalYieldingPollTest::~SignalYieldingPollTest(void) {}

void SignalYieldingPollTest::SetUp(void) {
  // The flag is read in hsa_init(), so it must be set before TestBase::SetUp().
  if (yielding_poll_) {
    rocrtst::SetEnv(kYieldingPollEnv, "1");
  } else {
    unsetenv(kYieldingPollEnv);
  }

  TestBase::SetUp();
  if (test_skipped_) return;

  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_frequency_));
  ASSERT_GT(timestamp_frequency_, 0u);

  // HSA_AMD_SIGNAL_AMD_GPU_ONLY always yields a DefaultSignal (BusyWaitSignal),
  // whatever the interrupt setting.
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal_));

  // hsa_amd_signal_value_pointer only succeeds for busy-wait signals, so this
  // confirms the waits below run BusyWaitSignal::WaitRelaxed.
  volatile hsa_signal_value_t* value_ptr = nullptr;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_signal_value_pointer(signal_, &value_ptr));
}

void SignalYieldingPollTest::Run(void) { TestBase::Run(); }

void SignalYieldingPollTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void SignalYieldingPollTest::DisplayResults(void) const { TestBase::DisplayResults(); }

void SignalYieldingPollTest::Close() {
  if (signal_.handle != 0) {
    EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(signal_));
    signal_.handle = 0;
  }
  TestBase::Close();
  unsetenv(kYieldingPollEnv);
}

SignalYieldingPollTest::WaitSample SignalYieldingPollTest::TimedWait(double timeout_ms,
                                                                     hsa_wait_state_t hint) {
  const uint64_t timeout_ticks =
      static_cast<uint64_t>(timeout_ms * 1e-3 * double(timestamp_frequency_));

  const double cpu_start = ThreadCpuMs();
  const auto wall_start = std::chrono::steady_clock::now();
  const hsa_signal_value_t value =
      hsa_signal_wait_scacquire(signal_, HSA_SIGNAL_CONDITION_EQ, 0, timeout_ticks, hint);
  const auto wall_end = std::chrono::steady_clock::now();
  const double cpu_end = ThreadCpuMs();

  WaitSample s;
  s.value = value;
  s.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
  s.cpu_ms = cpu_end - cpu_start;
  if (verbosity() > 0) {
    std::cout << "  wait timeout=" << timeout_ms
              << "ms hint=" << (hint == HSA_WAIT_STATE_ACTIVE ? "active" : "blocked")
              << ": value=" << value << " wall=" << s.wall_ms << "ms cpu=" << s.cpu_ms << "ms"
              << std::endl;
  }
  return s;
}

/*
 * Test Name: TestLongWaitYieldsCpu
 * Scope: Functional
 *
 * Purpose: With HSA_ENABLE_YIELDING_POLL=1, a wait that never completes must
 * stop burning the core once it is past its hot window (200 us blocked, 10 ms
 * active) and nap with the escalating backoff.
 *
 * Expected Results: Each 500 ms wait times out with the signal unchanged, and
 * the waiting thread's CPU time is below 20% of the wall time for both hints
 * (the default spin path is close to 100%).
 */
void SignalYieldingPollTest::TestLongWaitYieldsCpu(void) {
  for (hsa_wait_state_t hint : {HSA_WAIT_STATE_BLOCKED, HSA_WAIT_STATE_ACTIVE}) {
    const WaitSample s = TimedWait(500, hint);
    EXPECT_EQ(1, s.value);
    EXPECT_GE(s.wall_ms, 500.0);
    EXPECT_LT(s.cpu_ms, 0.2 * s.wall_ms)
        << "hint=" << hint << " wall=" << s.wall_ms << "ms cpu=" << s.cpu_ms << "ms";
  }
}

/*
 * Test Name: TestNapHonorsTimeout
 * Scope: Functional / Regression
 *
 * Purpose: The nap must be clamped to the caller's remaining timeout. Past
 * ~5 ms a blocked wait naps kPollNapCeilingUs (2 ms) between deadline checks,
 * so without the clamp a wait returns anywhere from 0 to 2 ms after its
 * timeout, depending on where the deadline falls between two checks. The
 * timeouts below sweep one nap period (6.0 .. 8.0 ms in 0.1 ms steps), so the
 * unclamped median overshoot is ~1 ms whatever the sleep slack of the host.
 *
 * Expected Results: Every wait times out with the signal unchanged, not
 * before its deadline (0.1 ms clock tolerance), and the median overshoot of
 * at least one of up to 3 sweeps is below 0.5 ms.
 */
void SignalYieldingPollTest::TestNapHonorsTimeout(void) {
  // A loaded or virtualised host can delay a single wake-up by more than the
  // margin, so allow up to kRounds sweeps and pass on the first good median.
  // The missing clamp costs ~1 ms in every sweep, so it still fails all rounds.
  constexpr int kRounds = 3;
  double best_median_ms = 0;
  for (int round = 0; round < kRounds; ++round) {
    std::vector<double> overshoot_ms;
    for (int i = 0; i <= 20; ++i) {
      const double timeout_ms = 6.0 + 0.1 * i;
      const WaitSample s = TimedWait(timeout_ms, HSA_WAIT_STATE_BLOCKED);
      EXPECT_EQ(1, s.value);
      // steady_clock vs the runtime's TSC-calibrated deadline: allow 0.1 ms
      // of calibration error before calling a return early.
      EXPECT_GE(s.wall_ms, timeout_ms - 0.1);
      overshoot_ms.push_back(s.wall_ms - timeout_ms);
    }
    const double median_ms = Median(overshoot_ms);
    best_median_ms = (round == 0) ? median_ms : std::min(best_median_ms, median_ms);
    if (best_median_ms < 0.5) break;
  }
  EXPECT_LT(best_median_ms, 0.5) << "a nap ran past the caller's timeout in " << kRounds
                                 << " sweeps";
}

/*
 * Test Name: TestCompletionDuringNap
 * Scope: Functional
 *
 * Purpose: A completion that lands while the waiter is napping (well past the
 * hot window) must still be observed: the value is rechecked after every nap,
 * and a nap is at most kPollNapCeilingUs (2 ms).
 *
 * Expected Results: The wait returns 0 (satisfied, not timed out) and wakes
 * within 20 ms of the store, for both hints.
 */
void SignalYieldingPollTest::TestCompletionDuringNap(void) {
  for (hsa_wait_state_t hint : {HSA_WAIT_STATE_BLOCKED, HSA_WAIT_STATE_ACTIVE}) {
    hsa_signal_store_screlease(signal_, 1);

    std::chrono::steady_clock::time_point stored_at;
    std::thread completer([&]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      stored_at = std::chrono::steady_clock::now();
      hsa_signal_store_screlease(signal_, 0);
    });

    const WaitSample s = TimedWait(5000, hint);
    const auto woke_at = std::chrono::steady_clock::now();
    completer.join();

    EXPECT_EQ(0, s.value);
    const double latency_ms =
        std::chrono::duration<double, std::milli>(woke_at - stored_at).count();
    EXPECT_LT(latency_ms, 20.0) << "hint=" << hint;
  }
  hsa_signal_store_screlease(signal_, 1);
}

/*
 * Test Name: TestActiveHintWidensHotWindow
 * Scope: Functional
 *
 * Purpose: The wait hint must reach the hot-window selection. A 5 ms wait is
 * inside the 10 ms active window, so it spins the whole time, but outside the
 * 200 us blocked window, so it mostly naps.
 *
 * Expected Results: In at least one of up to 3 rounds of 5 waits each, the
 * median CPU share of the active waits is at least 50% and that of the
 * blocked waits at most 30%.
 */
void SignalYieldingPollTest::TestActiveHintWidensHotWindow(void) {
  // Preemption on a loaded host lowers the measured CPU share of a spinning
  // wait, so allow up to kRounds rounds. An ignored hint gives ~5% active
  // share in every round, so it still fails.
  constexpr int kRounds = 3;
  double active_median = 0, blocked_median = 1;
  for (int round = 0; round < kRounds; ++round) {
    std::vector<double> active_share, blocked_share;
    for (int i = 0; i < 5; ++i) {
      WaitSample a = TimedWait(5, HSA_WAIT_STATE_ACTIVE);
      WaitSample b = TimedWait(5, HSA_WAIT_STATE_BLOCKED);
      EXPECT_EQ(1, a.value);
      EXPECT_EQ(1, b.value);
      active_share.push_back(a.cpu_ms / a.wall_ms);
      blocked_share.push_back(b.cpu_ms / b.wall_ms);
    }
    active_median = Median(active_share);
    blocked_median = Median(blocked_share);
    if (active_median >= 0.5 && blocked_median <= 0.3) break;
  }
  EXPECT_GE(active_median, 0.5);
  EXPECT_LE(blocked_median, 0.3);
}

/*
 * Test Name: TestDefaultKeepsSpinning
 * Scope: Functional
 *
 * Purpose: HSA_ENABLE_YIELDING_POLL is opt-in. Without it, a blocked-hint wait
 * on a busy-wait signal keeps spinning (or mwaitx-polling) for the whole wait.
 *
 * Expected Results: The median CPU share over 3 waits of 50 ms is at least 50%.
 */
void SignalYieldingPollTest::TestDefaultKeepsSpinning(void) {
  std::vector<double> share;
  for (int i = 0; i < 3; ++i) {
    const WaitSample s = TimedWait(50, HSA_WAIT_STATE_BLOCKED);
    EXPECT_EQ(1, s.value);
    share.push_back(s.cpu_ms / s.wall_ms);
  }
  EXPECT_GE(Median(share), 0.5);
}

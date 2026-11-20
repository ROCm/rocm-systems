/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "suites/functional/signal_wait_multi.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

SignalWaitMultiTest::SignalWaitMultiTest()
    : TestBase() {
  set_num_iteration(1);  // Number of iterations to execute of the main test;
                         // This is a default value which can be overridden
                         // on the command line.
  set_title("RocR Signal Wait Multi Test");
  set_description("This test verifies that hsa_amd_signal_wait_any and "
                  "hsa_amd_signal_wait_all report the correct satisfying index "
                  "and values, including when the satisfying signal is not the "
                  "first and when a NULL signal is skipped");
}

SignalWaitMultiTest::~SignalWaitMultiTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void SignalWaitMultiTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();
  if (test_skipped_) return;

  // This test needs no agent or pool, only hsa_init() (done by TestBase::SetUp)
  // and signals, so we do not call SetDefaultAgents / SetPoolsTypical.
  uint64_t frequency = 0;
  err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  timeout_hint_ = frequency / 2;
  ASSERT_GT(timeout_hint_, 0u);
  return;
}

void SignalWaitMultiTest::Run(void) {
  TestBase::Run();
}

void SignalWaitMultiTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void SignalWaitMultiTest::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void SignalWaitMultiTest::Close() {
  // Signals must be destroyed before TestBase::Close() -> CommonCleanUp()
  // -> hsa_shut_down().
  for (hsa_signal_t s : signals_) {
    EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(s));
  }
  signals_.clear();

  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

hsa_status_t SignalWaitMultiTest::CreateSignal(hsa_signal_value_t initial_value,
                                               hsa_signal_t* signal) {
  *signal = {0};
  hsa_status_t status = hsa_signal_create(initial_value, 0, nullptr, signal);
  if (status == HSA_STATUS_SUCCESS) {
    signals_.push_back(*signal);
  }
  return status;
}

/*
 * Test Name: TestWaitAnyNonzeroSatisfyingIndex
 * Scope: Conformance / Regression
 *
 * Purpose: Regression for an out-of-bounds write and wrong satisfying value in
 * hsa_amd_signal_wait_any when the satisfying signal is not at index 0. The
 * satisfying_value_vec was sized to 1, so a nonzero satisfying index wrote and
 * read past the end of that vector.
 *
 * Expected Results: The satisfying index is 1, the returned handle matches
 * signals[1], and the satisfying value is 41.
 */
void SignalWaitMultiTest::TestWaitAnyNonzeroSatisfyingIndex(void) {
  hsa_signal_t signals[3];
  ASSERT_EQ(HSA_STATUS_SUCCESS, CreateSignal(0, &signals[0]));
  ASSERT_EQ(HSA_STATUS_SUCCESS, CreateSignal(0, &signals[1]));
  ASSERT_EQ(HSA_STATUS_SUCCESS, CreateSignal(0, &signals[2]));

  hsa_signal_store_relaxed(signals[0], 0);
  hsa_signal_store_relaxed(signals[1], 41);
  hsa_signal_store_relaxed(signals[2], 0);

  hsa_signal_condition_t conds[] = {HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ,
                                    HSA_SIGNAL_CONDITION_EQ};
  hsa_signal_value_t values[] = {1, 41, 1};
  hsa_signal_value_t satisfying_value = -1;

  const uint32_t satisfying_index = hsa_amd_signal_wait_any(
      3, signals, conds, values, timeout_hint_, HSA_WAIT_STATE_ACTIVE, &satisfying_value);

  ASSERT_EQ(1u, satisfying_index);
  EXPECT_EQ(signals[1].handle, signals[satisfying_index].handle);
  EXPECT_EQ(41, satisfying_value);
}

/*
 * Test Name: TestWaitAnyCompactsConditionsAndValues
 * Scope: Conformance / Regression
 *
 * Purpose: Regression for conds/values not being compacted alongside the
 * valid_signals list when a NULL/invalid signal is skipped. Misalignment after
 * a skipped signal caused the wrong condition/value to be applied.
 *
 * Expected Results: The satisfying index is 2, the returned handle matches
 * signals[2], and the satisfying value is 22.
 */
void SignalWaitMultiTest::TestWaitAnyCompactsConditionsAndValues(void) {
  hsa_signal_t signals[3] = {};
  // signals[1] is intentionally left null (handle == 0) to exercise the
  // NULL/invalid-signal skip and the conds/values compaction path.
  ASSERT_EQ(HSA_STATUS_SUCCESS, CreateSignal(0, &signals[0]));
  ASSERT_EQ(HSA_STATUS_SUCCESS, CreateSignal(0, &signals[2]));

  hsa_signal_store_relaxed(signals[0], 10);
  hsa_signal_store_relaxed(signals[2], 22);

  hsa_signal_condition_t conds[] = {HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_LT,
                                    HSA_SIGNAL_CONDITION_GTE};
  hsa_signal_value_t values[] = {11, 0, 22};
  hsa_signal_value_t satisfying_value = -1;

  const uint32_t satisfying_index = hsa_amd_signal_wait_any(
      3, signals, conds, values, timeout_hint_, HSA_WAIT_STATE_ACTIVE, &satisfying_value);

  ASSERT_EQ(2u, satisfying_index);
  EXPECT_EQ(signals[2].handle, signals[satisfying_index].handle);
  EXPECT_EQ(22, satisfying_value);
}

/*
 * Test Name: TestWaitAllReportsSatisfyingValues
 * Scope: Conformance / Regression
 *
 * Purpose: Non-regression check that hsa_amd_signal_wait_all continues to
 * report the satisfying value for every signal in order and returns success.
 *
 * Expected Results: The result is 0 and satisfying_values are {3, 4, 5}.
 */
void SignalWaitMultiTest::TestWaitAllReportsSatisfyingValues(void) {
  hsa_signal_t signals[3];
  ASSERT_EQ(HSA_STATUS_SUCCESS, CreateSignal(0, &signals[0]));
  ASSERT_EQ(HSA_STATUS_SUCCESS, CreateSignal(0, &signals[1]));
  ASSERT_EQ(HSA_STATUS_SUCCESS, CreateSignal(0, &signals[2]));

  hsa_signal_store_relaxed(signals[0], 3);
  hsa_signal_store_relaxed(signals[1], 4);
  hsa_signal_store_relaxed(signals[2], 5);

  hsa_signal_condition_t conds[] = {HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_GTE,
                                    HSA_SIGNAL_CONDITION_LT};
  hsa_signal_value_t values[] = {3, 4, 6};
  hsa_signal_value_t satisfying_values[] = {-1, -1, -1};

  const uint32_t result = hsa_amd_signal_wait_all(3, signals, conds, values, timeout_hint_,
                                                  HSA_WAIT_STATE_ACTIVE, satisfying_values);

  EXPECT_EQ(0u, result);
  EXPECT_EQ(3, satisfying_values[0]);
  EXPECT_EQ(4, satisfying_values[1]);
  EXPECT_EQ(5, satisfying_values[2]);
}

/*
 * Test Name: TestAsyncHandlerWakeRaceStress
 * Scope: Regression / Stress
 *
 * Purpose: Regression test for the lost-wakeup hang fixed in rocm-systems#10463
 * (ROCM-21824, merged 2026-08-21). Every call to hsa_amd_signal_async_handler
 * causes the runtime to call StoreReleaseAndNotify(1) on the internal
 * control.wake interrupt signal so the async-events loop picks up the newly
 * registered signal. The race: the async-events loop had exhausted its ~200 µs
 * polling window and was incrementing waiting_++ on control.wake (about to call
 * hsaKmtWaitOnMultipleEvents) at the exact moment the registration called
 * StoreReleaseAndNotify. Without the fix, StoreReleaseAndNotify observed
 * InWaiting()==false and skipped hsaKmtSetEvent; the loop then slept without
 * ever adding the new signal to its monitored set, so the registered completion
 * handler never fired — a hang. QMCPACK hit this because every hipGraphLaunch
 * registers one async handler (one StoreReleaseAndNotify per launch).
 *
 * This test registers a one-shot async handler on a fresh signal, immediately
 * fires the signal, and asserts the handler fires within a 2-second deadline.
 * Without the fix the loop eventually misses a registration wake, sleeps
 * indefinitely, and the test fails with a timeout.
 *
 * Expected Results: All kIterations handlers fire within 2 seconds total.
 */
void SignalWaitMultiTest::TestAsyncHandlerWakeRaceStress(void) {
  constexpr int kIterations = 2000;

  std::atomic<int> fired{0};

  // One-shot handler: increments the counter and returns false (do not
  // re-register). Cast avoids a lambda-with-capture in a C callback context.
  struct Ctx { std::atomic<int>* fired; };
  auto handler = [](hsa_signal_value_t, void* arg) -> bool {
    static_cast<Ctx*>(arg)->fired->fetch_add(1, std::memory_order_release);
    return false;
  };
  Ctx ctx{&fired};

  for (int i = 0; i < kIterations; ++i) {
    // Create a signal with initial value 1; condition fires when it reaches 0.
    hsa_signal_t signal;
    ASSERT_EQ(HSA_STATUS_SUCCESS, CreateSignal(1, &signal));

    // Register the handler. Internally the runtime calls
    // StoreReleaseAndNotify(1) on control.wake so the async-events loop
    // thread picks up this new signal. This is the exact call that raced
    // with the loop's waiting_++ in ROCM-21824.
    ASSERT_EQ(HSA_STATUS_SUCCESS,
              hsa_amd_signal_async_handler(signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                           handler, &ctx));

    // Occasionally yield to let the async loop's 200 µs polling window expire
    // and transition to its kernel-sleep path, widening the race window.
    if (i % 50 == 0) std::this_thread::sleep_for(std::chrono::microseconds(500));

    // Satisfy the condition. If the loop missed the registration wake it won't
    // be watching this signal and the handler will not fire.
    hsa_signal_store_screlease(signal, 0);
  }

  // All handlers must fire within 2 seconds. Without the fix the async loop
  // eventually gets stuck sleeping and this deadline is exceeded.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (fired.load(std::memory_order_acquire) < kIterations) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "Async-handler wake race: only " << fired.load() << " of "
        << kIterations << " handlers fired before deadline. "
        << "The async-events loop likely missed a StoreReleaseAndNotify wake "
        << "on control.wake and is sleeping without monitoring a signal.";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  EXPECT_EQ(kIterations, fired.load());
}

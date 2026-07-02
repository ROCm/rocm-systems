/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

class SignalWaitMultiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_init());
    initialized_ = true;

    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_iterate_agents(CountAgent, &agent_count_));
    ASSERT_GT(agent_count_, 0u);

    uint64_t timestamp_frequency = 0;
    ASSERT_EQ(HSA_STATUS_SUCCESS,
              hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_frequency));
    timeout_hint_ = timestamp_frequency / 2;
    ASSERT_GT(timeout_hint_, 0u);
  }

  void TearDown() override {
    for (hsa_signal_t signal : signals_) {
      EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(signal));
    }

    if (initialized_) {
      EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_shut_down());
    }
  }

  hsa_status_t CreateSignal(hsa_signal_value_t initial_value, hsa_signal_t* signal) {
    *signal = {0};
    hsa_status_t status = hsa_signal_create(initial_value, 0, nullptr, signal);
    if (status == HSA_STATUS_SUCCESS) {
      signals_.push_back(*signal);
    }
    return status;
  }

  static hsa_status_t CountAgent(hsa_agent_t, void* data) {
    ++*static_cast<uint32_t*>(data);
    return HSA_STATUS_SUCCESS;
  }

  bool initialized_ = false;
  uint32_t agent_count_ = 0;
  uint64_t timeout_hint_ = 0;
  std::vector<hsa_signal_t> signals_;
};

TEST_F(SignalWaitMultiTest, WaitAnyReportsNonzeroSatisfyingIndex) {
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

TEST_F(SignalWaitMultiTest, WaitAnyCompactsConditionsAndValues) {
  hsa_signal_t signals[3] = {};
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

TEST_F(SignalWaitMultiTest, WaitAllReportsSatisfyingValues) {
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

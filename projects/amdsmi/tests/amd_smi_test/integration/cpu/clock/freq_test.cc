// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// CPU frequency, active-limit, rail iso-frequency and DFC control APIs. Getters
// that only guard the handle omit the null-output test to avoid dereferencing a
// null pointer on success. The core current-frequency limit derives a core
// index from the handle and therefore iterates cpu_cores().

// ---- amdsmi_get_cpu_fclk_mclk (handle guarded only) ----
TEST_F(CpuIntegration, GetFclkMclk_InvalidHandle) {
  uint32_t fclk = 0, mclk = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_fclk_mclk", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_fclk_mclk(kInvalidHandle, &fclk, &mclk);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetFclkMclk_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_fclk_mclk");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t fclk = 0, mclk = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_fclk_mclk", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_fclk_mclk(cpus()[i], &fclk, &mclk);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_cclk_limit (handle guarded only) ----
TEST_F(CpuIntegration, GetCclkLimit_InvalidHandle) {
  uint32_t cclk = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_cclk_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_cclk_limit(kInvalidHandle, &cclk);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetCclkLimit_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_cclk_limit");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t cclk = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_cclk_limit", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_cclk_limit(cpus()[i], &cclk);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_socket_current_active_freq_limit (handle guarded only) ----
TEST_F(CpuIntegration, GetSocketActiveFreqLimit_InvalidHandle) {
  uint16_t freq = 0;
  char* src_type = nullptr;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_current_active_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_cpu_socket_current_active_freq_limit(kInvalidHandle, &freq, &src_type);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSocketActiveFreqLimit_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_socket_current_active_freq_limit");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint16_t freq = 0;
    char* src_type = nullptr;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_current_active_freq_limit",
                       "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err =
        amdsmi_get_cpu_socket_current_active_freq_limit(cpus()[i], &freq, &src_type);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_socket_freq_range (handle guarded only) ----
TEST_F(CpuIntegration, GetSocketFreqRange_InvalidHandle) {
  uint16_t fmax = 0, fmin = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_freq_range", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_freq_range(kInvalidHandle, &fmax, &fmin);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSocketFreqRange_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_socket_freq_range");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint16_t fmax = 0, fmin = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_freq_range", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_socket_freq_range(cpus()[i], &fmax, &fmin);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_core_current_freq_limit (handle guarded only, core handle) ----
TEST_F(CpuIntegration, GetCoreCurrentFreqLimit_InvalidHandle) {
  uint32_t freq = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_current_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_core_current_freq_limit(kInvalidHandle, &freq);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetCoreCurrentFreqLimit_AllCores) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_core_current_freq_limit");
  if (cpu_cores().empty()) GTEST_SKIP() << "No CPU cores";
  for (size_t i = 0; i < cpu_cores().size(); ++i) {
    uint32_t freq = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_current_freq_limit", "core=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_core_current_freq_limit(cpu_cores()[i], &freq);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("core=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_rail_isofreq_policy (output guarded) ----
TEST_F(CpuIntegration, GetRailIsofreqPolicy_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_rail_isofreq_policy", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_rail_isofreq_policy(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetRailIsofreqPolicy_InvalidHandle) {
  uint8_t policy = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_rail_isofreq_policy", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_rail_isofreq_policy(kInvalidHandle, &policy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetRailIsofreqPolicy_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_rail_isofreq_policy");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t policy = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_rail_isofreq_policy", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_rail_isofreq_policy(cpus()[i], &policy);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_set_cpu_rail_isofreq_policy (invalid input only; valid-input cases are in
// functional/) ----
TEST_F(CpuIntegration, SetRailIsofreqPolicy_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_rail_isofreq_policy", "policy=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_rail_isofreq_policy(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, SetRailIsofreqPolicy_InvalidHandle) {
  bool policy = false;
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_rail_isofreq_policy", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_rail_isofreq_policy(kInvalidHandle, &policy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---- amdsmi_get_cpu_dfc_ctrl (output guarded) ----
TEST_F(CpuIntegration, GetDfcCtrl_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_dfc_ctrl", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_dfc_ctrl(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetDfcCtrl_InvalidHandle) {
  uint8_t dfc = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_dfc_ctrl", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_dfc_ctrl(kInvalidHandle, &dfc);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetDfcCtrl_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_dfc_ctrl");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t dfc = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_dfc_ctrl", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_dfc_ctrl(cpus()[i], &dfc);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_set_cpu_dfc_ctrl (invalid input only; valid-input cases are in functional/) ----
TEST_F(CpuIntegration, SetDfcCtrl_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_dfc_ctrl", "dfc=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_dfc_ctrl(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, SetDfcCtrl_InvalidHandle) {
  uint8_t dfc = 0;
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_dfc_ctrl", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_dfc_ctrl(kInvalidHandle, &dfc);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

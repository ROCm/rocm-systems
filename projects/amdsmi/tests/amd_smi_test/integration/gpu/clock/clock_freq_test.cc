// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

static constexpr amdsmi_clk_type_t kClkTypes[] = {
    AMDSMI_CLK_TYPE_SYS,   AMDSMI_CLK_TYPE_DF,   AMDSMI_CLK_TYPE_DCEF,  AMDSMI_CLK_TYPE_SOC,
    AMDSMI_CLK_TYPE_MEM,   AMDSMI_CLK_TYPE_PCIE, AMDSMI_CLK_TYPE_VCLK0, AMDSMI_CLK_TYPE_VCLK1,
    AMDSMI_CLK_TYPE_DCLK0, AMDSMI_CLK_TYPE_DCLK1};

static constexpr amdsmi_clk_limit_type_t kClkLimitTypes[] = {AMDSMI_CLK_LIMIT_MIN,
                                                             AMDSMI_CLK_LIMIT_MAX};

// ---------------- amdsmi_get_clk_freq (enum) ----------------
TEST(GpuIntegration, GetClkFreq_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_clk_freq(any_gpu(), AMDSMI_CLK_TYPE_SYS, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetClkFreq_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_frequencies_t f;
  memset(&f, 0, sizeof(f));
  DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_clk_freq(kInvalidHandle, AMDSMI_CLK_TYPE_SYS, &f);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetClkFreq_AllGpusAllTypes) {
  AMDSMI_API_TEST_SCOPE();
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_clk_freq returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause unknown, "
         "under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_clk_freq");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto ct : kClkTypes) {
      amdsmi_frequencies_t f;
      memset(&f, 0, sizeof(f));
      DISPLAY_AMDSMI_API("amdsmi_get_clk_freq",
                         "gpu=" + std::to_string(i) + " clk=" + std::to_string(ct), kVerbose);
      amdsmi_status_t err = amdsmi_get_clk_freq(gpus()[i], ct, &f);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " clk=" + std::to_string(ct), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_clock_info (enum) ----------------
TEST(GpuIntegration, GetClockInfo_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_clock_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_clock_info(any_gpu(), AMDSMI_CLK_TYPE_SYS, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetClockInfo_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_clk_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_clock_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_clock_info(kInvalidHandle, AMDSMI_CLK_TYPE_SYS, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetClockInfo_AllGpusAllTypes) {
  AMDSMI_API_TEST_SCOPE();
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_clk_info returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause unknown, "
         "under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_clock_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto ct : kClkTypes) {
      amdsmi_clk_info_t info;
      memset(&info, 0, sizeof(info));
      DISPLAY_AMDSMI_API("amdsmi_get_clock_info",
                         "gpu=" + std::to_string(i) + " clk=" + std::to_string(ct), kVerbose);
      amdsmi_status_t err = amdsmi_get_clock_info(gpus()[i], ct, &info);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " clk=" + std::to_string(ct), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_soc_pstate ----------------
TEST(GpuIntegration, GetSocPstate_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_soc_pstate", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_soc_pstate(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetSocPstate_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_dpm_policy_t policy;
  memset(&policy, 0, sizeof(policy));
  DISPLAY_AMDSMI_API("amdsmi_get_soc_pstate", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_soc_pstate(kInvalidHandle, &policy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetSocPstate_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_soc_pstate");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dpm_policy_t policy;
    memset(&policy, 0, sizeof(policy));
    DISPLAY_AMDSMI_API("amdsmi_get_soc_pstate", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_soc_pstate(gpus()[i], &policy);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_xgmi_plpd ----------------
TEST(GpuIntegration, GetXgmiPlpd_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_xgmi_plpd", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_xgmi_plpd(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetXgmiPlpd_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_dpm_policy_t plpd;
  memset(&plpd, 0, sizeof(plpd));
  DISPLAY_AMDSMI_API("amdsmi_get_xgmi_plpd", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_xgmi_plpd(kInvalidHandle, &plpd);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetXgmiPlpd_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_xgmi_plpd");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dpm_policy_t plpd;
    memset(&plpd, 0, sizeof(plpd));
    DISPLAY_AMDSMI_API("amdsmi_get_xgmi_plpd", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_xgmi_plpd(gpus()[i], &plpd);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_set_clk_freq (SET, enum) ----------------
TEST(GpuIntegration, SetClkFreq_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_clk_freq", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_clk_freq(kInvalidHandle, AMDSMI_CLK_TYPE_SYS, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_clk_limit (SET, two enums) ----------------
TEST(GpuIntegration, SetClkLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_clk_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_clk_limit(kInvalidHandle, AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_LIMIT_MAX, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_soc_pstate (SET) ----------------
TEST(GpuIntegration, SetSocPstate_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_soc_pstate", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_soc_pstate(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_xgmi_plpd (SET) ----------------
TEST(GpuIntegration, SetXgmiPlpd_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_xgmi_plpd", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_xgmi_plpd(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

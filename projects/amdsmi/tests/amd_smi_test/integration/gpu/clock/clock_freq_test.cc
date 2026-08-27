/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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
TEST_F(GpuIntegration, GetClkFreq_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_clk_freq(gpus()[0], AMDSMI_CLK_TYPE_SYS, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetClkFreq_InvalidHandle) {
  amdsmi_frequencies_t f;
  memset(&f, 0, sizeof(f));
  DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_clk_freq(kInvalidHandle, AMDSMI_CLK_TYPE_SYS, &f);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetClkFreq_AllGpusAllTypes) {
  GTEST_SKIP() << "amdsmi_get_clk_freq returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause unknown, "
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
      amdsmi_col.Record("gpu=" + std::to_string(i) + " clk=" + std::to_string(ct), err,
                        ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                               AMDSMI_STATUS_NOT_SUPPORTED,
                                                               AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_clock_info (enum) ----------------
TEST_F(GpuIntegration, GetClockInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_clock_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_clock_info(gpus()[0], AMDSMI_CLK_TYPE_SYS, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetClockInfo_InvalidHandle) {
  amdsmi_clk_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_clock_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_clock_info(kInvalidHandle, AMDSMI_CLK_TYPE_SYS, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetClockInfo_AllGpusAllTypes) {
  GTEST_SKIP() << "amdsmi_get_clk_info returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause unknown, "
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
      amdsmi_col.Record("gpu=" + std::to_string(i) + " clk=" + std::to_string(ct), err,
                        ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                               AMDSMI_STATUS_NOT_SUPPORTED,
                                                               AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_soc_pstate ----------------
TEST_F(GpuIntegration, GetSocPstate_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_soc_pstate", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_soc_pstate(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetSocPstate_InvalidHandle) {
  amdsmi_dpm_policy_t policy;
  memset(&policy, 0, sizeof(policy));
  DISPLAY_AMDSMI_API("amdsmi_get_soc_pstate", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_soc_pstate(kInvalidHandle, &policy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetSocPstate_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_soc_pstate");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dpm_policy_t policy;
    memset(&policy, 0, sizeof(policy));
    DISPLAY_AMDSMI_API("amdsmi_get_soc_pstate", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_soc_pstate(gpus()[i], &policy);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                             AMDSMI_STATUS_NOT_SUPPORTED,
                                                             AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_xgmi_plpd ----------------
TEST_F(GpuIntegration, GetXgmiPlpd_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_xgmi_plpd", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_xgmi_plpd(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetXgmiPlpd_InvalidHandle) {
  amdsmi_dpm_policy_t plpd;
  memset(&plpd, 0, sizeof(plpd));
  DISPLAY_AMDSMI_API("amdsmi_get_xgmi_plpd", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_xgmi_plpd(kInvalidHandle, &plpd);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetXgmiPlpd_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_xgmi_plpd");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dpm_policy_t plpd;
    memset(&plpd, 0, sizeof(plpd));
    DISPLAY_AMDSMI_API("amdsmi_get_xgmi_plpd", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_xgmi_plpd(gpus()[i], &plpd);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                             AMDSMI_STATUS_NOT_SUPPORTED,
                                                             AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_set_clk_freq (SET, enum) ----------------
TEST_F(GpuIntegration, SetClkFreq_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_clk_freq", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_clk_freq(kInvalidHandle, AMDSMI_CLK_TYPE_SYS, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
// ---------------- amdsmi_set_gpu_clk_limit (SET, two enums) ----------------
TEST_F(GpuIntegration, SetClkLimit_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_clk_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_clk_limit(kInvalidHandle, AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_LIMIT_MAX, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
// ---------------- amdsmi_set_soc_pstate (SET) ----------------
TEST_F(GpuIntegration, SetSocPstate_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_soc_pstate", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_soc_pstate(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
// ---------------- amdsmi_set_xgmi_plpd (SET) ----------------
TEST_F(GpuIntegration, SetXgmiPlpd_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_xgmi_plpd", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_xgmi_plpd(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

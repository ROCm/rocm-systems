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

#include <cstdint>
#include <cstring>
#include <string>

#include "unit/unit_test_framework.h"

using amdsmi::unittest::kInvalidHandle;
using amdsmi::unittest::kVerbose;

// CPU interconnect, P-state, DPM, APB and bandwidth APIs (socket scoped).
// Writes that reconfigure link width, PCIe link rate, DF/xGMI P-states, LCLK
// DPM levels or APB would destabilize live hardware, so those are covered by
// the invalid-handle test only. The PC6/CC6 enable writes read the current
// value back and rewrite it (a no-op) for their valid path.

namespace {
constexpr amdsmi_io_bw_encoding_t kBwTypes[] = {AMDSMI_AGG_BW0, AMDSMI_RD_BW0, AMDSMI_WR_BW0};
constexpr uint8_t kNbioIds[] = {0, 1, 2, 3};
}  // namespace

// ---- amdsmi_set_cpu_xgmi_width (invalid-handle only; live write is unsafe) ----
TEST_F(CpuUnit, SetXgmiWidth_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_xgmi_width", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_xgmi_width(kInvalidHandle, 0, 2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---- amdsmi_set_cpu_gmi3_link_width_range (invalid-handle only) ----
TEST_F(CpuUnit, SetGmi3LinkWidthRange_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_gmi3_link_width_range", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_gmi3_link_width_range(kInvalidHandle, 0, 2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---- amdsmi_cpu_apb_enable / amdsmi_cpu_apb_disable (invalid-handle only) ----
TEST_F(CpuUnit, ApbEnable_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_cpu_apb_enable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_cpu_apb_enable(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuUnit, ApbDisable_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_cpu_apb_disable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_cpu_apb_disable(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---- amdsmi_set_cpu_socket_lclk_dpm_level (invalid-handle only) ----
TEST_F(CpuUnit, SetSocketLclkDpmLevel_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_socket_lclk_dpm_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_socket_lclk_dpm_level(kInvalidHandle, 0, 0, 1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---- amdsmi_get_cpu_socket_lclk_dpm_level (handle guarded only) ----
TEST_F(CpuUnit, GetSocketLclkDpmLevel_InvalidHandle) {
  amdsmi_dpm_level_t nbio;
  memset(&nbio, 0, sizeof(nbio));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_lclk_dpm_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_lclk_dpm_level(kInvalidHandle, 0, &nbio);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuUnit, GetSocketLclkDpmLevel_AllCpusAllNbio) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_cpu_socket_lclk_dpm_level");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i)
    for (auto nbio_id : kNbioIds) {
      amdsmi_dpm_level_t nbio;
      memset(&nbio, 0, sizeof(nbio));
      DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_lclk_dpm_level",
                         "cpu=" + std::to_string(i) + " nbio=" + std::to_string(nbio_id), kVerbose);
      amdsmi_status_t err = amdsmi_get_cpu_socket_lclk_dpm_level(cpus()[i], nbio_id, &nbio);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("cpu=" + std::to_string(i) + " nbio=" + std::to_string(nbio_id), err,
                        ::amdsmi::unittest::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---- amdsmi_set_cpu_pcie_link_rate (invalid-handle only) ----
TEST_F(CpuUnit, SetPcieLinkRate_InvalidHandle) {
  uint8_t prev_mode = 0;
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_pcie_link_rate", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_pcie_link_rate(kInvalidHandle, 0, &prev_mode);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---- amdsmi_set_cpu_df_pstate_range (invalid-handle only) ----
TEST_F(CpuUnit, SetDfPstateRange_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_df_pstate_range", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_df_pstate_range(kInvalidHandle, 0, 2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---- amdsmi_set_cpu_xgmi_pstate_range (invalid-handle only) ----
TEST_F(CpuUnit, SetXgmiPstateRange_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_xgmi_pstate_range", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_xgmi_pstate_range(kInvalidHandle, 0, 2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---- amdsmi_get_cpu_xgmi_pstate_range (outputs guarded) ----
TEST_F(CpuUnit, GetXgmiPstateRange_NullOutput) {
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  uint8_t max_pstate = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_xgmi_pstate_range", "min_pstate=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_xgmi_pstate_range(cpus()[0], nullptr, &max_pstate);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(CpuUnit, GetXgmiPstateRange_InvalidHandle) {
  uint8_t min_pstate = 0, max_pstate = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_xgmi_pstate_range", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_xgmi_pstate_range(kInvalidHandle, &min_pstate, &max_pstate);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuUnit, GetXgmiPstateRange_AllCpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_cpu_xgmi_pstate_range");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t min_pstate = 0, max_pstate = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_xgmi_pstate_range", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_xgmi_pstate_range(cpus()[i], &min_pstate, &max_pstate);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("cpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---- amdsmi_get_cpu_pc6_enable (output guarded) ----
TEST_F(CpuUnit, GetPc6Enable_NullOutput) {
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_pc6_enable", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_pc6_enable(cpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(CpuUnit, GetPc6Enable_InvalidHandle) {
  uint8_t enabled = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_pc6_enable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_pc6_enable(kInvalidHandle, &enabled);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuUnit, GetPc6Enable_AllCpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_cpu_pc6_enable");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t enabled = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_pc6_enable", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_pc6_enable(cpus()[i], &enabled);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("cpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---- amdsmi_set_cpu_pc6_enable (write; restores current value) ----
TEST_F(CpuUnit, SetPc6Enable_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_pc6_enable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_pc6_enable(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuUnit, SetPc6Enable_AllCpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_cpu_pc6_enable");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t enabled = 0;
    if (amdsmi_get_cpu_pc6_enable(cpus()[i], &enabled) != AMDSMI_STATUS_SUCCESS) continue;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_pc6_enable",
                       "cpu=" + std::to_string(i) + " enable=" + std::to_string(enabled), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_pc6_enable(cpus()[i], enabled);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    amdsmi_col.Record("cpu=" + std::to_string(i) + " enable=" + std::to_string(enabled), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---- amdsmi_get_cpu_cc6_enable (output guarded) ----
TEST_F(CpuUnit, GetCc6Enable_NullOutput) {
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_cc6_enable", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_cc6_enable(cpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(CpuUnit, GetCc6Enable_InvalidHandle) {
  uint8_t enabled = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_cc6_enable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_cc6_enable(kInvalidHandle, &enabled);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuUnit, GetCc6Enable_AllCpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_cpu_cc6_enable");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t enabled = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_cc6_enable", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_cc6_enable(cpus()[i], &enabled);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("cpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---- amdsmi_set_cpu_cc6_enable (write; restores current value) ----
TEST_F(CpuUnit, SetCc6Enable_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_cc6_enable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_cc6_enable(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuUnit, SetCc6Enable_AllCpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_cpu_cc6_enable");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t enabled = 0;
    if (amdsmi_get_cpu_cc6_enable(cpus()[i], &enabled) != AMDSMI_STATUS_SUCCESS) continue;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_cc6_enable",
                       "cpu=" + std::to_string(i) + " enable=" + std::to_string(enabled), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_cc6_enable(cpus()[i], enabled);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    amdsmi_col.Record("cpu=" + std::to_string(i) + " enable=" + std::to_string(enabled), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---- amdsmi_get_cpu_current_io_bandwidth (handle guarded only; loops bw type) ----
TEST_F(CpuUnit, GetCurrentIoBandwidth_InvalidHandle) {
  amdsmi_link_id_bw_type_t link;
  memset(&link, 0, sizeof(link));
  char lname[] = "G0";
  link.bw_type = AMDSMI_AGG_BW0;
  link.link_name = lname;
  uint32_t io_bw = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_current_io_bandwidth", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_current_io_bandwidth(kInvalidHandle, link, &io_bw);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuUnit, GetCurrentIoBandwidth_AllCpusAllBwTypes) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_cpu_current_io_bandwidth");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i)
    for (auto bw_type : kBwTypes) {
      amdsmi_link_id_bw_type_t link;
      memset(&link, 0, sizeof(link));
      char lname[] = "G0";
      link.bw_type = bw_type;
      link.link_name = lname;
      uint32_t io_bw = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_cpu_current_io_bandwidth",
                         "cpu=" + std::to_string(i) + " bw_type=" + std::to_string(bw_type),
                         kVerbose);
      amdsmi_status_t err = amdsmi_get_cpu_current_io_bandwidth(cpus()[i], link, &io_bw);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("cpu=" + std::to_string(i) + " bw_type=" + std::to_string(bw_type), err,
                        ::amdsmi::unittest::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---- amdsmi_get_cpu_current_xgmi_bw (handle guarded only; loops bw type) ----
TEST_F(CpuUnit, GetCurrentXgmiBw_InvalidHandle) {
  amdsmi_link_id_bw_type_t link;
  memset(&link, 0, sizeof(link));
  char lname[] = "P0";
  link.bw_type = AMDSMI_AGG_BW0;
  link.link_name = lname;
  uint32_t xgmi_bw = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_current_xgmi_bw", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_current_xgmi_bw(kInvalidHandle, link, &xgmi_bw);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuUnit, GetCurrentXgmiBw_AllCpusAllBwTypes) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_cpu_current_xgmi_bw");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i)
    for (auto bw_type : kBwTypes) {
      amdsmi_link_id_bw_type_t link;
      memset(&link, 0, sizeof(link));
      char lname[] = "P0";
      link.bw_type = bw_type;
      link.link_name = lname;
      uint32_t xgmi_bw = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_cpu_current_xgmi_bw",
                         "cpu=" + std::to_string(i) + " bw_type=" + std::to_string(bw_type),
                         kVerbose);
      amdsmi_status_t err = amdsmi_get_cpu_current_xgmi_bw(cpus()[i], link, &xgmi_bw);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("cpu=" + std::to_string(i) + " bw_type=" + std::to_string(bw_type), err,
                        ::amdsmi::unittest::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

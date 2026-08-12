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

#include "unit/unit_test_framework.h"

using amdsmi::unittest::AmdsmiStatusIsExpected;
using amdsmi::unittest::kInvalidHandle;
using amdsmi::unittest::kVerbose;

// amdsmi_get_cpu_pc6_enable / amdsmi_set_cpu_pc6_enable.
// amdsmi_get_cpu_cc6_enable / amdsmi_set_cpu_cc6_enable.
// amdsmi_get_cpu_rail_isofreq_policy / amdsmi_set_cpu_rail_isofreq_policy.
// amdsmi_get_cpu_dfc_ctrl / amdsmi_set_cpu_dfc_ctrl.
// amdsmi_get_cpu_xgmi_pstate_range / amdsmi_set_cpu_xgmi_pstate_range.
// amdsmi_get_cpu_socket_lclk_dpm_level / amdsmi_set_cpu_socket_lclk_dpm_level.
// Link/pstate setters with no getter.
// ---- amdsmi_set_cpu_pc6_enable (0/1 toggle) ----
TEST_F(CpuFunctionalReadWrite, SetPc6Enable_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_pc6_enable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_pc6_enable(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, Pc6Enable_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_cpu_pc6_enable");
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t initial = 0;
    if (amdsmi_get_cpu_pc6_enable(cpus()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;

    uint8_t target = initial ? 0 : 1;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_pc6_enable",
                       "cpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_pc6_enable(cpus()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint8_t readback = 0;
      if (amdsmi_get_cpu_pc6_enable(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "cpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_cpu_pc6_enable(cpus()[i], initial);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore pc6";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_pc6_enable(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "cpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

// ---- amdsmi_set_cpu_cc6_enable (0/1 toggle) ----
TEST_F(CpuFunctionalReadWrite, SetCc6Enable_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_cc6_enable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_cc6_enable(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, Cc6Enable_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_cpu_cc6_enable");
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t initial = 0;
    if (amdsmi_get_cpu_cc6_enable(cpus()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;

    uint8_t target = initial ? 0 : 1;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_cc6_enable",
                       "cpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_cc6_enable(cpus()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint8_t readback = 0;
      if (amdsmi_get_cpu_cc6_enable(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "cpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_cpu_cc6_enable(cpus()[i], initial);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore cc6";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_cc6_enable(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "cpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

// ---- amdsmi_set_cpu_rail_isofreq_policy (bool toggle; setter takes bool*) ----
TEST_F(CpuFunctionalReadWrite, SetRailIsofreqPolicy_InvalidHandle) {
  RequireInit();
  bool policy = false;
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_rail_isofreq_policy", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_rail_isofreq_policy(kInvalidHandle, &policy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, SetRailIsofreqPolicy_NullOutput) {
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_rail_isofreq_policy", "policy=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_rail_isofreq_policy(cpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST_F(CpuFunctionalReadWrite, RailIsofreqPolicy_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_cpu_rail_isofreq_policy");
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t initial = 0;
    if (amdsmi_get_cpu_rail_isofreq_policy(cpus()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;

    bool target = initial ? false : true;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_rail_isofreq_policy",
                       "cpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_rail_isofreq_policy(cpus()[i], &target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint8_t readback = 0;
      if (amdsmi_get_cpu_rail_isofreq_policy(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target ? 1 : 0) << "cpu=" << i << " set did not take effect";
      }
      bool restore = (initial != 0);
      amdsmi_status_t rerr = amdsmi_set_cpu_rail_isofreq_policy(cpus()[i], &restore);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore rail policy";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_rail_isofreq_policy(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "cpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

// ---- amdsmi_set_cpu_dfc_ctrl (uint8_t* payload) ----
TEST_F(CpuFunctionalReadWrite, SetDfcCtrl_InvalidHandle) {
  RequireInit();
  uint8_t dfc = 0;
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_dfc_ctrl", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_dfc_ctrl(kInvalidHandle, &dfc);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, SetDfcCtrl_NullOutput) {
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_dfc_ctrl", "dfc=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_dfc_ctrl(cpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST_F(CpuFunctionalReadWrite, DfcCtrl_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_cpu_dfc_ctrl");
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t initial = 0;
    if (amdsmi_get_cpu_dfc_ctrl(cpus()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;

    uint8_t target = initial ? 0 : 1;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_dfc_ctrl", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_dfc_ctrl(cpus()[i], &target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint8_t readback = 0;
      if (amdsmi_get_cpu_dfc_ctrl(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "cpu=" << i << " set did not take effect";
      }
      uint8_t restore = initial;
      amdsmi_status_t rerr = amdsmi_set_cpu_dfc_ctrl(cpus()[i], &restore);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore dfc ctrl";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_dfc_ctrl(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "cpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

// ---- amdsmi_set_cpu_xgmi_pstate_range (min/max) ----
TEST_F(CpuFunctionalReadWrite, SetXgmiPstateRange_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_xgmi_pstate_range", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_xgmi_pstate_range(kInvalidHandle, 0, 2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, XgmiPstateRange_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_cpu_xgmi_pstate_range");
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t init_min = 0;
    uint8_t init_max = 0;
    if (amdsmi_get_cpu_xgmi_pstate_range(cpus()[i], &init_min, &init_max) != AMDSMI_STATUS_SUCCESS)
      continue;

    uint8_t target_min = 0;
    uint8_t target_max = 2;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_xgmi_pstate_range", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_xgmi_pstate_range(cpus()[i], target_min, target_max);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      // restore original range
      amdsmi_status_t rerr = amdsmi_set_cpu_xgmi_pstate_range(cpus()[i], init_min, init_max);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS)
          << "cpu=" << i << " failed to restore xgmi pstate range";
    }
  }
  col.ExpectNoFailures();
}

// ---- amdsmi_set_cpu_socket_lclk_dpm_level (nbio id + min/max) ----
TEST_F(CpuFunctionalReadWrite, SetSocketLclkDpmLevel_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_socket_lclk_dpm_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_socket_lclk_dpm_level(kInvalidHandle, 0, 0, 1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, SocketLclkDpmLevel_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_cpu_socket_lclk_dpm_level");
  const uint8_t nbio_id = 0;
  for (size_t i = 0; i < cpus().size(); ++i) {
    amdsmi_dpm_level_t initial;
    std::memset(&initial, 0, sizeof(initial));
    if (amdsmi_get_cpu_socket_lclk_dpm_level(cpus()[i], nbio_id, &initial) != AMDSMI_STATUS_SUCCESS)
      continue;

    uint8_t target_min = 0;
    uint8_t target_max = 1;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_socket_lclk_dpm_level", "cpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err =
        amdsmi_set_cpu_socket_lclk_dpm_level(cpus()[i], nbio_id, target_min, target_max);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      amdsmi_status_t rerr = amdsmi_set_cpu_socket_lclk_dpm_level(
          cpus()[i], nbio_id, initial.min_dpm_level, initial.max_dpm_level);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore lclk dpm level";
    }
  }
  col.ExpectNoFailures();
}

// ---- link/pstate setters with no getter ----
TEST_F(CpuFunctionalReadWrite, SetXgmiWidth_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_xgmi_width", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_xgmi_width(kInvalidHandle, 0, 1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, SetGmi3LinkWidthRange_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_gmi3_link_width_range", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_gmi3_link_width_range(kInvalidHandle, 0, 1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, SetPcieLinkRate_InvalidHandle) {
  RequireInit();
  uint8_t prev_mode = 0;
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_pcie_link_rate", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_pcie_link_rate(kInvalidHandle, 0, &prev_mode);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, SetDfPstateRange_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_df_pstate_range", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_df_pstate_range(kInvalidHandle, 0, 2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, ApbEnable_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_cpu_apb_enable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_cpu_apb_enable(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, ApbDisable_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_cpu_apb_disable", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_cpu_apb_disable(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, LinkSetters_Set) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_cpu_link_setters");
  for (size_t i = 0; i < cpus().size(); ++i) {
    const std::string tag = "cpu=" + std::to_string(i);

    DISPLAY_AMDSMI_API("amdsmi_set_cpu_xgmi_width", tag, kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_xgmi_width(cpus()[i], 0, 1);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("xgmi_width " + tag, err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));

    DISPLAY_AMDSMI_API("amdsmi_set_cpu_gmi3_link_width_range", tag, kVerbose);
    err = amdsmi_set_cpu_gmi3_link_width_range(cpus()[i], 0, 1);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("gmi3_link_width " + tag, err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));

    uint8_t prev_mode = 0;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_pcie_link_rate", tag, kVerbose);
    err = amdsmi_set_cpu_pcie_link_rate(cpus()[i], 0, &prev_mode);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("pcie_link_rate " + tag, err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));

    DISPLAY_AMDSMI_API("amdsmi_set_cpu_df_pstate_range", tag, kVerbose);
    err = amdsmi_set_cpu_df_pstate_range(cpus()[i], 0, 2);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("df_pstate_range " + tag, err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));

    DISPLAY_AMDSMI_API("amdsmi_cpu_apb_enable", tag, kVerbose);
    err = amdsmi_cpu_apb_enable(cpus()[i]);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("apb_enable " + tag, err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));

    DISPLAY_AMDSMI_API("amdsmi_cpu_apb_disable", tag, kVerbose);
    err = amdsmi_cpu_apb_disable(cpus()[i], 0);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("apb_disable " + tag, err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));
  }
  col.ExpectNoFailures();
}

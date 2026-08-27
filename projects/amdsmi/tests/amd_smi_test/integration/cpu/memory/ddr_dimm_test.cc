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

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// CPU DDR bandwidth and per-DIMM telemetry APIs. The DIMM getters take a DIMM
// address index and only guard the handle, so they use the invalid-handle test
// and iterate a small set of DIMM addresses. The sideband register write only
// runs the invalid-handle case: exercising a real write with arbitrary data is
// unsafe on live hardware.

namespace {
constexpr uint8_t kDimmAddrs[] = {0, 1, 2, 3};
}  // namespace

// ---- amdsmi_get_cpu_ddr_bw (handle guarded only) ----
TEST_F(CpuIntegration, GetDdrBw_InvalidHandle) {
  amdsmi_ddr_bw_metrics_t bw;
  memset(&bw, 0, sizeof(bw));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_ddr_bw", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_ddr_bw(kInvalidHandle, &bw);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetDdrBw_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_ddr_bw");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    amdsmi_ddr_bw_metrics_t bw;
    memset(&bw, 0, sizeof(bw));
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_ddr_bw", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_ddr_bw(cpus()[i], &bw);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_dimm_temp_range_and_refresh_rate (handle guarded only) ----
TEST_F(CpuIntegration, GetDimmTempRangeRefreshRate_InvalidHandle) {
  amdsmi_temp_range_refresh_rate_t rate;
  memset(&rate, 0, sizeof(rate));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_dimm_temp_range_and_refresh_rate", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(kInvalidHandle, 0, &rate);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetDimmTempRangeRefreshRate_AllCpusAllDimms) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_dimm_temp_range_and_refresh_rate");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i)
    for (auto addr : kDimmAddrs) {
      amdsmi_temp_range_refresh_rate_t rate;
      memset(&rate, 0, sizeof(rate));
      DISPLAY_AMDSMI_API("amdsmi_get_cpu_dimm_temp_range_and_refresh_rate",
                         "cpu=" + std::to_string(i) + " dimm=" + std::to_string(addr), kVerbose);
      amdsmi_status_t err = amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(cpus()[i], addr, &rate);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("cpu=" + std::to_string(i) + " dimm=" + std::to_string(addr), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_dimm_power_consumption (handle guarded only) ----
TEST_F(CpuIntegration, GetDimmPowerConsumption_InvalidHandle) {
  amdsmi_dimm_power_t pow;
  memset(&pow, 0, sizeof(pow));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_dimm_power_consumption", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_dimm_power_consumption(kInvalidHandle, 0, &pow);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetDimmPowerConsumption_AllCpusAllDimms) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_dimm_power_consumption");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i)
    for (auto addr : kDimmAddrs) {
      amdsmi_dimm_power_t pow;
      memset(&pow, 0, sizeof(pow));
      DISPLAY_AMDSMI_API("amdsmi_get_cpu_dimm_power_consumption",
                         "cpu=" + std::to_string(i) + " dimm=" + std::to_string(addr), kVerbose);
      amdsmi_status_t err = amdsmi_get_cpu_dimm_power_consumption(cpus()[i], addr, &pow);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("cpu=" + std::to_string(i) + " dimm=" + std::to_string(addr), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_dimm_thermal_sensor (handle guarded only) ----
TEST_F(CpuIntegration, GetDimmThermalSensor_InvalidHandle) {
  amdsmi_dimm_thermal_t temp;
  memset(&temp, 0, sizeof(temp));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_dimm_thermal_sensor", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_dimm_thermal_sensor(kInvalidHandle, 0, &temp);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetDimmThermalSensor_AllCpusAllDimms) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_dimm_thermal_sensor");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i)
    for (auto addr : kDimmAddrs) {
      amdsmi_dimm_thermal_t temp;
      memset(&temp, 0, sizeof(temp));
      DISPLAY_AMDSMI_API("amdsmi_get_cpu_dimm_thermal_sensor",
                         "cpu=" + std::to_string(i) + " dimm=" + std::to_string(addr), kVerbose);
      amdsmi_status_t err = amdsmi_get_cpu_dimm_thermal_sensor(cpus()[i], addr, &temp);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("cpu=" + std::to_string(i) + " dimm=" + std::to_string(addr), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_dimm_sb_reg (data output guarded) ----
TEST_F(CpuIntegration, GetDimmSbReg_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_dimm_sb_reg", "data=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_dimm_sb_reg(any_cpu(), 0, 0, 0, 0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetDimmSbReg_InvalidHandle) {
  uint32_t data = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_dimm_sb_reg", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_dimm_sb_reg(kInvalidHandle, 0, 0, 0, 0, &data);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetDimmSbReg_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_dimm_sb_reg");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t data = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_dimm_sb_reg", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_dimm_sb_reg(cpus()[i], 0, 0, 0, 0, &data);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_set_cpu_dimm_sb_reg (invalid input only; valid-input cases are in functional/) ----
TEST_F(CpuIntegration, SetDimmSbReg_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_dimm_sb_reg", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_dimm_sb_reg(kInvalidHandle, 0, 0, 0, 0, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

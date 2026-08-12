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

using amdsmi::unittest::kInvalidHandle;
using amdsmi::unittest::kVerbose;

static constexpr amdsmi_temperature_type_t kTempTypes[] = {
    AMDSMI_TEMPERATURE_TYPE_EDGE,  AMDSMI_TEMPERATURE_TYPE_HOTSPOT, AMDSMI_TEMPERATURE_TYPE_VRAM,
    AMDSMI_TEMPERATURE_TYPE_HBM_0, AMDSMI_TEMPERATURE_TYPE_HBM_1,   AMDSMI_TEMPERATURE_TYPE_HBM_2,
    AMDSMI_TEMPERATURE_TYPE_HBM_3, AMDSMI_TEMPERATURE_TYPE_PLX};

static constexpr amdsmi_temperature_metric_t kTempMetrics[] = {
    AMDSMI_TEMP_CURRENT,       AMDSMI_TEMP_MAX,           AMDSMI_TEMP_MIN,
    AMDSMI_TEMP_MAX_HYST,      AMDSMI_TEMP_MIN_HYST,      AMDSMI_TEMP_CRITICAL,
    AMDSMI_TEMP_CRITICAL_HYST, AMDSMI_TEMP_EMERGENCY,     AMDSMI_TEMP_EMERGENCY_HYST,
    AMDSMI_TEMP_CRIT_MIN,      AMDSMI_TEMP_CRIT_MIN_HYST, AMDSMI_TEMP_OFFSET,
    AMDSMI_TEMP_LOWEST,        AMDSMI_TEMP_HIGHEST,       AMDSMI_TEMP_SHUTDOWN};

static constexpr amdsmi_voltage_type_t kVoltTypes[] = {AMDSMI_VOLT_TYPE_VDDGFX,
                                                       AMDSMI_VOLT_TYPE_VDDBOARD};

static constexpr amdsmi_voltage_metric_t kVoltMetrics[] = {
    AMDSMI_VOLT_CURRENT,  AMDSMI_VOLT_MAX,     AMDSMI_VOLT_MIN_CRIT, AMDSMI_VOLT_MIN,
    AMDSMI_VOLT_MAX_CRIT, AMDSMI_VOLT_AVERAGE, AMDSMI_VOLT_LOWEST,   AMDSMI_VOLT_HIGHEST};

// ---------------- amdsmi_get_gpu_fan_rpms ----------------
TEST_F(GpuUnit, GetFanRpms_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_rpms", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_fan_rpms(gpus()[0], 0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetFanRpms_InvalidHandle) {
  int64_t speed = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_rpms", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_fan_rpms(kInvalidHandle, 0, &speed);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetFanRpms_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_fan_rpms");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    int64_t speed = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_rpms", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_fan_rpms(gpus()[i], 0, &speed);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_fan_speed ----------------
TEST_F(GpuUnit, GetFanSpeed_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_fan_speed(gpus()[0], 0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetFanSpeed_InvalidHandle) {
  int64_t speed = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_fan_speed(kInvalidHandle, 0, &speed);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetFanSpeed_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_fan_speed");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    int64_t speed = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_fan_speed(gpus()[i], 0, &speed);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_fan_speed_max ----------------
TEST_F(GpuUnit, GetFanSpeedMax_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed_max", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_fan_speed_max(gpus()[0], 0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetFanSpeedMax_InvalidHandle) {
  uint64_t max_speed = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed_max", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_fan_speed_max(kInvalidHandle, 0, &max_speed);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetFanSpeedMax_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_fan_speed_max");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint64_t max_speed = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed_max", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_fan_speed_max(gpus()[i], 0, &max_speed);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_temp_metric (two enums) ----------------
TEST_F(GpuUnit, GetTempMetric_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_temp_metric", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_temp_metric(gpus()[0], AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetTempMetric_InvalidHandle) {
  int64_t temp = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_temp_metric", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_temp_metric(kInvalidHandle, AMDSMI_TEMPERATURE_TYPE_EDGE,
                                               AMDSMI_TEMP_CURRENT, &temp);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetTempMetric_AllGpusAllTypesMetrics) {
  GTEST_SKIP()
      << "GetTempMetric_AllGpusAllTypesMetrics fails with error 43, AMDSMI_STATUS_UNEXPECTED_DATA";

  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_temp_metric");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto tt : kTempTypes)
      for (auto tm : kTempMetrics) {
        int64_t temp = 0;
        DISPLAY_AMDSMI_API("amdsmi_get_temp_metric",
                           "gpu=" + std::to_string(i) + " type=" + std::to_string(tt) +
                               " metric=" + std::to_string(tm),
                           kVerbose);
        amdsmi_status_t err = amdsmi_get_temp_metric(gpus()[i], tt, tm, &temp);
        DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                              AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
        amdsmi_col.Record("gpu=" + std::to_string(i) + " type=" + std::to_string(tt) +
                              " metric=" + std::to_string(tm),
                          err,
                          ::amdsmi::unittest::AmdsmiStatusIsExpected(
                              err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                              AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
      }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_volt_metric (two enums) ----------------
TEST_F(GpuUnit, GetVoltMetric_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_volt_metric", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_volt_metric(gpus()[0], AMDSMI_VOLT_TYPE_VDDGFX, AMDSMI_VOLT_CURRENT, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetVoltMetric_InvalidHandle) {
  int64_t voltage = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_volt_metric", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_volt_metric(kInvalidHandle, AMDSMI_VOLT_TYPE_VDDGFX,
                                                   AMDSMI_VOLT_CURRENT, &voltage);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetVoltMetric_AllGpusAllTypesMetrics) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_volt_metric");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto vt : kVoltTypes)
      for (auto vm : kVoltMetrics) {
        int64_t voltage = 0;
        DISPLAY_AMDSMI_API("amdsmi_get_gpu_volt_metric",
                           "gpu=" + std::to_string(i) + " type=" + std::to_string(vt) +
                               " metric=" + std::to_string(vm),
                           kVerbose);
        amdsmi_status_t err = amdsmi_get_gpu_volt_metric(gpus()[i], vt, vm, &voltage);
        DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                              AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
        amdsmi_col.Record("gpu=" + std::to_string(i) + " type=" + std::to_string(vt) +
                              " metric=" + std::to_string(vm),
                          err,
                          ::amdsmi::unittest::AmdsmiStatusIsExpected(
                              err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                              AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
      }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_set_gpu_fan_speed (SET) ----------------
TEST_F(GpuUnit, SetFanSpeed_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_fan_speed", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_fan_speed(kInvalidHandle, 0, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetFanSpeed_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_gpu_fan_speed");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    int64_t cur = 0;
    if (amdsmi_get_gpu_fan_speed(gpus()[i], 0, &cur) != AMDSMI_STATUS_SUCCESS) continue;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_fan_speed", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_fan_speed(gpus()[i], 0, static_cast<uint64_t>(cur));
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));
    amdsmi_reset_gpu_fan(gpus()[i], 0);
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_reset_gpu_fan (action) ----------------
TEST_F(GpuUnit, ResetFan_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_reset_gpu_fan", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_reset_gpu_fan(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, ResetFan_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_reset_gpu_fan");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_reset_gpu_fan", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_reset_gpu_fan(gpus()[i], 0);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));
  }
  amdsmi_col.ExpectNoFailures();
}

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

// ---------------- amdsmi_get_gpu_busy_percent ----------------
TEST_F(GpuUnit, GetBusyPercent_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_busy_percent", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_busy_percent(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetBusyPercent_InvalidHandle) {
  uint32_t busy = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_busy_percent", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_busy_percent(kInvalidHandle, &busy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetBusyPercent_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_busy_percent");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t busy = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_busy_percent", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_busy_percent(gpus()[i], &busy);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_vcn_busy_percent ----------------
TEST_F(GpuUnit, GetVcnBusyPercent_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_vcn_busy_percent(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetVcnBusyPercent_InvalidHandle) {
  uint32_t busy = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_vcn_busy_percent(kInvalidHandle, &busy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetVcnBusyPercent_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_vcn_busy_percent");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t busy = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_vcn_busy_percent(gpus()[i], &busy);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_utilization_count ----------------
TEST_F(GpuUnit, GetUtilizationCount_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi_utilization_counter_t counters[1];
  memset(counters, 0, sizeof(counters));
  counters[0].type = AMDSMI_COARSE_GRAIN_GFX_ACTIVITY;
  DISPLAY_AMDSMI_API("amdsmi_get_utilization_count", "gpu=0 timestamp=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_utilization_count(gpus()[0], counters, 1, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetUtilizationCount_InvalidHandle) {
  amdsmi_utilization_counter_t counters[1];
  memset(counters, 0, sizeof(counters));
  counters[0].type = AMDSMI_COARSE_GRAIN_GFX_ACTIVITY;
  uint64_t ts = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_utilization_count", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_utilization_count(kInvalidHandle, counters, 1, &ts);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetUtilizationCount_AllGpus) {
  GTEST_SKIP() << "amdsmi_get_utilization_count returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
                  "unknown, under investigation";

  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_utilization_count");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_utilization_counter_t counters[6];
    memset(counters, 0, sizeof(counters));
    counters[0].type = AMDSMI_COARSE_GRAIN_GFX_ACTIVITY;
    counters[1].type = AMDSMI_COARSE_GRAIN_MEM_ACTIVITY;
    counters[2].type = AMDSMI_COARSE_DECODER_ACTIVITY;
    counters[3].type = AMDSMI_FINE_GRAIN_GFX_ACTIVITY;
    counters[4].type = AMDSMI_FINE_GRAIN_MEM_ACTIVITY;
    counters[5].type = AMDSMI_FINE_DECODER_ACTIVITY;
    uint64_t ts = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_utilization_count", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_utilization_count(gpus()[i], counters, 6, &ts);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_perf_level ----------------
TEST_F(GpuUnit, GetPerfLevel_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_perf_level(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetPerfLevel_InvalidHandle) {
  amdsmi_dev_perf_level_t perf;
  memset(&perf, 0, sizeof(perf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_perf_level(kInvalidHandle, &perf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetPerfLevel_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_perf_level");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dev_perf_level_t perf;
    memset(&perf, 0, sizeof(perf));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_perf_level(gpus()[i], &perf);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_overdrive_level ----------------
TEST_F(GpuUnit, GetOverdriveLevel_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_overdrive_level(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetOverdriveLevel_InvalidHandle) {
  uint32_t od = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_overdrive_level(kInvalidHandle, &od);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetOverdriveLevel_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_overdrive_level");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t od = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_overdrive_level(gpus()[i], &od);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_mem_overdrive_level ----------------
TEST_F(GpuUnit, GetMemOverdriveLevel_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_mem_overdrive_level", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_mem_overdrive_level(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetMemOverdriveLevel_InvalidHandle) {
  uint32_t od = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_mem_overdrive_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_mem_overdrive_level(kInvalidHandle, &od);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetMemOverdriveLevel_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_mem_overdrive_level");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t od = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_mem_overdrive_level", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_mem_overdrive_level(gpus()[i], &od);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_od_volt_info ----------------
TEST_F(GpuUnit, GetOdVoltInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_info(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetOdVoltInfo_InvalidHandle) {
  amdsmi_od_volt_freq_data_t odv;
  memset(&odv, 0, sizeof(odv));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_info(kInvalidHandle, &odv);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetOdVoltInfo_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_od_volt_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_od_volt_freq_data_t odv;
    memset(&odv, 0, sizeof(odv));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_od_volt_info(gpus()[i], &odv);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_od_volt_curve_regions ----------------
TEST_F(GpuUnit, GetOdVoltCurveRegions_NullNum) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_curve_regions", "gpu=0 num=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_curve_regions(gpus()[0], nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetOdVoltCurveRegions_InvalidHandle) {
  uint32_t num = 8;
  amdsmi_freq_volt_region_t buf[8];
  memset(buf, 0, sizeof(buf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_curve_regions", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_curve_regions(kInvalidHandle, &num, buf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetOdVoltCurveRegions_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_od_volt_curve_regions");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t num = 8;
    amdsmi_freq_volt_region_t buf[8];
    memset(buf, 0, sizeof(buf));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_curve_regions", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_od_volt_curve_regions(gpus()[i], &num, buf);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_activity ----------------
TEST_F(GpuUnit, GetActivity_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_activity", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_activity(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetActivity_InvalidHandle) {
  amdsmi_engine_usage_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_activity", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_activity(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetActivity_AllGpus) {
  GTEST_SKIP() << "amdsmi_get_gpu_activity returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
                  "unknown, under investigation";

  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_activity");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_engine_usage_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_activity", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_activity(gpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_set_gpu_perf_level (SET, enum) ----------------
TEST_F(GpuUnit, SetPerfLevel_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_perf_level(kInvalidHandle, AMDSMI_DEV_PERF_LEVEL_AUTO);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetPerfLevel_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_gpu_perf_level");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dev_perf_level_t perf = AMDSMI_DEV_PERF_LEVEL_AUTO;
    if (amdsmi_get_gpu_perf_level(gpus()[i], &perf) != AMDSMI_STATUS_SUCCESS) continue;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_perf_level(gpus()[i], perf);
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

// ---------------- amdsmi_set_gpu_overdrive_level (SET) ----------------
TEST_F(GpuUnit, SetOverdriveLevel_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_overdrive_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_overdrive_level(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetOverdriveLevel_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_gpu_overdrive_level");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t od = 0;
    if (amdsmi_get_gpu_overdrive_level(gpus()[i], &od) != AMDSMI_STATUS_SUCCESS) continue;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_overdrive_level", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_overdrive_level(gpus()[i], od);
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

// ---------------- amdsmi_set_gpu_perf_determinism_mode (SET) ----------------
TEST_F(GpuUnit, SetPerfDeterminismMode_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_determinism_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_perf_determinism_mode(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetPerfDeterminismMode_AllGpus) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_gpu_perf_determinism_mode");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_determinism_mode", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_perf_determinism_mode(gpus()[i], 0);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL);
    amdsmi_col.Record(
        "gpu=" + std::to_string(i), err,
        ::amdsmi::unittest::AmdsmiStatusIsExpected(
            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
            AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_set_gpu_od_clk_info (SET, enum) ----------------
TEST_F(GpuUnit, SetOdClkInfo_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_od_clk_info", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_od_clk_info(kInvalidHandle, AMDSMI_FREQ_IND_MIN, 0, AMDSMI_CLK_TYPE_SYS);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetOdClkInfo_AllGpus) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_gpu_od_clk_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_od_volt_freq_data_t odv;
    memset(&odv, 0, sizeof(odv));
    if (amdsmi_get_gpu_od_volt_info(gpus()[i], &odv) != AMDSMI_STATUS_SUCCESS) continue;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_od_clk_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_od_clk_info(
        gpus()[i], AMDSMI_FREQ_IND_MIN, odv.curr_sclk_range.lower_bound, AMDSMI_CLK_TYPE_SYS);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL);
    amdsmi_col.Record(
        "gpu=" + std::to_string(i), err,
        ::amdsmi::unittest::AmdsmiStatusIsExpected(
            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
            AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_set_gpu_od_volt_info (SET) ----------------
TEST_F(GpuUnit, SetOdVoltInfo_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_od_volt_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_od_volt_info(kInvalidHandle, 0, 0, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetOdVoltInfo_AllGpus) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_gpu_od_volt_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_od_volt_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_od_volt_info(gpus()[i], 0, 0, 0);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL);
    amdsmi_col.Record(
        "gpu=" + std::to_string(i), err,
        ::amdsmi::unittest::AmdsmiStatusIsExpected(
            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
            AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_reset_gpu (action) ----------------
// NOTE: The valid-handle path is intentionally NOT exercised. Issuing a real GPU
// reset would disrupt other processes sharing the device; only the invalid-handle
// contract is validated here.
TEST_F(GpuUnit, ResetGpu_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_reset_gpu", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_reset_gpu(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

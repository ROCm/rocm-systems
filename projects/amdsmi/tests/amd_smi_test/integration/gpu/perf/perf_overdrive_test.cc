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

// ---------------- amdsmi_get_gpu_busy_percent ----------------
TEST_F(GpuIntegration, GetBusyPercent_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_busy_percent", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_busy_percent(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetBusyPercent_InvalidHandle) {
  uint32_t busy = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_busy_percent", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_busy_percent(kInvalidHandle, &busy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetBusyPercent_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_busy_percent");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t busy = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_busy_percent", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_busy_percent(gpus()[i], &busy);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_vcn_busy_percent ----------------
TEST_F(GpuIntegration, GetVcnBusyPercent_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_vcn_busy_percent(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetVcnBusyPercent_InvalidHandle) {
  uint32_t busy = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_vcn_busy_percent(kInvalidHandle, &busy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetVcnBusyPercent_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_vcn_busy_percent");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t busy = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_vcn_busy_percent(gpus()[i], &busy);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_utilization_count ----------------
TEST_F(GpuIntegration, GetUtilizationCount_NullOutput) {
  amdsmi_utilization_counter_t counters[1];
  memset(counters, 0, sizeof(counters));
  counters[0].type = AMDSMI_COARSE_GRAIN_GFX_ACTIVITY;
  DISPLAY_AMDSMI_API("amdsmi_get_utilization_count", "gpu=0 timestamp=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_utilization_count(any_gpu(), counters, 1, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetUtilizationCount_InvalidHandle) {
  amdsmi_utilization_counter_t counters[1];
  memset(counters, 0, sizeof(counters));
  counters[0].type = AMDSMI_COARSE_GRAIN_GFX_ACTIVITY;
  uint64_t ts = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_utilization_count", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_utilization_count(kInvalidHandle, counters, 1, &ts);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetUtilizationCount_AllGpus) {
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_utilization_count returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
         "unknown, under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_utilization_count");
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
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_perf_level ----------------
TEST_F(GpuIntegration, GetPerfLevel_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_perf_level(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetPerfLevel_InvalidHandle) {
  amdsmi_dev_perf_level_t perf;
  memset(&perf, 0, sizeof(perf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_perf_level(kInvalidHandle, &perf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetPerfLevel_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_perf_level");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dev_perf_level_t perf;
    memset(&perf, 0, sizeof(perf));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_perf_level(gpus()[i], &perf);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_overdrive_level ----------------
TEST_F(GpuIntegration, GetOverdriveLevel_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_overdrive_level(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetOverdriveLevel_InvalidHandle) {
  uint32_t od = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_overdrive_level(kInvalidHandle, &od);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetOverdriveLevel_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_overdrive_level");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t od = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_overdrive_level(gpus()[i], &od);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_mem_overdrive_level ----------------
TEST_F(GpuIntegration, GetMemOverdriveLevel_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_mem_overdrive_level", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_mem_overdrive_level(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetMemOverdriveLevel_InvalidHandle) {
  uint32_t od = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_mem_overdrive_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_mem_overdrive_level(kInvalidHandle, &od);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetMemOverdriveLevel_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_mem_overdrive_level");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t od = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_mem_overdrive_level", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_mem_overdrive_level(gpus()[i], &od);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_od_volt_info ----------------
TEST_F(GpuIntegration, GetOdVoltInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_info(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetOdVoltInfo_InvalidHandle) {
  amdsmi_od_volt_freq_data_t odv;
  memset(&odv, 0, sizeof(odv));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_info(kInvalidHandle, &odv);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetOdVoltInfo_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_od_volt_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_od_volt_freq_data_t odv;
    memset(&odv, 0, sizeof(odv));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_od_volt_info(gpus()[i], &odv);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_od_volt_curve_regions ----------------
TEST_F(GpuIntegration, GetOdVoltCurveRegions_NullNum) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_curve_regions", "gpu=0 num=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_curve_regions(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetOdVoltCurveRegions_InvalidHandle) {
  uint32_t num = 8;
  amdsmi_freq_volt_region_t buf[8];
  memset(buf, 0, sizeof(buf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_curve_regions", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_curve_regions(kInvalidHandle, &num, buf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetOdVoltCurveRegions_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_od_volt_curve_regions");
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
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_activity ----------------
TEST_F(GpuIntegration, GetActivity_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_activity", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_activity(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetActivity_InvalidHandle) {
  amdsmi_engine_usage_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_activity", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_activity(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetActivity_AllGpus) {
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_gpu_activity returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
         "unknown, under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_activity");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_engine_usage_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_activity", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_activity(gpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_set_gpu_perf_level (SET, enum) ----------------
TEST_F(GpuIntegration, SetPerfLevel_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_perf_level(kInvalidHandle, AMDSMI_DEV_PERF_LEVEL_AUTO);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_overdrive_level (SET) ----------------
TEST_F(GpuIntegration, SetOverdriveLevel_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_overdrive_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_overdrive_level(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_perf_determinism_mode (SET) ----------------
TEST_F(GpuIntegration, SetPerfDeterminismMode_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_determinism_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_perf_determinism_mode(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_od_clk_info (SET, enum) ----------------
TEST_F(GpuIntegration, SetOdClkInfo_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_od_clk_info", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_od_clk_info(kInvalidHandle, AMDSMI_FREQ_IND_MIN, 0, AMDSMI_CLK_TYPE_SYS);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_od_volt_info (SET) ----------------
TEST_F(GpuIntegration, SetOdVoltInfo_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_od_volt_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_od_volt_info(kInvalidHandle, 0, 0, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_reset_gpu (action) ----------------
// NOTE: The valid-handle path is intentionally NOT exercised. Issuing a real GPU
// reset would disrupt other processes sharing the device; only the invalid-handle
// contract is validated here.
TEST_F(GpuIntegration, ResetGpu_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_reset_gpu", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_reset_gpu(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

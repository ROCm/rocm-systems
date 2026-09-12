// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// ---------------- amdsmi_get_gpu_busy_percent ----------------
TEST(GpuIntegration, GetBusyPercent_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_busy_percent", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_busy_percent(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetBusyPercent_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t busy = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_busy_percent", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_busy_percent(kInvalidHandle, &busy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetBusyPercent_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetVcnBusyPercent_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_vcn_busy_percent(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetVcnBusyPercent_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t busy = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_vcn_busy_percent", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_vcn_busy_percent(kInvalidHandle, &busy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetVcnBusyPercent_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetUtilizationCount_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_utilization_counter_t counters[1];
  memset(counters, 0, sizeof(counters));
  counters[0].type = AMDSMI_COARSE_GRAIN_GFX_ACTIVITY;
  DISPLAY_AMDSMI_API("amdsmi_get_utilization_count", "gpu=0 timestamp=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_utilization_count(any_gpu(), counters, 1, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetUtilizationCount_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetUtilizationCount_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetPerfLevel_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_perf_level(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetPerfLevel_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_dev_perf_level_t perf;
  memset(&perf, 0, sizeof(perf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_perf_level(kInvalidHandle, &perf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetPerfLevel_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetOverdriveLevel_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_overdrive_level(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetOverdriveLevel_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t od = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_overdrive_level(kInvalidHandle, &od);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetOverdriveLevel_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetMemOverdriveLevel_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_mem_overdrive_level", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_mem_overdrive_level(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetMemOverdriveLevel_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t od = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_mem_overdrive_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_mem_overdrive_level(kInvalidHandle, &od);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetMemOverdriveLevel_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetOdVoltInfo_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_info(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetOdVoltInfo_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_od_volt_freq_data_t odv;
  memset(&odv, 0, sizeof(odv));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_info(kInvalidHandle, &odv);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetOdVoltInfo_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetOdVoltCurveRegions_NullNum) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_curve_regions", "gpu=0 num=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_curve_regions(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetOdVoltCurveRegions_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t num = 8;
  amdsmi_freq_volt_region_t buf[8];
  memset(buf, 0, sizeof(buf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_curve_regions", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_od_volt_curve_regions(kInvalidHandle, &num, buf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetOdVoltCurveRegions_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetActivity_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_activity", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_activity(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetActivity_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_engine_usage_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_activity", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_activity(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetActivity_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, SetPerfLevel_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_perf_level(kInvalidHandle, AMDSMI_DEV_PERF_LEVEL_AUTO);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_overdrive_level (SET) ----------------
TEST(GpuIntegration, SetOverdriveLevel_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_overdrive_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_overdrive_level(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_perf_determinism_mode (SET) ----------------
TEST(GpuIntegration, SetPerfDeterminismMode_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_determinism_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_perf_determinism_mode(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_od_clk_info (SET, enum) ----------------
TEST(GpuIntegration, SetOdClkInfo_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_od_clk_info", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_od_clk_info(kInvalidHandle, AMDSMI_FREQ_IND_MIN, 0, AMDSMI_CLK_TYPE_SYS);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_od_volt_info (SET) ----------------
TEST(GpuIntegration, SetOdVoltInfo_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, ResetGpu_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_reset_gpu", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_reset_gpu(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

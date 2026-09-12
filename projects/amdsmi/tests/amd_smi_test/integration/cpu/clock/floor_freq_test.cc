// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// CPU floor-frequency, effective-floor, MSR-floor and SDPS limit APIs. Core
// variants derive a core index from the handle (cpu_cores()); socket variants
// use cpus(). Write APIs guard only the handle, so they omit the null-output
// test and restore benign values in the valid path.

// ---- amdsmi_get_cpu_core_floor_freq_limit (output guarded, core handle) ----
TEST(CpuIntegration, GetCoreFloorFreqLimit_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_floor_freq_limit", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_core_floor_freq_limit(any_cpu_core(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(CpuIntegration, GetCoreFloorFreqLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t floor = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_core_floor_freq_limit(kInvalidHandle, &floor);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(CpuIntegration, GetCoreFloorFreqLimit_AllCores) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_core_floor_freq_limit");
  if (cpu_cores().empty()) GTEST_SKIP() << "No CPU cores";
  for (size_t i = 0; i < cpu_cores().size(); ++i) {
    uint32_t floor = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_floor_freq_limit", "core=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_core_floor_freq_limit(cpu_cores()[i], &floor);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("core=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_floor_freq_limit (output guarded, socket) ----
TEST(CpuIntegration, GetFloorFreqLimit_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_floor_freq_limit", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_floor_freq_limit(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(CpuIntegration, GetFloorFreqLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t floor = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_floor_freq_limit(kInvalidHandle, &floor);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(CpuIntegration, GetFloorFreqLimit_AllCpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_floor_freq_limit");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t floor = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_floor_freq_limit", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_floor_freq_limit(cpus()[i], &floor);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_core_eff_floor_freq_limit (output guarded, core handle) ----
TEST(CpuIntegration, GetCoreEffFloorFreqLimit_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_eff_floor_freq_limit", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_core_eff_floor_freq_limit(any_cpu_core(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(CpuIntegration, GetCoreEffFloorFreqLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t eff = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_eff_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_core_eff_floor_freq_limit(kInvalidHandle, &eff);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(CpuIntegration, GetCoreEffFloorFreqLimit_AllCores) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_core_eff_floor_freq_limit");
  if (cpu_cores().empty()) GTEST_SKIP() << "No CPU cores";
  for (size_t i = 0; i < cpu_cores().size(); ++i) {
    uint32_t eff = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_eff_floor_freq_limit", "core=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_core_eff_floor_freq_limit(cpu_cores()[i], &eff);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("core=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_eff_floor_freq_limit (output guarded, socket) ----
TEST(CpuIntegration, GetEffFloorFreqLimit_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_eff_floor_freq_limit", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_eff_floor_freq_limit(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(CpuIntegration, GetEffFloorFreqLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t eff = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_eff_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_eff_floor_freq_limit(kInvalidHandle, &eff);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(CpuIntegration, GetEffFloorFreqLimit_AllCpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_eff_floor_freq_limit");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t eff = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_eff_floor_freq_limit", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_eff_floor_freq_limit(cpus()[i], &eff);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_sdps_limit (output guarded, socket) ----
TEST(CpuIntegration, GetSdpsLimit_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_sdps_limit", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_sdps_limit(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(CpuIntegration, GetSdpsLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint32_t sdps = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_sdps_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_sdps_limit(kInvalidHandle, &sdps);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(CpuIntegration, GetSdpsLimit_AllCpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_sdps_limit");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t sdps = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_sdps_limit", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_sdps_limit(cpus()[i], &sdps);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- write APIs (invalid input only; valid-input cases are in functional/) ----
TEST(CpuIntegration, SetCoreFloorFreqLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_core_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_core_floor_freq_limit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(CpuIntegration, SetFloorFreqLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_floor_freq_limit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(CpuIntegration, SetMsrFloorFreqLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_msr_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_msr_floor_freq_limit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(CpuIntegration, SetCoreMsrFloorFreqLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_core_msr_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_core_msr_floor_freq_limit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(CpuIntegration, SetSdpsLimit_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_sdps_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_sdps_limit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

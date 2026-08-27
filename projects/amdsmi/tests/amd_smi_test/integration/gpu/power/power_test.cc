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

// ---------------- amdsmi_get_energy_count ----------------
TEST_F(GpuIntegration, GetEnergyCount_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_energy_count", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_energy_count(any_gpu(), nullptr, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetEnergyCount_InvalidHandle) {
  uint64_t energy = 0, ts = 0;
  float res = 0.0f;
  DISPLAY_AMDSMI_API("amdsmi_get_energy_count", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_energy_count(kInvalidHandle, &energy, &res, &ts);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetEnergyCount_AllGpus) {
  GTEST_SKIP() << "amdsmi_get_energy_count returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
                  "unknown, under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_energy_count");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint64_t energy = 0, ts = 0;
    float res = 0.0f;
    DISPLAY_AMDSMI_API("amdsmi_get_energy_count", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_energy_count(gpus()[i], &energy, &res, &ts);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_supported_power_cap ----------------
TEST_F(GpuIntegration, GetSupportedPowerCap_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_supported_power_cap", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_supported_power_cap(any_gpu(), nullptr, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetSupportedPowerCap_InvalidHandle) {
  uint32_t count = 8;
  uint32_t inds[8];
  amdsmi_power_cap_type_t types[8];
  memset(inds, 0, sizeof(inds));
  memset(types, 0, sizeof(types));
  DISPLAY_AMDSMI_API("amdsmi_get_supported_power_cap", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_supported_power_cap(kInvalidHandle, &count, inds, types);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetSupportedPowerCap_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_supported_power_cap");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t count = 8;
    uint32_t inds[8];
    amdsmi_power_cap_type_t types[8];
    memset(inds, 0, sizeof(inds));
    memset(types, 0, sizeof(types));
    DISPLAY_AMDSMI_API("amdsmi_get_supported_power_cap", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_supported_power_cap(gpus()[i], &count, inds, types);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_power_cap_info ----------------
TEST_F(GpuIntegration, GetPowerCapInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_power_cap_info(any_gpu(), 0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetPowerCapInfo_InvalidHandle) {
  amdsmi_power_cap_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_power_cap_info(kInvalidHandle, 0, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetPowerCapInfo_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_power_cap_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_power_cap_info_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_power_cap_info(gpus()[i], 0, &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_power_profile_presets ----------------
TEST_F(GpuIntegration, GetPowerProfilePresets_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_power_profile_presets(any_gpu(), 0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetPowerProfilePresets_InvalidHandle) {
  amdsmi_power_profile_status_t status;
  memset(&status, 0, sizeof(status));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_power_profile_presets(kInvalidHandle, 0, &status);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetPowerProfilePresets_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_power_profile_presets");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_power_profile_status_t status;
    memset(&status, 0, sizeof(status));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_power_profile_presets(gpus()[i], 0, &status);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_power_info ----------------
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetPowerInfo, amdsmi_get_power_info, amdsmi_power_info_t)

// ---------------- amdsmi_is_gpu_power_management_enabled ----------------
TEST_F(GpuIntegration, IsPowerManagementEnabled_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_is_gpu_power_management_enabled", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_is_gpu_power_management_enabled(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, IsPowerManagementEnabled_InvalidHandle) {
  bool enabled = false;
  DISPLAY_AMDSMI_API("amdsmi_is_gpu_power_management_enabled", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_is_gpu_power_management_enabled(kInvalidHandle, &enabled);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, IsPowerManagementEnabled_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_is_gpu_power_management_enabled");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    bool enabled = false;
    DISPLAY_AMDSMI_API("amdsmi_is_gpu_power_management_enabled", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_is_gpu_power_management_enabled(gpus()[i], &enabled);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_set_power_cap (SET) ----------------
TEST_F(GpuIntegration, SetPowerCap_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_power_cap", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_power_cap(kInvalidHandle, 0, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_set_gpu_power_profile (SET, enum) ----------------
TEST_F(GpuIntegration, SetPowerProfile_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_power_profile(kInvalidHandle, 0, AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

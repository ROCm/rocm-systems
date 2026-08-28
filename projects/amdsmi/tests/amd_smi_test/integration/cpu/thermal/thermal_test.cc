// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// Socket-level CPU temperature and thermal-delta APIs. Socket temperature only
// guards the handle so it omits the null-output test; tdelta and the SVI3 VR
// controller temperature guard their outputs and get a null-output test.

// ---- amdsmi_get_cpu_socket_temperature (handle guarded only) ----
TEST_F(CpuIntegration, GetSocketTemperature_InvalidHandle) {
  uint32_t tmon = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_temperature", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_temperature(kInvalidHandle, &tmon);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSocketTemperature_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_socket_temperature");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t tmon = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_temperature", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_socket_temperature(cpus()[i], &tmon);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_tdelta (output guarded) ----
TEST_F(CpuIntegration, GetTdelta_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_tdelta", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_tdelta(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetTdelta_InvalidHandle) {
  uint8_t tdelta = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_tdelta", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_tdelta(kInvalidHandle, &tdelta);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetTdelta_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_tdelta");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint8_t tdelta = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_tdelta", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_tdelta(cpus()[i], &tdelta);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_svi3_vr_controller_temp (outputs guarded) ----
TEST_F(CpuIntegration, GetSvi3VrControllerTemp_NullOutput) {
  uint32_t rail_index = 0, temp = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_svi3_vr_controller_temp", "rail_selection=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_cpu_svi3_vr_controller_temp(any_cpu(), nullptr, &rail_index, &temp);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetSvi3VrControllerTemp_InvalidHandle) {
  uint32_t rail_selection = 0, rail_index = 0, temp = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_svi3_vr_controller_temp", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_cpu_svi3_vr_controller_temp(kInvalidHandle, &rail_selection, &rail_index, &temp);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSvi3VrControllerTemp_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_svi3_vr_controller_temp");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t rail_selection = 0, rail_index = 0, temp = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_svi3_vr_controller_temp", "cpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err =
        amdsmi_get_cpu_svi3_vr_controller_temp(cpus()[i], &rail_selection, &rail_index, &temp);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

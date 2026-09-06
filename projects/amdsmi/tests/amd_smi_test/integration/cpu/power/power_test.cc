// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// CPU power, power-cap, power-efficiency and energy APIs. Socket-scoped calls
// use cpus(); the core-scoped power/energy calls derive a core index from the
// handle and therefore iterate cpu_cores(). Getters that only guard the handle
// dereference the output on success, so those omit the null-output test.

// ---- amdsmi_get_cpu_socket_power (output guarded) ----
TEST_F(CpuIntegration, GetSocketPower_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_power", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_power(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetSocketPower_InvalidHandle) {
  uint32_t power = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_power", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_power(kInvalidHandle, &power);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSocketPower_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_socket_power");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t power = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_power", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_socket_power(cpus()[i], &power);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_socket_power_cap (output guarded) ----
TEST_F(CpuIntegration, GetSocketPowerCap_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_power_cap", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_power_cap(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetSocketPowerCap_InvalidHandle) {
  uint32_t cap = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_power_cap", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_power_cap(kInvalidHandle, &cap);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSocketPowerCap_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_socket_power_cap");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t cap = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_power_cap", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_socket_power_cap(cpus()[i], &cap);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_socket_power_cap_max (output guarded) ----
TEST_F(CpuIntegration, GetSocketPowerCapMax_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_power_cap_max", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_power_cap_max(any_cpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetSocketPowerCapMax_InvalidHandle) {
  uint32_t cap_max = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_power_cap_max", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_power_cap_max(kInvalidHandle, &cap_max);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSocketPowerCapMax_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_socket_power_cap_max");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t cap_max = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_power_cap_max", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_socket_power_cap_max(cpus()[i], &cap_max);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_pwr_svi_telemetry_all_rails (handle guarded only) ----
TEST_F(CpuIntegration, GetPwrSviTelemetryAllRails_InvalidHandle) {
  uint32_t power = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_pwr_svi_telemetry_all_rails", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_pwr_svi_telemetry_all_rails(kInvalidHandle, &power);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetPwrSviTelemetryAllRails_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_pwr_svi_telemetry_all_rails");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t power = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_pwr_svi_telemetry_all_rails", "cpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_pwr_svi_telemetry_all_rails(cpus()[i], &power);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_pwr_efficiency_mode (handle guarded only) ----
TEST_F(CpuIntegration, GetPwrEfficiencyMode_InvalidHandle) {
  uint32_t mode = 0, util = 0, ppt = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_pwr_efficiency_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_pwr_efficiency_mode(kInvalidHandle, &mode, &util, &ppt);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetPwrEfficiencyMode_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_pwr_efficiency_mode");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t mode = 0, util = 0, ppt = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_pwr_efficiency_mode", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_pwr_efficiency_mode(cpus()[i], &mode, &util, &ppt);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_socket_energy (handle guarded only) ----
TEST_F(CpuIntegration, GetSocketEnergy_InvalidHandle) {
  uint64_t energy = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_energy", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_energy(kInvalidHandle, &energy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSocketEnergy_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_socket_energy");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint64_t energy = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_energy", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_socket_energy(cpus()[i], &energy);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_socket_c0_residency (handle guarded only) ----
TEST_F(CpuIntegration, GetSocketC0Residency_InvalidHandle) {
  uint32_t residency = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_c0_residency", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_c0_residency(kInvalidHandle, &residency);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSocketC0Residency_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_socket_c0_residency");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t residency = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_c0_residency", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_socket_c0_residency(cpus()[i], &residency);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_core_ccd_power (output guarded, core handle) ----
TEST_F(CpuIntegration, GetCoreCcdPower_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_ccd_power", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_core_ccd_power(any_cpu_core(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetCoreCcdPower_InvalidHandle) {
  uint32_t power = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_ccd_power", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_core_ccd_power(kInvalidHandle, &power);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetCoreCcdPower_AllCores) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_core_ccd_power");
  if (cpu_cores().empty()) GTEST_SKIP() << "No CPU cores";
  for (size_t i = 0; i < cpu_cores().size(); ++i) {
    uint32_t power = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_ccd_power", "core=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_core_ccd_power(cpu_cores()[i], &power);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("core=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_core_energy (handle guarded only, core handle) ----
TEST_F(CpuIntegration, GetCoreEnergy_InvalidHandle) {
  uint64_t energy = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_energy", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_core_energy(kInvalidHandle, &energy);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetCoreEnergy_AllCores) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_core_energy");
  if (cpu_cores().empty()) GTEST_SKIP() << "No CPU cores";
  for (size_t i = 0; i < cpu_cores().size(); ++i) {
    uint64_t energy = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_energy", "core=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_core_energy(cpu_cores()[i], &energy);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("core=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_set_cpu_socket_power_cap (invalid input only; valid-input cases are in functional/)
// ----
TEST_F(CpuIntegration, SetSocketPowerCap_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_socket_power_cap", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_socket_power_cap(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---- amdsmi_set_cpu_pwr_efficiency_mode (invalid input only; valid-input cases are in
// functional/) ----
TEST_F(CpuIntegration, SetPwrEfficiencyMode_NullOutput) {
  uint32_t ppt = 0;
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_pwr_efficiency_mode", "utilization=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_pwr_efficiency_mode(any_cpu(), 0, nullptr, &ppt);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, SetPwrEfficiencyMode_InvalidHandle) {
  uint32_t util = 0, ppt = 0;
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_pwr_efficiency_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_pwr_efficiency_mode(kInvalidHandle, 0, &util, &ppt);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

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

// Socket-level CPU identity, version and capability APIs. The version/identity
// getters below only guard the processor handle in the current implementation
// (the output pointer is dereferenced on success), so their negative coverage
// is the invalid-handle test; a null-output test is omitted to avoid a crash on
// CPU hardware.

// ---- amdsmi_get_cpu_hsmp_driver_version ----
TEST_F(CpuIntegration, GetHsmpDriverVersion_InvalidHandle) {
  amdsmi_hsmp_driver_version_t ver;
  memset(&ver, 0, sizeof(ver));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_hsmp_driver_version", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_hsmp_driver_version(kInvalidHandle, &ver);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetHsmpDriverVersion_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_hsmp_driver_version");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    amdsmi_hsmp_driver_version_t ver;
    memset(&ver, 0, sizeof(ver));
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_hsmp_driver_version", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_hsmp_driver_version(cpus()[i], &ver);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_smu_fw_version ----
TEST_F(CpuIntegration, GetSmuFwVersion_InvalidHandle) {
  amdsmi_smu_fw_version_t fw;
  memset(&fw, 0, sizeof(fw));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_smu_fw_version", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_smu_fw_version(kInvalidHandle, &fw);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetSmuFwVersion_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_smu_fw_version");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    amdsmi_smu_fw_version_t fw;
    memset(&fw, 0, sizeof(fw));
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_smu_fw_version", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_smu_fw_version(cpus()[i], &fw);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_hsmp_proto_ver ----
TEST_F(CpuIntegration, GetHsmpProtoVer_InvalidHandle) {
  uint32_t proto = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_hsmp_proto_ver", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_hsmp_proto_ver(kInvalidHandle, &proto);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetHsmpProtoVer_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_hsmp_proto_ver");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t proto = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_hsmp_proto_ver", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_hsmp_proto_ver(cpus()[i], &proto);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_model_name ----
TEST_F(CpuIntegration, GetCpuModelName_InvalidHandle) {
  amdsmi_cpu_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_model_name", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_model_name(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetCpuModelName_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_model_name");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    amdsmi_cpu_info_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_model_name", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_model_name(cpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_prochot_status ----
TEST_F(CpuIntegration, GetProchotStatus_InvalidHandle) {
  uint32_t prochot = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_prochot_status", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_prochot_status(kInvalidHandle, &prochot);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetProchotStatus_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_prochot_status");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t prochot = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_prochot_status", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_prochot_status(cpus()[i], &prochot);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_first_online_core_on_cpu_socket ----
TEST_F(CpuIntegration, FirstOnlineCore_InvalidHandle) {
  uint32_t core_ind = 0;
  DISPLAY_AMDSMI_API("amdsmi_first_online_core_on_cpu_socket", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_first_online_core_on_cpu_socket(kInvalidHandle, &core_ind);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, FirstOnlineCore_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_first_online_core_on_cpu_socket");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t core_ind = 0;
    DISPLAY_AMDSMI_API("amdsmi_first_online_core_on_cpu_socket", "cpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_first_online_core_on_cpu_socket(cpus()[i], &core_ind);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_enabled_commands (outputs guarded) ----
TEST_F(CpuIntegration, GetEnabledCommands_NullOutput) {
  uint32_t m0 = 0, m1 = 0, m2 = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_enabled_commands", "r_mask=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_enabled_commands(any_cpu(), nullptr, &m0, &m1, &m2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetEnabledCommands_InvalidHandle) {
  bool r_mask = false;
  uint32_t m0 = 0, m1 = 0, m2 = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_enabled_commands", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_enabled_commands(kInvalidHandle, &r_mask, &m0, &m1, &m2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetEnabledCommands_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_enabled_commands");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    bool r_mask = false;
    uint32_t m0 = 0, m1 = 0, m2 = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_enabled_commands", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_enabled_commands(cpus()[i], &r_mask, &m0, &m1, &m2);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_cpu_affinity_with_scope (cpu_set guarded; loops scope) ----
static constexpr amdsmi_affinity_scope_t kAffinityScopes[] = {AMDSMI_AFFINITY_SCOPE_NODE,
                                                              AMDSMI_AFFINITY_SCOPE_SOCKET};

TEST_F(CpuIntegration, GetAffinityWithScope_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_affinity_with_scope", "cpu_set=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_cpu_affinity_with_scope(any_cpu(), 8, nullptr, AMDSMI_AFFINITY_SCOPE_NODE);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetAffinityWithScope_InvalidHandle) {
  uint64_t cpu_set[8];
  memset(cpu_set, 0, sizeof(cpu_set));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_affinity_with_scope", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_cpu_affinity_with_scope(kInvalidHandle, 8, cpu_set, AMDSMI_AFFINITY_SCOPE_NODE);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetAffinityWithScope_AllCpusAllScopes) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_affinity_with_scope");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i)
    for (auto scope : kAffinityScopes) {
      uint64_t cpu_set[8];
      memset(cpu_set, 0, sizeof(cpu_set));
      DISPLAY_AMDSMI_API("amdsmi_get_cpu_affinity_with_scope",
                         "cpu=" + std::to_string(i) + " scope=" + std::to_string(scope), kVerbose);
      amdsmi_status_t err = amdsmi_get_cpu_affinity_with_scope(cpus()[i], 8, cpu_set, scope);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("cpu=" + std::to_string(i) + " scope=" + std::to_string(scope),
                                err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

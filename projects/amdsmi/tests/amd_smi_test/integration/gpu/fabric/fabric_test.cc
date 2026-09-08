// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// ---------------- amdsmi_get_gpu_fabric_info ----------------
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetFabricInfo, amdsmi_get_gpu_fabric_info,
                                     amdsmi_fabric_info_t)

// ---------------- amdsmi_get_fabric_telemetry_data ----------------
TEST(GpuIntegration, GetFabricTelemetryData_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_telemetry_data", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_fabric_telemetry_data(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetFabricTelemetryData_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_fabric_telemetry_t telemetry;
  memset(&telemetry, 0, sizeof(telemetry));
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_telemetry_data", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_fabric_telemetry_data(kInvalidHandle, &telemetry);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetFabricTelemetryData_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_fabric_telemetry_data");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_fabric_telemetry_t telemetry;
    memset(&telemetry, 0, sizeof(telemetry));
    DISPLAY_AMDSMI_API("amdsmi_get_fabric_telemetry_data", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_fabric_telemetry_data(gpus()[i], &telemetry);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_alloc_fabric_telemetry / free ----------------
TEST(GpuIntegration, AllocFabricTelemetry_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_alloc_fabric_telemetry", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_alloc_fabric_telemetry(any_gpu(), 0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, AllocFabricTelemetry_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_fabric_telemetry_t* telemetry = nullptr;
  DISPLAY_AMDSMI_API("amdsmi_alloc_fabric_telemetry", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_alloc_fabric_telemetry(kInvalidHandle, 0, &telemetry);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, AllocFreeFabricTelemetry_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_alloc_fabric_telemetry");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_fabric_telemetry_t* telemetry = nullptr;
    DISPLAY_AMDSMI_API("amdsmi_alloc_fabric_telemetry", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_alloc_fabric_telemetry(gpus()[i], 0, &telemetry);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
    if (err == AMDSMI_STATUS_SUCCESS && telemetry != nullptr) {
      DISPLAY_AMDSMI_API("amdsmi_free_fabric_telemetry", "gpu=" + std::to_string(i), kVerbose);
      amdsmi_status_t ferr = amdsmi_free_fabric_telemetry(gpus()[i], telemetry);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, ferr, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("gpu=" + std::to_string(i), ferr,
                        ::amdsmi::test::AmdsmiStatusIsExpected(ferr, AMDSMI_STATUS_SUCCESS,
                                                               AMDSMI_STATUS_NOT_SUPPORTED,
                                                               AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_free_fabric_telemetry (invalid) ----------------
TEST(GpuIntegration, FreeFabricTelemetry_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_free_fabric_telemetry", "telemetry=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_free_fabric_telemetry(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---------------- amdsmi_fabric_telem_id_to_string (no handle) ----------------
TEST(GpuIntegration, FabricTelemIdToString_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_fabric_telem_id_to_string", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_fabric_telem_id_to_string(0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST(GpuIntegration, FabricTelemIdToString_Valid) {
  AMDSMI_API_TEST_SCOPE();
  const char* name = nullptr;
  DISPLAY_AMDSMI_API("amdsmi_fabric_telem_id_to_string", "telem_id=0", kVerbose);
  amdsmi_status_t err = amdsmi_fabric_telem_id_to_string(0, &name);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                        AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_FOUND);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_INVAL,
                       AMDSMI_STATUS_NOT_FOUND);
}

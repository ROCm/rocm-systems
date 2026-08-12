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

// ---------------- amdsmi_get_gpu_fabric_info ----------------
TEST_F(GpuUnit, GetFabricInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fabric_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_fabric_info(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetFabricInfo_InvalidHandle) {
  amdsmi_fabric_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fabric_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_fabric_info(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetFabricInfo_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_fabric_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_fabric_info_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fabric_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_fabric_info(gpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_fabric_telemetry_data ----------------
TEST_F(GpuUnit, GetFabricTelemetryData_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_telemetry_data", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_fabric_telemetry_data(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetFabricTelemetryData_InvalidHandle) {
  amdsmi_fabric_telemetry_t telemetry;
  memset(&telemetry, 0, sizeof(telemetry));
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_telemetry_data", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_fabric_telemetry_data(kInvalidHandle, &telemetry);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetFabricTelemetryData_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_fabric_telemetry_data");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_fabric_telemetry_t telemetry;
    memset(&telemetry, 0, sizeof(telemetry));
    DISPLAY_AMDSMI_API("amdsmi_get_fabric_telemetry_data", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_fabric_telemetry_data(gpus()[i], &telemetry);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_alloc_fabric_telemetry / free ----------------
TEST_F(GpuUnit, AllocFabricTelemetry_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_alloc_fabric_telemetry", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_alloc_fabric_telemetry(gpus()[0], 0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, AllocFabricTelemetry_InvalidHandle) {
  amdsmi_fabric_telemetry_t* telemetry = nullptr;
  DISPLAY_AMDSMI_API("amdsmi_alloc_fabric_telemetry", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_alloc_fabric_telemetry(kInvalidHandle, 0, &telemetry);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, AllocFreeFabricTelemetry_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_alloc_fabric_telemetry");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_fabric_telemetry_t* telemetry = nullptr;
    DISPLAY_AMDSMI_API("amdsmi_alloc_fabric_telemetry", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_alloc_fabric_telemetry(gpus()[i], 0, &telemetry);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    if (err == AMDSMI_STATUS_SUCCESS && telemetry != nullptr) {
      DISPLAY_AMDSMI_API("amdsmi_free_fabric_telemetry", "gpu=" + std::to_string(i), kVerbose);
      amdsmi_status_t ferr = amdsmi_free_fabric_telemetry(gpus()[i], telemetry);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, ferr, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("gpu=" + std::to_string(i), ferr,
                        ::amdsmi::unittest::AmdsmiStatusIsExpected(
                            ferr, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_free_fabric_telemetry (invalid) ----------------
TEST_F(GpuUnit, FreeFabricTelemetry_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_free_fabric_telemetry", "telemetry=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_free_fabric_telemetry(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---------------- amdsmi_fabric_telem_id_to_string (no handle) ----------------
TEST_F(GpuUnit, FabricTelemIdToString_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_fabric_telem_id_to_string", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_fabric_telem_id_to_string(0, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, FabricTelemIdToString_Valid) {
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

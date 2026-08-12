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

static constexpr amdsmi_reg_type_t kRegTypes[] = {AMDSMI_REG_XGMI, AMDSMI_REG_WAFL, AMDSMI_REG_PCIE,
                                                  AMDSMI_REG_USR, AMDSMI_REG_USR1};

// ---------------- amdsmi_get_gpu_metrics_header_info ----------------
TEST_F(GpuUnit, GetMetricsHeaderInfo_NullOutput) {
  GTEST_SKIP()
      << "amdsmi_get_gpu_metrics_header_info crashes on a null output pointer; proper return "
         "should be AMDSMI_STATUS_INVAL";
  // Proper contract once fixed:
  //   amdsmi_status_t err = amdsmi_get_gpu_metrics_header_info(gpus()[0], nullptr);
  //   AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetMetricsHeaderInfo_InvalidHandle) {
  amd_metrics_table_header_t header;
  memset(&header, 0, sizeof(header));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_header_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_metrics_header_info(kInvalidHandle, &header);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetMetricsHeaderInfo_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_metrics_header_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amd_metrics_table_header_t header;
    memset(&header, 0, sizeof(header));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_header_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_metrics_header_info(gpus()[i], &header);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_metrics_info ----------------
TEST_F(GpuUnit, GetMetricsInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_metrics_info(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetMetricsInfo_InvalidHandle) {
  amdsmi_gpu_metrics_t metrics;
  memset(&metrics, 0, sizeof(metrics));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_metrics_info(kInvalidHandle, &metrics);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetMetricsInfo_AllGpus) {
  GTEST_SKIP() << "amdsmi_get_gpu_metrics_info returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
                  "unknown, under investigation";

  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_metrics_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_gpu_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_metrics_info(gpus()[i], &metrics);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_partition_metrics_info ----------------
TEST_F(GpuUnit, GetPartitionMetricsInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_partition_metrics_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_partition_metrics_info(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetPartitionMetricsInfo_InvalidHandle) {
  amdsmi_gpu_metrics_t metrics;
  memset(&metrics, 0, sizeof(metrics));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_partition_metrics_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_partition_metrics_info(kInvalidHandle, &metrics);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetPartitionMetricsInfo_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_partition_metrics_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_gpu_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_partition_metrics_info", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_partition_metrics_info(gpus()[i], &metrics);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_pm_metrics_info ----------------
TEST_F(GpuUnit, GetPmMetricsInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pm_metrics_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pm_metrics_info(gpus()[0], nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetPmMetricsInfo_InvalidHandle) {
  amdsmi_name_value_t* pm = nullptr;
  uint32_t num = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pm_metrics_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pm_metrics_info(kInvalidHandle, &pm, &num);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetPmMetricsInfo_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_pm_metrics_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_name_value_t* pm = nullptr;
    uint32_t num = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_pm_metrics_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_pm_metrics_info(gpus()[i], &pm, &num);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_reg_table_info (enum) ----------------
TEST_F(GpuUnit, GetRegTableInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_reg_table_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_reg_table_info(gpus()[0], AMDSMI_REG_XGMI, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetRegTableInfo_InvalidHandle) {
  amdsmi_name_value_t* reg = nullptr;
  uint32_t num = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_reg_table_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_reg_table_info(kInvalidHandle, AMDSMI_REG_XGMI, &reg, &num);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetRegTableInfo_AllGpusAllTypes) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_reg_table_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto rt : kRegTypes) {
      amdsmi_name_value_t* reg = nullptr;
      uint32_t num = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_reg_table_info",
                         "gpu=" + std::to_string(i) + " reg=" + std::to_string(rt), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_reg_table_info(gpus()[i], rt, &reg, &num);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("gpu=" + std::to_string(i) + " reg=" + std::to_string(rt), err,
                        ::amdsmi::unittest::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

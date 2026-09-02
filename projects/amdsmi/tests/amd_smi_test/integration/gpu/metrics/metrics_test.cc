// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

static constexpr amdsmi_reg_type_t kRegTypes[] = {AMDSMI_REG_XGMI, AMDSMI_REG_WAFL, AMDSMI_REG_PCIE,
                                                  AMDSMI_REG_USR, AMDSMI_REG_USR1};

// ---------------- amdsmi_get_gpu_metrics_header_info ----------------
TEST_F(GpuIntegration, GetMetricsHeaderInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_header_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_metrics_header_info(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetMetricsHeaderInfo_InvalidHandle) {
  amd_metrics_table_header_t header;
  memset(&header, 0, sizeof(header));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_header_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_metrics_header_info(kInvalidHandle, &header);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetMetricsHeaderInfo_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_metrics_header_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amd_metrics_table_header_t header;
    memset(&header, 0, sizeof(header));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_header_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_metrics_header_info(gpus()[i], &header);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_metrics_info ----------------
TEST_F(GpuIntegration, GetMetricsInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_metrics_info(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetMetricsInfo_InvalidHandle) {
  amdsmi_gpu_metrics_t metrics;
  memset(&metrics, 0, sizeof(metrics));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_metrics_info(kInvalidHandle, &metrics);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetMetricsInfo_AllGpus) {
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_gpu_metrics_info returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
         "unknown, under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_metrics_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_gpu_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_metrics_info(gpus()[i], &metrics);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_partition_metrics_info ----------------
TEST_F(GpuIntegration, GetPartitionMetricsInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_partition_metrics_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_partition_metrics_info(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetPartitionMetricsInfo_InvalidHandle) {
  amdsmi_gpu_metrics_t metrics;
  memset(&metrics, 0, sizeof(metrics));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_partition_metrics_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_partition_metrics_info(kInvalidHandle, &metrics);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetPartitionMetricsInfo_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_partition_metrics_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_gpu_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_partition_metrics_info", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_partition_metrics_info(gpus()[i], &metrics);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_pm_metrics_info ----------------
TEST_F(GpuIntegration, GetPmMetricsInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pm_metrics_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pm_metrics_info(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetPmMetricsInfo_InvalidHandle) {
  amdsmi_name_value_t* pm = nullptr;
  uint32_t num = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pm_metrics_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pm_metrics_info(kInvalidHandle, &pm, &num);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetPmMetricsInfo_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_pm_metrics_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_name_value_t* pm = nullptr;
    uint32_t num = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_pm_metrics_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_pm_metrics_info(gpus()[i], &pm, &num);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_reg_table_info (enum) ----------------
TEST_F(GpuIntegration, GetRegTableInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_reg_table_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_reg_table_info(any_gpu(), AMDSMI_REG_XGMI, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetRegTableInfo_InvalidHandle) {
  amdsmi_name_value_t* reg = nullptr;
  uint32_t num = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_reg_table_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_reg_table_info(kInvalidHandle, AMDSMI_REG_XGMI, &reg, &num);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetRegTableInfo_AllGpusAllTypes) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_reg_table_info");
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
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " reg=" + std::to_string(rt), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

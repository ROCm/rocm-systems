// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// ---------------- amdsmi_get_gpu_pci_bandwidth ----------------
TEST(GpuIntegration, GetPciBandwidth_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_bandwidth", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pci_bandwidth(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetPciBandwidth_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_pcie_bandwidth_t bw;
  memset(&bw, 0, sizeof(bw));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_bandwidth", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pci_bandwidth(kInvalidHandle, &bw);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetPciBandwidth_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_pcie_bandwidth returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
         "unknown, under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_pci_bandwidth");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_pcie_bandwidth_t bw;
    memset(&bw, 0, sizeof(bw));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_bandwidth", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_pci_bandwidth(gpus()[i], &bw);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_bdf_id ----------------
TEST(GpuIntegration, GetBdfId_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bdf_id", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bdf_id(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetBdfId_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint64_t bdfid = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bdf_id", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bdf_id(kInvalidHandle, &bdfid);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetBdfId_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_bdf_id");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint64_t bdfid = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_bdf_id", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_bdf_id(gpus()[i], &bdfid);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_pci_throughput ----------------
TEST(GpuIntegration, GetPciThroughput_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "GetPciThroughput_NullOutput fails with Successful return, expected AMDSMI_STATUS_INVAL";

  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_throughput", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pci_throughput(any_gpu(), nullptr, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetPciThroughput_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint64_t sent = 0, received = 0, max_pkt = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_throughput", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pci_throughput(kInvalidHandle, &sent, &received, &max_pkt);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetPciThroughput_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_pci_throughput");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint64_t sent = 0, received = 0, max_pkt = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_throughput", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_pci_throughput(gpus()[i], &sent, &received, &max_pkt);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_pci_replay_counter ----------------
TEST(GpuIntegration, GetPciReplayCounter_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_replay_counter", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pci_replay_counter(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetPciReplayCounter_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  uint64_t counter = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_replay_counter", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_pci_replay_counter(kInvalidHandle, &counter);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetPciReplayCounter_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_pci_replay_counter");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint64_t counter = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_replay_counter", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_pci_replay_counter(gpus()[i], &counter);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_set_gpu_pci_bandwidth ----------------
TEST(GpuIntegration, SetPciBandwidth_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_pci_bandwidth", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_pci_bandwidth(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_get_pcie_info ----------------
TEST(GpuIntegration, GetPcieInfo_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_pcie_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_pcie_info(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetPcieInfo_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_pcie_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_pcie_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_pcie_info(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetPcieInfo_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_pcie_info returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause unknown, "
         "under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_pcie_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_pcie_info_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_pcie_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_pcie_info(gpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

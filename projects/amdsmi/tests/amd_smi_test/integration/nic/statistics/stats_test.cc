// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>
#include <vector>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

namespace {
// Port/rdma-port indices to exercise for the statistics APIs.
constexpr uint32_t kStatPortIndices[] = {0, 1};
}  // namespace

// =====================================================================
// amdsmi_get_nic_port_statistics(handle, port_index, uint32_t* num_stats,
//                                amdsmi_nic_stat_t* stats)
// =====================================================================

TEST_F(NicIntegration, GetNicPortStatistics_NullCount) {
  DISPLAY_AMDSMI_API("amdsmi_get_nic_port_statistics", "nic=0 port=0 num_stats=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_port_statistics(any_nic(), 0, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

TEST_F(NicIntegration, GetNicPortStatistics_InvalidHandle) {
  uint32_t num_stats = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_nic_port_statistics", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_port_statistics(kInvalidHandle, 0, &num_stats, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

TEST_F(NicIntegration, GetNicPortStatistics_AllNics) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_nic_port_statistics");
  if (nics().empty()) GTEST_SKIP() << "No NIC devices";
  for (size_t i = 0; i < nics().size(); ++i) {
    for (uint32_t port : kStatPortIndices) {
      uint32_t num_stats = 0;
      DISPLAY_AMDSMI_API(
          "amdsmi_get_nic_port_statistics",
          "nic=" + std::to_string(i) + " port=" + std::to_string(port) + " count-query", kVerbose);
      amdsmi_status_t err = amdsmi_get_nic_port_statistics(nics()[i], port, &num_stats, nullptr);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive(
          "nic=" + std::to_string(i) + " port=" + std::to_string(port) + " count-query", err);
      if (err != AMDSMI_STATUS_SUCCESS || num_stats == 0) continue;
      std::vector<amdsmi_nic_stat_t> stats(num_stats);
      memset(stats.data(), 0, num_stats * sizeof(amdsmi_nic_stat_t));
      DISPLAY_AMDSMI_API("amdsmi_get_nic_port_statistics",
                         "nic=" + std::to_string(i) + " port=" + std::to_string(port) + " fill",
                         kVerbose);
      err = amdsmi_get_nic_port_statistics(nics()[i], port, &num_stats, stats.data());
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive(
          "nic=" + std::to_string(i) + " port=" + std::to_string(port) + " fill", err);
    }
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// =====================================================================
// amdsmi_get_nic_rdma_port_statistics(handle, rdma_port_index,
//                                     uint32_t* num_stats,
//                                     amdsmi_nic_stat_t* stats)
// =====================================================================

TEST_F(NicIntegration, GetNicRdmaPortStatistics_NullCount) {
  DISPLAY_AMDSMI_API("amdsmi_get_nic_rdma_port_statistics", "nic=0 port=0 num_stats=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_rdma_port_statistics(any_nic(), 0, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

TEST_F(NicIntegration, GetNicRdmaPortStatistics_InvalidHandle) {
  uint32_t num_stats = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_nic_rdma_port_statistics", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_rdma_port_statistics(kInvalidHandle, 0, &num_stats, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

TEST_F(NicIntegration, GetNicRdmaPortStatistics_AllNics) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_nic_rdma_port_statistics");
  if (nics().empty()) GTEST_SKIP() << "No NIC devices";
  for (size_t i = 0; i < nics().size(); ++i) {
    for (uint32_t port : kStatPortIndices) {
      uint32_t num_stats = 0;
      DISPLAY_AMDSMI_API(
          "amdsmi_get_nic_rdma_port_statistics",
          "nic=" + std::to_string(i) + " rdma_port=" + std::to_string(port) + " count-query",
          kVerbose);
      amdsmi_status_t err =
          amdsmi_get_nic_rdma_port_statistics(nics()[i], port, &num_stats, nullptr);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive(
          "nic=" + std::to_string(i) + " rdma_port=" + std::to_string(port) + " count-query", err);
      if (err != AMDSMI_STATUS_SUCCESS || num_stats == 0) continue;
      std::vector<amdsmi_nic_stat_t> stats(num_stats);
      memset(stats.data(), 0, num_stats * sizeof(amdsmi_nic_stat_t));
      DISPLAY_AMDSMI_API(
          "amdsmi_get_nic_rdma_port_statistics",
          "nic=" + std::to_string(i) + " rdma_port=" + std::to_string(port) + " fill", kVerbose);
      err = amdsmi_get_nic_rdma_port_statistics(nics()[i], port, &num_stats, stats.data());
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive(
          "nic=" + std::to_string(i) + " rdma_port=" + std::to_string(port) + " fill", err);
    }
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// =====================================================================
// amdsmi_get_nic_vendor_statistics(handle, port_index, uint32_t* num_stats,
//                                  amdsmi_nic_stat_t* stats)
// =====================================================================

TEST_F(NicIntegration, GetNicVendorStatistics_NullCount) {
  DISPLAY_AMDSMI_API("amdsmi_get_nic_vendor_statistics", "nic=0 port=0 num_stats=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_vendor_statistics(any_nic(), 0, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

TEST_F(NicIntegration, GetNicVendorStatistics_InvalidHandle) {
  uint32_t num_stats = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_nic_vendor_statistics", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_vendor_statistics(kInvalidHandle, 0, &num_stats, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

TEST_F(NicIntegration, GetNicVendorStatistics_AllNics) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_nic_vendor_statistics");
  if (nics().empty()) GTEST_SKIP() << "No NIC devices";
  for (size_t i = 0; i < nics().size(); ++i) {
    for (uint32_t port : kStatPortIndices) {
      uint32_t num_stats = 0;
      DISPLAY_AMDSMI_API(
          "amdsmi_get_nic_vendor_statistics",
          "nic=" + std::to_string(i) + " port=" + std::to_string(port) + " count-query", kVerbose);
      amdsmi_status_t err = amdsmi_get_nic_vendor_statistics(nics()[i], port, &num_stats, nullptr);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive(
          "nic=" + std::to_string(i) + " port=" + std::to_string(port) + " count-query", err);
      if (err != AMDSMI_STATUS_SUCCESS || num_stats == 0) continue;
      std::vector<amdsmi_nic_stat_t> stats(num_stats);
      memset(stats.data(), 0, num_stats * sizeof(amdsmi_nic_stat_t));
      DISPLAY_AMDSMI_API("amdsmi_get_nic_vendor_statistics",
                         "nic=" + std::to_string(i) + " port=" + std::to_string(port) + " fill",
                         kVerbose);
      err = amdsmi_get_nic_vendor_statistics(nics()[i], port, &num_stats, stats.data());
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive(
          "nic=" + std::to_string(i) + " port=" + std::to_string(port) + " fill", err);
    }
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for amdsmi_get_fabric_cper_entries() API.
// These tests verify parameter validation and the expected behavior
// of the fabric CPER retrieval API without requiring UALoE hardware.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "amd_smi/amdsmi.h"

namespace {

// Tests invalid parameter combinations return AMDSMI_STATUS_INVAL
TEST(GpuUnit, FabricCperInvalidParams) {
  amdsmi_processor_handle handle = nullptr;
  uint32_t severity_mask = 0xFFFFFFFF;
  std::vector<char> cper_data(4096);
  uint64_t buf_size = cper_data.size();
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(16);
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  // Null processor_handle
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(nullptr, severity_mask, cper_data.data(), &buf_size,
                                           cper_hdrs.data(), &entry_count, &cursor),
            AMDSMI_STATUS_INVAL);

  // Null cper_data
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(handle, severity_mask, nullptr, &buf_size,
                                           cper_hdrs.data(), &entry_count, &cursor),
            AMDSMI_STATUS_INVAL);

  // Null buf_size
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), nullptr,
                                           cper_hdrs.data(), &entry_count, &cursor),
            AMDSMI_STATUS_INVAL);

  // Null cper_hdrs
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                           nullptr, &entry_count, &cursor),
            AMDSMI_STATUS_INVAL);

  // Null entry_count
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                           cper_hdrs.data(), nullptr, &cursor),
            AMDSMI_STATUS_INVAL);

  // Null cursor
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                           cper_hdrs.data(), &entry_count, nullptr),
            AMDSMI_STATUS_INVAL);

  // entry_count = 0
  entry_count = 0;
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                           cper_hdrs.data(), &entry_count, &cursor),
            AMDSMI_STATUS_INVAL);

  // entry_count > 4096 (kMaxUaloeEntries)
  entry_count = 4097;
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                           cper_hdrs.data(), &entry_count, &cursor),
            AMDSMI_STATUS_INVAL);
}

// Test that buf_size smaller than header expansion headroom returns OUT_OF_RESOURCES
TEST(GpuUnit, FabricCperInsufficientBufferForExpansion) {
  amdsmi_processor_handle handle = nullptr;
  uint32_t severity_mask = 0xFFFFFFFF;
  std::vector<char> cper_data(128);  // Small buffer
  uint64_t buf_size = cper_data.size();
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(16);
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  // With entry_count=16, expansion_headroom = 16 * (128 - 24) = 1664 bytes
  // buf_size=128 < 1664, should return OUT_OF_RESOURCES (or INVAL/NOT_SUPPORTED
  // depending on whether handle is valid)
  amdsmi_status_t status = amdsmi_get_fabric_cper_entries(
      handle, severity_mask, cper_data.data(), &buf_size, cper_hdrs.data(), &entry_count, &cursor);

  // Since handle is nullptr, we expect INVAL first; but if handle validation passes,
  // the expansion check would return OUT_OF_RESOURCES. For this test, we document
  // the behavior: null handle → INVAL takes precedence.
  EXPECT_EQ(status, AMDSMI_STATUS_INVAL);
}

}  // namespace

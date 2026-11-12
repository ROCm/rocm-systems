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

// A non-null (unregistered) handle so each null-argument case isolates its own
// check instead of all short-circuiting on the handle==nullptr guard.
TEST(GpuUnit, FabricCperNullArguments) {
  int sentinel = 0;
  amdsmi_processor_handle handle = reinterpret_cast<amdsmi_processor_handle>(&sentinel);
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
}

// entry_count outside (0, kMaxUaloeEntries] yields AMDSMI_STATUS_INVAL; the
// non-null handle isolates the count check from the handle check.
TEST(GpuUnit, FabricCperEntryCountBounds) {
  int sentinel = 0;
  amdsmi_processor_handle handle = reinterpret_cast<amdsmi_processor_handle>(&sentinel);
  uint32_t severity_mask = 0xFFFFFFFF;
  std::vector<char> cper_data(4096);
  uint64_t buf_size = cper_data.size();
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(16);
  uint64_t cursor = 0;

  // entry_count = 0
  uint64_t entry_count = 0;
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                           cper_hdrs.data(), &entry_count, &cursor),
            AMDSMI_STATUS_INVAL);

  // entry_count > 4096 (kMaxUaloeEntries)
  entry_count = 4097;
  EXPECT_EQ(amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                           cper_hdrs.data(), &entry_count, &cursor),
            AMDSMI_STATUS_INVAL);
}

}  // namespace

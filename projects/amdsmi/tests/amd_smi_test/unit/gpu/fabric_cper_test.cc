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
#include "test_common.h"

namespace {

// A non-null (unregistered) handle so each null-argument case isolates its own
// check instead of all short-circuiting on the handle==nullptr guard.
TEST(GpuUnit, FabricCperNullArguments) {
  amdsmi_status_t err;
  int sentinel = 0;
  amdsmi_processor_handle handle = reinterpret_cast<amdsmi_processor_handle>(&sentinel);
  uint32_t severity_mask = 0xFFFFFFFF;
  std::vector<char> cper_data(4096);
  uint64_t buf_size = cper_data.size();
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(16);
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  // Null processor_handle
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_cper_entries",
                     "FabricCperNullArguments, Null processor_handle", GetTestVerbosity() >= 1);
  err = amdsmi_get_fabric_cper_entries(nullptr, severity_mask, cper_data.data(), &buf_size,
                                       cper_hdrs.data(), &entry_count, &cursor);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);

  // Null cper_data
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_cper_entries", "FabricCperNullArguments, Null cper_data",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_fabric_cper_entries(handle, severity_mask, nullptr, &buf_size, cper_hdrs.data(),
                                       &entry_count, &cursor);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);

  // Null buf_size
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_cper_entries", "FabricCperNullArguments, Null buf_size",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), nullptr,
                                       cper_hdrs.data(), &entry_count, &cursor);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);

  // Null cper_hdrs
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_cper_entries", "FabricCperNullArguments, Null cper_hdrs",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size, nullptr,
                                       &entry_count, &cursor);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);

  // Null entry_count
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_cper_entries", "FabricCperNullArguments, Null entry_count",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                       cper_hdrs.data(), nullptr, &cursor);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);

  // Null cursor
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_cper_entries", "FabricCperNullArguments, Null cursor",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                       cper_hdrs.data(), &entry_count, nullptr);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

// entry_count outside (0, kMaxUaloeEntries] yields AMDSMI_STATUS_INVAL; the
// non-null handle isolates the count check from the handle check.
TEST(GpuUnit, FabricCperEntryCountBounds) {
  amdsmi_status_t err;
  int sentinel = 0;
  amdsmi_processor_handle handle = reinterpret_cast<amdsmi_processor_handle>(&sentinel);
  uint32_t severity_mask = 0xFFFFFFFF;
  std::vector<char> cper_data(4096);
  uint64_t buf_size = cper_data.size();
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(16);
  uint64_t cursor = 0;

  // entry_count = 0
  uint64_t entry_count = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_cper_entries",
                     "FabricCperEntryCountBounds, entry_count = 0", GetTestVerbosity() >= 1);
  err = amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                       cper_hdrs.data(), &entry_count, &cursor);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);

  // entry_count > 4096 (kMaxUaloeEntries)
  entry_count = 4097;
  DISPLAY_AMDSMI_API("amdsmi_get_fabric_cper_entries",
                     "FabricCperEntryCountBounds, entry_count > 4096 (kMaxUaloeEntries)",
                     GetTestVerbosity() >= 1);
  err = amdsmi_get_fabric_cper_entries(handle, severity_mask, cper_data.data(), &buf_size,
                                       cper_hdrs.data(), &entry_count, &cursor);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

}  // namespace

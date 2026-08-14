/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Pure-logic tests for RDC_FI_GPU_CU_OCCUPANCY and its percentage companion,
// which report how many compute units are occupied by processes on a GPU
// (sourced from amdsmi's per-process cu_occupancy, itself from KFD sysfs).
// These need no hardware: they lock down the field registration contract that
// the daemon and `rdci dmon -e CU_OCCUPANCY` depend on. A field missing from
// rdc_field.data / rdc_fields_supported becomes silently unqueryable, so these
// guard the wiring. The value itself needs a GPU with a running process and is
// covered by the hardware integration tests.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "common/rdc_fields_supported.h"
#include "rdc/rdc.h"

namespace {

// Both fields must be registered and valid, or the telemetry layer rejects them.
TEST(CuOccupancyField, IsValid) {
  EXPECT_TRUE(amd::rdc::is_field_valid(RDC_FI_GPU_CU_OCCUPANCY));
  EXPECT_TRUE(amd::rdc::is_field_valid(RDC_FI_GPU_CU_OCCUPANCY_PERCENT));
}

// The descriptor table must carry both with enum name, label and description,
// and mark them for display (rdci dmon -l relies on all of these).
TEST(CuOccupancyField, DescriptorMetadata) {
  auto& table = amd::rdc::get_field_id_description_from_id();

  auto raw = table.find(static_cast<uint32_t>(RDC_FI_GPU_CU_OCCUPANCY));
  ASSERT_NE(raw, table.end()) << "RDC_FI_GPU_CU_OCCUPANCY missing from descriptor table";
  EXPECT_EQ(raw->second.enum_name, "RDC_FI_GPU_CU_OCCUPANCY");
  EXPECT_EQ(raw->second.label, "CU_OCCUPANCY");
  EXPECT_FALSE(raw->second.description.empty());
  EXPECT_TRUE(raw->second.do_display);

  auto pct = table.find(static_cast<uint32_t>(RDC_FI_GPU_CU_OCCUPANCY_PERCENT));
  ASSERT_NE(pct, table.end()) << "RDC_FI_GPU_CU_OCCUPANCY_PERCENT missing from descriptor table";
  EXPECT_EQ(pct->second.enum_name, "RDC_FI_GPU_CU_OCCUPANCY_PERCENT");
  EXPECT_EQ(pct->second.label, "CU_OCCUPANCY_PERCENT");
  EXPECT_FALSE(pct->second.description.empty());
  EXPECT_TRUE(pct->second.do_display);
}

// Name -> id resolution must work by enum name, since the CLI accepts that form.
TEST(CuOccupancyField, NameLookupResolvesToId) {
  rdc_field_t id = RDC_FI_INVALID;

  ASSERT_TRUE(amd::rdc::get_field_id_from_name("RDC_FI_GPU_CU_OCCUPANCY", &id));
  EXPECT_EQ(id, RDC_FI_GPU_CU_OCCUPANCY);

  id = RDC_FI_INVALID;
  ASSERT_TRUE(amd::rdc::get_field_id_from_name("RDC_FI_GPU_CU_OCCUPANCY_PERCENT", &id));
  EXPECT_EQ(id, RDC_FI_GPU_CU_OCCUPANCY_PERCENT);

  // The C wrapper used by the CLI must agree.
  EXPECT_EQ(get_field_id_from_name("RDC_FI_GPU_CU_OCCUPANCY"), RDC_FI_GPU_CU_OCCUPANCY);
}

// field_id_string() (used throughout logging and the CLI) must return the label.
TEST(CuOccupancyField, FieldIdStringReturnsLabel) {
  EXPECT_STREQ(field_id_string(RDC_FI_GPU_CU_OCCUPANCY), "CU_OCCUPANCY");
  EXPECT_STREQ(field_id_string(RDC_FI_GPU_CU_OCCUPANCY_PERCENT), "CU_OCCUPANCY_PERCENT");
}

// Both belong to the GPU-usage block that starts at RDC_FI_GPU_UTIL = 500 and
// must stay below the page block at 550. They were appended after the existing
// members so no previously published id shifts.
TEST(CuOccupancyField, IdsInGpuUsageBlock) {
  EXPECT_GT(static_cast<uint32_t>(RDC_FI_GPU_CU_OCCUPANCY),
            static_cast<uint32_t>(RDC_FI_GPU_JPEG_BUSY_INST));
  EXPECT_LT(static_cast<uint32_t>(RDC_FI_GPU_CU_OCCUPANCY_PERCENT),
            static_cast<uint32_t>(RDC_FI_GPU_PAGE_RETRIED));
  EXPECT_EQ(static_cast<uint32_t>(RDC_FI_GPU_CU_OCCUPANCY_PERCENT),
            static_cast<uint32_t>(RDC_FI_GPU_CU_OCCUPANCY) + 1);
}

// Appending must not have shifted the ids that are already published.
TEST(CuOccupancyField, ExistingUsageIdsUnchanged) {
  EXPECT_EQ(static_cast<uint32_t>(RDC_FI_GPU_UTIL), 500u);
  EXPECT_EQ(static_cast<uint32_t>(RDC_FI_GPU_BUSY_PERCENT), 508u);
  EXPECT_EQ(static_cast<uint32_t>(RDC_FI_GPU_JPEG_BUSY_INST), 521u);
  EXPECT_EQ(static_cast<uint32_t>(RDC_FI_GPU_PAGE_RETRIED), 550u);
}

}  // namespace

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file virtual_lds_metadata_test.cpp
/// @brief Wire-format tests for virtual-LDS-only policy metadata.

#include "rocjitsu/code/dbt/virtual_lds.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace rocjitsu {
namespace {

TEST(VirtualLdsMetadata, RoundTripsWithoutSidecarOrKernargFields) {
  const std::vector<VirtualLdsKernelMetadata> input = {{
      .kernel_name = "kernel",
      .sidecar_variant_name = "virtual-lds",
      .static_lds_bytes = 70000,
      .normal_private_segment_size = 40,
      .virtual_private_segment_size = 96,
      .virtual_lds_base_sgpr = 8,
      .flags =
          static_cast<uint16_t>(kVirtualLdsFlagRuntimeStateBlock | kVirtualLdsFlagWorkgroupIdX),
  }};

  const auto parsed = parse_virtual_lds_metadata(serialize_virtual_lds_metadata(input));

  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto &record = parsed->front();
  EXPECT_EQ(record.kernel_name, input.front().kernel_name);
  EXPECT_EQ(record.sidecar_variant_name, input.front().sidecar_variant_name);
  EXPECT_EQ(record.static_lds_bytes, input.front().static_lds_bytes);
  EXPECT_EQ(record.normal_private_segment_size, input.front().normal_private_segment_size);
  EXPECT_EQ(record.virtual_private_segment_size, input.front().virtual_private_segment_size);
  EXPECT_EQ(record.virtual_lds_base_sgpr, input.front().virtual_lds_base_sgpr);
  EXPECT_EQ(record.flags, input.front().flags);
}

} // namespace
} // namespace rocjitsu

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

namespace {

// Wire layout mirrored from virtual_lds.cpp: a 24-byte header followed by
// fixed-size records and a trailing string blob. These offsets let the negative
// tests corrupt one field at a time without depending on the serializer.
constexpr size_t kHeaderSize = 24;
constexpr size_t kVersionOffset = 8;
constexpr size_t kRecordCountOffset = 12;
constexpr size_t kReservedOffset = 20;

std::vector<uint8_t> valid_metadata() {
  const std::vector<VirtualLdsKernelMetadata> input = {{
      .kernel_name = "kernel",
      .sidecar_variant_name = "virtual-lds",
      .static_lds_bytes = 70000,
      .normal_private_segment_size = 40,
      .virtual_private_segment_size = 96,
      .virtual_lds_base_sgpr = 8,
      .flags = kVirtualLdsFlagRuntimeStateBlock,
  }};
  return serialize_virtual_lds_metadata(input);
}

void write_u32(std::vector<uint8_t> &bytes, size_t offset, uint32_t value) {
  bytes[offset + 0] = static_cast<uint8_t>(value & 0xFF);
  bytes[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
  bytes[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFF);
  bytes[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFF);
}

} // namespace

TEST(VirtualLdsMetadata, RejectsTruncatedHeader) {
  auto bytes = valid_metadata();
  bytes.resize(kHeaderSize - 1);
  EXPECT_FALSE(parse_virtual_lds_metadata(bytes).has_value());
}

TEST(VirtualLdsMetadata, RejectsWrongMagic) {
  auto bytes = valid_metadata();
  bytes[0] ^= 0xFF;
  EXPECT_FALSE(parse_virtual_lds_metadata(bytes).has_value());
}

TEST(VirtualLdsMetadata, RejectsWrongVersion) {
  auto bytes = valid_metadata();
  write_u32(bytes, kVersionOffset, 0xDEAD);
  EXPECT_FALSE(parse_virtual_lds_metadata(bytes).has_value());
}

TEST(VirtualLdsMetadata, RejectsNonZeroReserved) {
  auto bytes = valid_metadata();
  write_u32(bytes, kReservedOffset, 1);
  EXPECT_FALSE(parse_virtual_lds_metadata(bytes).has_value());
}

TEST(VirtualLdsMetadata, RejectsRecordCountLargerThanBuffer) {
  auto bytes = valid_metadata();
  write_u32(bytes, kRecordCountOffset, 4096);
  EXPECT_FALSE(parse_virtual_lds_metadata(bytes).has_value());
}

TEST(VirtualLdsMetadata, RejectsUnknownFlagBits) {
  const std::vector<VirtualLdsKernelMetadata> input = {{
      .kernel_name = "kernel",
      .sidecar_variant_name = "virtual-lds",
      .static_lds_bytes = 70000,
      .normal_private_segment_size = 40,
      .virtual_private_segment_size = 96,
      .virtual_lds_base_sgpr = 8,
      // Bit 0 is outside kVirtualLdsKnownFlagsMask; the parser must reject it.
      .flags = static_cast<uint16_t>(kVirtualLdsFlagRuntimeStateBlock | 0x1u),
  }};
  EXPECT_FALSE(parse_virtual_lds_metadata(serialize_virtual_lds_metadata(input)).has_value());
}

} // namespace
} // namespace rocjitsu

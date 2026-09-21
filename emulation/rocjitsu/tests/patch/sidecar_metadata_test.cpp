// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sidecar_metadata_test.cpp
/// @brief Wire-format tests for generic sidecar descriptor metadata.

#include "rocjitsu/code/patch/sidecar_metadata.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rocjitsu {
namespace {

TEST(SidecarMetadata, RoundTripsIndependentVariants) {
  const std::vector<SidecarVariantMetadata> input = {
      {.kernel_name = "kernel",
       .variant_name = "virtual-lds",
       .normal_descriptor_vaddr = 0x1000,
       .variant_descriptor_vaddr = 0x2000},
      {.kernel_name = "kernel",
       .variant_name = "instrumented",
       .normal_descriptor_vaddr = 0x1000,
       .variant_descriptor_vaddr = 0x3000},
  };

  const auto bytes = serialize_sidecar_metadata(input);
  const auto parsed = parse_sidecar_metadata(bytes);

  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ((*parsed)[i].kernel_name, input[i].kernel_name);
    EXPECT_EQ((*parsed)[i].variant_name, input[i].variant_name);
    EXPECT_EQ((*parsed)[i].normal_descriptor_vaddr, input[i].normal_descriptor_vaddr);
    EXPECT_EQ((*parsed)[i].variant_descriptor_vaddr, input[i].variant_descriptor_vaddr);
  }
}

TEST(SidecarMetadata, HeaderBytesPinMagicAndVersion) {
  const std::vector<SidecarVariantMetadata> input = {{.kernel_name = "k", .variant_name = "v"}};
  const auto bytes = serialize_sidecar_metadata(input);

  ASSERT_GE(bytes.size(), 24u);
  constexpr std::array<uint8_t, 8> expected_magic = {'R', 'J', 'S', 'I', 'D', 'E', '1', '\0'};
  EXPECT_TRUE(std::equal(expected_magic.begin(), expected_magic.end(), bytes.begin()));
  uint32_t version = 0;
  std::memcpy(&version, bytes.data() + 8, sizeof(version));
  EXPECT_EQ(version, 1u);
}

TEST(SidecarMetadata, RejectsUnknownWireVersion) {
  const std::vector<SidecarVariantMetadata> input = {{.kernel_name = "k", .variant_name = "v"}};
  auto bytes = serialize_sidecar_metadata(input);
  ASSERT_GE(bytes.size(), 12u);
  const uint32_t unknown_version = 2;
  std::memcpy(bytes.data() + 8, &unknown_version, sizeof(unknown_version));

  EXPECT_FALSE(parse_sidecar_metadata(bytes).has_value());
}

TEST(SidecarMetadata, RejectsTruncatedHeader) {
  const std::vector<SidecarVariantMetadata> input = {{.kernel_name = "k", .variant_name = "v"}};
  auto bytes = serialize_sidecar_metadata(input);
  ASSERT_GE(bytes.size(), 24u);
  // Cut the header short so read_pod<MetadataHeader> cannot succeed.
  bytes.resize(23);
  EXPECT_FALSE(parse_sidecar_metadata(bytes).has_value());
}

TEST(SidecarMetadata, RejectsTruncatedRecord) {
  const std::vector<SidecarVariantMetadata> input = {
      {.kernel_name = "kernel", .variant_name = "virtual-lds"}};
  auto bytes = serialize_sidecar_metadata(input);
  // A 24-byte header plus one 32-byte record is the minimum for record_count=1.
  // Dropping bytes mid-record must be rejected by the per-record bounds check.
  ASSERT_GT(bytes.size(), 24u + 16u);
  bytes.resize(24u + 16u);
  EXPECT_FALSE(parse_sidecar_metadata(bytes).has_value());
}

TEST(SidecarMetadata, RejectsStringBytesExceedingBuffer) {
  const std::vector<SidecarVariantMetadata> input = {
      {.kernel_name = "kernel", .variant_name = "virtual-lds"}};
  auto bytes = serialize_sidecar_metadata(input);
  ASSERT_GE(bytes.size(), 24u);
  // string_bytes lives at header offset 16. Inflate it past the buffer so the
  // strings-subspan bounds check must reject the section.
  const uint32_t bogus_string_bytes = 0x10000u;
  std::memcpy(bytes.data() + 16, &bogus_string_bytes, sizeof(bogus_string_bytes));
  EXPECT_FALSE(parse_sidecar_metadata(bytes).has_value());
}

TEST(SidecarMetadata, RejectsNonZeroReservedField) {
  const std::vector<SidecarVariantMetadata> input = {{.kernel_name = "k", .variant_name = "v"}};
  auto bytes = serialize_sidecar_metadata(input);
  ASSERT_GE(bytes.size(), 24u);
  // reserved lives at header offset 20. A non-zero value must be rejected so the
  // field stays available for a forward-compatible extension.
  const uint32_t bogus_reserved = 1;
  std::memcpy(bytes.data() + 20, &bogus_reserved, sizeof(bogus_reserved));
  EXPECT_FALSE(parse_sidecar_metadata(bytes).has_value());
}

TEST(SidecarMetadata, RejectsRecordCountExceedingBuffer) {
  const std::vector<SidecarVariantMetadata> input = {
      {.kernel_name = "kernel", .variant_name = "virtual-lds"}};
  auto bytes = serialize_sidecar_metadata(input);
  ASSERT_GE(bytes.size(), 24u);
  // record_count lives at header offset 12. Inflate it so records_bytes pushes
  // strings_offset past the buffer; the strings-subspan bounds check must reject
  // it without integer overflow (records_bytes is computed in uint64_t).
  const uint32_t bogus_record_count = 0x10000u;
  std::memcpy(bytes.data() + 12, &bogus_record_count, sizeof(bogus_record_count));
  EXPECT_FALSE(parse_sidecar_metadata(bytes).has_value());
}

} // namespace
} // namespace rocjitsu

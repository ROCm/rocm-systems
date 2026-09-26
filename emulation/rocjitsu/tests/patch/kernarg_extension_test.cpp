// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernarg_extension_test.cpp
/// @brief Unit tests for rocjitsu kernarg wrapper layout helpers.

#include "rocjitsu/code/patch/kernarg_extension.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace {

TEST(KernargExtensionTest, LayoutAndWriteSupportsMultiplePayloads) {
  const std::array<rocjitsu::KernargExtensionPayloadLayout, 2> payloads = {{
      {.size = 3, .alignment = 4},
      {.size = 16, .alignment = 8},
  }};

  const auto layout =
      rocjitsu::make_kernarg_extension_layout(/*original_kernarg_size=*/10, std::span{payloads});
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(layout->original_kernarg_size, 10u);
  EXPECT_EQ(layout->original_kernarg_pointer_offset, 16u);
  ASSERT_EQ(layout->payload_offsets.size(), 2u);
  EXPECT_EQ(layout->payload_offsets[0], 24u);
  EXPECT_EQ(layout->payload_offsets[1], 32u);
  EXPECT_EQ(layout->wrapper_size, 48u);

  const std::array<uint8_t, 10> original = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::array<uint8_t, 3> payload0 = {0xAA, 0xBB, 0xCC};
  const std::array<uint8_t, 16> payload1 = {
      0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
  };
  const uint64_t original_pointer = 0x123456789ABCDEF0ull;
  const std::array<rocjitsu::KernargExtensionPayloadWrite, 2> writes = {{
      {.data = payload0.data(), .size = static_cast<uint32_t>(payload0.size())},
      {.data = payload1.data(), .size = static_cast<uint32_t>(payload1.size())},
  }};

  std::vector<uint8_t> wrapper(layout->wrapper_size, 0x5A);
  ASSERT_TRUE(rocjitsu::write_kernarg_extension_wrapper(
      std::span<uint8_t>(wrapper.data(), wrapper.size()), *layout, original.data(),
      original_pointer, std::span{writes}));

  EXPECT_TRUE(std::equal(original.begin(), original.end(), wrapper.begin()));
  uint64_t copied_pointer = 0;
  std::memcpy(&copied_pointer, wrapper.data() + layout->original_kernarg_pointer_offset,
              sizeof(copied_pointer));
  EXPECT_EQ(copied_pointer, original_pointer);
  EXPECT_TRUE(
      std::equal(payload0.begin(), payload0.end(), wrapper.begin() + layout->payload_offsets[0]));
  EXPECT_TRUE(
      std::equal(payload1.begin(), payload1.end(), wrapper.begin() + layout->payload_offsets[1]));
}

TEST(KernargExtensionTest, RejectsInvalidPayloadAlignment) {
  const rocjitsu::KernargExtensionPayloadLayout payload{.size = 4, .alignment = 3};
  EXPECT_FALSE(
      rocjitsu::make_kernarg_extension_layout(/*original_kernarg_size=*/0, std::span{&payload, 1})
          .has_value());
}

TEST(KernargExtensionTest, AllowsZeroSizeOriginalKernarg) {
  const rocjitsu::KernargExtensionPayloadLayout payload{.size = 24, .alignment = 8};
  const auto layout =
      rocjitsu::make_kernarg_extension_layout(/*original_kernarg_size=*/0, std::span{&payload, 1});
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(layout->original_kernarg_pointer_offset, 0u);
  ASSERT_EQ(layout->payload_offsets.size(), 1u);
  EXPECT_EQ(layout->payload_offsets[0], 8u);
  EXPECT_EQ(layout->wrapper_size, 32u);
}

TEST(KernargExtensionTest, MetadataRoundTripsNamedPayloadContracts) {
  const std::vector<rocjitsu::KernargExtensionMetadata> input = {{
      .kernel_name = "kernel",
      .variant_name = "variant",
      .original_kernarg_size = 24,
      .payloads =
          {
              {.size = 24, .alignment = 8, .name = "state"},
              {.size = 12, .alignment = 4, .name = "workgroup-ids"},
          },
  }};

  const auto parsed = rocjitsu::parse_kernarg_extension_metadata(
      rocjitsu::serialize_kernarg_extension_metadata(input));

  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->size(), 1u);
  const auto &extension = parsed->front();
  EXPECT_EQ(extension.kernel_name, input.front().kernel_name);
  EXPECT_EQ(extension.variant_name, input.front().variant_name);
  EXPECT_EQ(extension.original_kernarg_size, input.front().original_kernarg_size);
  ASSERT_EQ(extension.payloads.size(), 2u);
  for (size_t i = 0; i < extension.payloads.size(); ++i) {
    EXPECT_EQ(extension.payloads[i].name, input.front().payloads[i].name);
    EXPECT_EQ(extension.payloads[i].size, input.front().payloads[i].size);
    EXPECT_EQ(extension.payloads[i].alignment, input.front().payloads[i].alignment);
  }
}

// --- write_kernarg_extension_wrapper negative paths -------------------------

TEST(KernargExtensionTest, WrapperWriteRejectsWrongWrapperSize) {
  const rocjitsu::KernargExtensionPayloadLayout payload{.size = 8, .alignment = 8};
  const auto layout =
      rocjitsu::make_kernarg_extension_layout(/*original_kernarg_size=*/0, std::span{&payload, 1});
  ASSERT_TRUE(layout.has_value());

  const std::array<uint8_t, 8> payload_bytes = {};
  const std::array<rocjitsu::KernargExtensionPayloadWrite, 1> writes = {{
      {.data = payload_bytes.data(), .size = static_cast<uint32_t>(payload_bytes.size())},
  }};
  // Buffer one byte short of layout.wrapper_size must be rejected.
  std::vector<uint8_t> wrapper(layout->wrapper_size - 1, 0);
  EXPECT_FALSE(rocjitsu::write_kernarg_extension_wrapper(
      std::span<uint8_t>(wrapper.data(), wrapper.size()), *layout, /*original_kernarg=*/nullptr,
      /*original_kernarg_pointer=*/0, std::span{writes}));
}

TEST(KernargExtensionTest, WrapperWriteRejectsPayloadCountMismatch) {
  const rocjitsu::KernargExtensionPayloadLayout payload{.size = 8, .alignment = 8};
  const auto layout =
      rocjitsu::make_kernarg_extension_layout(/*original_kernarg_size=*/0, std::span{&payload, 1});
  ASSERT_TRUE(layout.has_value());

  std::vector<uint8_t> wrapper(layout->wrapper_size, 0);
  // layout expects one payload; supply zero writes.
  EXPECT_FALSE(rocjitsu::write_kernarg_extension_wrapper(
      std::span<uint8_t>(wrapper.data(), wrapper.size()), *layout, /*original_kernarg=*/nullptr,
      /*original_kernarg_pointer=*/0, std::span<const rocjitsu::KernargExtensionPayloadWrite>{}));
}

TEST(KernargExtensionTest, WrapperWriteRejectsNullOriginalWithNonZeroSize) {
  const rocjitsu::KernargExtensionPayloadLayout payload{.size = 8, .alignment = 8};
  const auto layout =
      rocjitsu::make_kernarg_extension_layout(/*original_kernarg_size=*/16, std::span{&payload, 1});
  ASSERT_TRUE(layout.has_value());

  const std::array<uint8_t, 8> payload_bytes = {};
  const std::array<rocjitsu::KernargExtensionPayloadWrite, 1> writes = {{
      {.data = payload_bytes.data(), .size = static_cast<uint32_t>(payload_bytes.size())},
  }};
  std::vector<uint8_t> wrapper(layout->wrapper_size, 0);
  // original_kernarg_size is 16 but the original pointer is null.
  EXPECT_FALSE(rocjitsu::write_kernarg_extension_wrapper(
      std::span<uint8_t>(wrapper.data(), wrapper.size()), *layout, /*original_kernarg=*/nullptr,
      /*original_kernarg_pointer=*/0, std::span{writes}));
}

TEST(KernargExtensionTest, WrapperWriteRejectsNullPayloadDataWithNonZeroSize) {
  const rocjitsu::KernargExtensionPayloadLayout payload{.size = 8, .alignment = 8};
  const auto layout =
      rocjitsu::make_kernarg_extension_layout(/*original_kernarg_size=*/0, std::span{&payload, 1});
  ASSERT_TRUE(layout.has_value());

  // Non-zero payload size with a null data pointer must be rejected.
  const std::array<rocjitsu::KernargExtensionPayloadWrite, 1> writes = {{
      {.data = nullptr, .size = 8},
  }};
  std::vector<uint8_t> wrapper(layout->wrapper_size, 0);
  EXPECT_FALSE(rocjitsu::write_kernarg_extension_wrapper(
      std::span<uint8_t>(wrapper.data(), wrapper.size()), *layout, /*original_kernarg=*/nullptr,
      /*original_kernarg_pointer=*/0, std::span{writes}));
}

TEST(KernargExtensionTest, WrapperWriteRejectsForgedOversizedOriginalKernarg) {
  // KernargExtensionLayout is public plain data; a forged layout can claim an
  // original_kernarg_size far larger than the wrapper. The writer must reject it
  // rather than memcpy nearly 4 GiB into a tiny buffer.
  rocjitsu::KernargExtensionLayout layout;
  layout.original_kernarg_size = std::numeric_limits<uint32_t>::max();
  layout.original_kernarg_pointer_offset = 8;
  layout.wrapper_size = 16;

  std::vector<uint8_t> wrapper(layout.wrapper_size, 0);
  EXPECT_FALSE(rocjitsu::write_kernarg_extension_wrapper(
      std::span<uint8_t>(wrapper.data(), wrapper.size()), layout,
      /*original_kernarg=*/wrapper.data(),
      /*original_kernarg_pointer=*/0, std::span<const rocjitsu::KernargExtensionPayloadWrite>{}));
}

// --- make_kernarg_extension_layout overflow guards --------------------------

TEST(KernargExtensionTest, LayoutRejectsPayloadSizeOverflow) {
  // The saved pointer already advances the cursor past 0, so a payload claiming
  // the full uint32 range cannot fit and the layout must fail closed.
  const rocjitsu::KernargExtensionPayloadLayout payload{
      .size = std::numeric_limits<uint32_t>::max(), .alignment = 8};
  EXPECT_FALSE(
      rocjitsu::make_kernarg_extension_layout(/*original_kernarg_size=*/0, std::span{&payload, 1})
          .has_value());
}

TEST(KernargExtensionTest, LayoutRejectsOriginalSizeAlignmentOverflow) {
  // Aligning an original size within 7 bytes of the uint32 max up to 8 overflows.
  const rocjitsu::KernargExtensionPayloadLayout payload{.size = 8, .alignment = 8};
  EXPECT_FALSE(rocjitsu::make_kernarg_extension_layout(
                   /*original_kernarg_size=*/std::numeric_limits<uint32_t>::max() - 1,
                   std::span{&payload, 1})
                   .has_value());
}

// --- parse_kernarg_extension_metadata negative paths ------------------------

namespace {
// Header field byte offsets (see MetadataHeader in kernarg_extension.cpp):
// magic[8]@0, version@8, extension_count@12, payload_count@16, string_bytes@20.
constexpr size_t kVersionOffset = 8;
constexpr size_t kExtensionCountOffset = 12;
constexpr size_t kPayloadCountOffset = 16;
constexpr size_t kStringBytesOffset = 20;
// ExtensionRecord starts at 24; its reserved field is the 8th uint32.
constexpr size_t kFirstExtensionReservedOffset = 24 + 7 * sizeof(uint32_t);

std::vector<uint8_t> valid_metadata_bytes() {
  const std::vector<rocjitsu::KernargExtensionMetadata> input = {{
      .kernel_name = "kernel",
      .variant_name = "variant",
      .original_kernarg_size = 24,
      .payloads = {{.size = 24, .alignment = 8, .name = "state"}},
  }};
  return rocjitsu::serialize_kernarg_extension_metadata(input);
}

void poke_u32(std::vector<uint8_t> &bytes, size_t offset, uint32_t value) {
  ASSERT_GE(bytes.size(), offset + sizeof(value));
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
} // namespace

TEST(KernargExtensionTest, ParseAcceptsValidMetadata) {
  // Positive control: the negative tests below all mutate valid_metadata_bytes(),
  // so they would pass vacuously if the baseline builder stopped producing a
  // parseable buffer. Anchor it here so each EXPECT_FALSE can only fail for its
  // intended mutation.
  EXPECT_TRUE(rocjitsu::parse_kernarg_extension_metadata(valid_metadata_bytes()).has_value());
}

TEST(KernargExtensionTest, ParseRejectsTruncatedHeader) {
  auto bytes = valid_metadata_bytes();
  ASSERT_GE(bytes.size(), 24u);
  bytes.resize(23); // one byte short of the 24-byte header
  EXPECT_FALSE(rocjitsu::parse_kernarg_extension_metadata(bytes).has_value());
}

TEST(KernargExtensionTest, ParseRejectsWrongMagic) {
  auto bytes = valid_metadata_bytes();
  bytes[0] ^= 0xFF; // corrupt the first magic byte
  EXPECT_FALSE(rocjitsu::parse_kernarg_extension_metadata(bytes).has_value());
}

TEST(KernargExtensionTest, ParseRejectsUnknownVersion) {
  auto bytes = valid_metadata_bytes();
  poke_u32(bytes, kVersionOffset, 2);
  EXPECT_FALSE(rocjitsu::parse_kernarg_extension_metadata(bytes).has_value());
}

TEST(KernargExtensionTest, ParseRejectsExtensionCountExceedingBuffer) {
  auto bytes = valid_metadata_bytes();
  poke_u32(bytes, kExtensionCountOffset, 0x10000u);
  EXPECT_FALSE(rocjitsu::parse_kernarg_extension_metadata(bytes).has_value());
}

TEST(KernargExtensionTest, ParseRejectsPayloadCountExceedingBuffer) {
  auto bytes = valid_metadata_bytes();
  poke_u32(bytes, kPayloadCountOffset, 0x10000u);
  EXPECT_FALSE(rocjitsu::parse_kernarg_extension_metadata(bytes).has_value());
}

TEST(KernargExtensionTest, ParseRejectsStringBytesExceedingBuffer) {
  auto bytes = valid_metadata_bytes();
  poke_u32(bytes, kStringBytesOffset, 0x10000u);
  EXPECT_FALSE(rocjitsu::parse_kernarg_extension_metadata(bytes).has_value());
}

TEST(KernargExtensionTest, ParseRejectsNonZeroExtensionReserved) {
  auto bytes = valid_metadata_bytes();
  poke_u32(bytes, kFirstExtensionReservedOffset, 1);
  EXPECT_FALSE(rocjitsu::parse_kernarg_extension_metadata(bytes).has_value());
}

} // namespace

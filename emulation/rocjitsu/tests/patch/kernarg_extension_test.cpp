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

} // namespace

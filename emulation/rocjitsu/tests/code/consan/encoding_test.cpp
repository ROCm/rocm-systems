// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/encoding.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>

namespace rocjitsu::consan::wire {
namespace {

TEST(ConSanEncoding, IntegersUseLiteralLittleEndianBytes) {
  const std::vector<uint8_t> expected = {0x78, 0x56, 0x34, 0x12, 0xef, 0xcd,
                                         0xab, 0x89, 0x67, 0x45, 0x23, 0x01};
  Encoder encoder;
  encoder.u32(0x12345678);
  encoder.u64(0x0123456789abcdef);
  EXPECT_EQ(encoder.take(), expected);

  Decoder decoder(expected);
  uint32_t narrow = 0;
  uint64_t wide = 0;
  ASSERT_TRUE(decoder.u32(narrow));
  ASSERT_TRUE(decoder.u64(wide));
  EXPECT_EQ(narrow, 0x12345678u);
  EXPECT_EQ(wide, uint64_t{0x0123456789abcdef});
  EXPECT_EQ(decoder.remaining(), 0u);
}

TEST(ConSanEncoding, MixedFieldsPreserveBinaryStringsAndRawBytes) {
  const std::array<uint8_t, 3> raw = {0xff, 0, 0x80};
  const std::string binary("A\0B", 3);
  Encoder encoder;
  encoder.u32(7);
  encoder.string(binary);
  encoder.bytes(raw);
  encoder.string("");
  encoder.u64(std::numeric_limits<uint64_t>::max());
  const std::vector<uint8_t> encoded = encoder.take();
  const std::vector<uint8_t> expected = {7,    0,    0,    0,    3,    0,    0,    0,   'A',
                                         0,    'B',  0xff, 0,    0x80, 0,    0,    0,   0,
                                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
  EXPECT_EQ(encoded, expected);

  Decoder decoder(encoded);
  uint32_t number = 0;
  std::string text;
  std::array<uint8_t, 3> copied{};
  uint64_t maximum = 0;
  ASSERT_TRUE(decoder.u32(number));
  ASSERT_TRUE(decoder.string(text, 3));
  EXPECT_EQ(number, 7u);
  EXPECT_EQ(text, binary);
  ASSERT_TRUE(decoder.bytes(copied));
  EXPECT_EQ(copied, raw);
  ASSERT_TRUE(decoder.string(text, 0));
  EXPECT_TRUE(text.empty());
  ASSERT_TRUE(decoder.u64(maximum));
  EXPECT_EQ(maximum, std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(decoder.remaining_bytes().empty());
}

TEST(ConSanEncoding, TruncatedIntegersAndRawBytesAreRejected) {
  const std::array<uint8_t, 8> bytes = {1, 2, 3, 4, 5, 6, 7, 8};
  for (size_t extent = 0; extent < bytes.size(); ++extent) {
    SCOPED_TRACE(extent);
    const std::span<const uint8_t> prefix = std::span(bytes).first(extent);
    Decoder decoder(prefix);
    uint64_t wide = 0;
    EXPECT_FALSE(decoder.u64(wide));
    // A caller can still consume the available prefix after a failed wide read.
    if (extent >= 4) {
      uint32_t narrow = 0;
      ASSERT_TRUE(decoder.u32(narrow));
      EXPECT_EQ(narrow, 0x04030201u);
    }
    Decoder raw_decoder(prefix);
    std::array<uint8_t, 8> copied{};
    EXPECT_FALSE(raw_decoder.bytes(copied));
  }
  for (size_t extent = 0; extent < 4; ++extent) {
    const std::span<const uint8_t> prefix = std::span(bytes).first(extent);
    Decoder decoder(prefix);
    uint32_t narrow = 0;
    EXPECT_FALSE(decoder.u32(narrow));
  }
}

TEST(ConSanEncoding, StringsRequireBoundedLengthAndCompletePayload) {
  const std::array<uint8_t, 7> encoded = {3, 0, 0, 0, 'a', 'b', 'c'};
  std::string text;
  Decoder over_limit(encoded);
  EXPECT_FALSE(over_limit.string(text, 2));
  Decoder within_limit(encoded);
  ASSERT_TRUE(within_limit.string(text, 3));
  EXPECT_EQ(text, "abc");
  for (size_t extent = 0; extent < encoded.size(); ++extent) {
    SCOPED_TRACE(extent);
    const std::span<const uint8_t> prefix = std::span(encoded).first(extent);
    Decoder truncated(prefix);
    EXPECT_FALSE(truncated.string(text, 3));
  }
  const std::array<uint8_t, 4> oversized = {0xff, 0xff, 0xff, 0xff};
  Decoder missing_payload(oversized);
  EXPECT_FALSE(missing_payload.string(text, std::numeric_limits<uint32_t>::max()));
}

TEST(ConSanEncoding, SkipChecksExtentWithoutLosingTheRemainingPayload) {
  const std::array<uint8_t, 6> bytes = {0xaa, 0xbb, 0x78, 0x56, 0x34, 0x12};
  Decoder decoder(bytes);
  EXPECT_TRUE(decoder.skip(0));
  EXPECT_FALSE(decoder.skip(std::numeric_limits<uint64_t>::max()));
  ASSERT_TRUE(decoder.skip(2));
  EXPECT_EQ(decoder.remaining(), 4u);
  EXPECT_EQ(decoder.remaining_bytes().front(), 0x78);
  EXPECT_FALSE(decoder.skip(5));
  uint32_t value = 0;
  ASSERT_TRUE(decoder.u32(value));
  EXPECT_EQ(value, 0x12345678u);
  EXPECT_FALSE(decoder.skip(1));
  EXPECT_TRUE(decoder.skip(0));
  EXPECT_TRUE(decoder.remaining_bytes().empty());

  Decoder exact(bytes);
  ASSERT_TRUE(exact.skip(bytes.size()));
  EXPECT_TRUE(exact.remaining_bytes().empty());
}

} // namespace
} // namespace rocjitsu::consan::wire

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_fuzz_core.h"

#include "rocjitsu/isa/decoder.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>

namespace rocjitsu::decode_fuzz {
namespace {

template <size_t N>
std::array<uint8_t, kWindowSize> make_window(const std::array<uint32_t, N> &words) {
  std::array<uint8_t, kWindowSize> result{};
  for (size_t word = 0; word < words.size(); ++word) {
    for (size_t byte = 0; byte < sizeof(uint32_t); ++byte)
      result[word * sizeof(uint32_t) + byte] =
          static_cast<uint8_t>((words[word] >> (byte * 8)) & 0xffu);
  }
  return result;
}

class DecodeFuzzCoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    decoder = create_decoder("gfx1250");
    ASSERT_NE(decoder, nullptr);
  }

  std::unique_ptr<Decoder> decoder;
};

TEST(DecodeFuzzTargetsTest, CreatesEachSupportedDecoder) {
  constexpr std::string_view targets[] = {"cdna1", "cdna2", "cdna3",   "cdna4", "rdna1",
                                          "rdna2", "rdna3", "rdna3_5", "rdna4", "gfx1250"};
  for (const std::string_view target : targets) {
    SCOPED_TRACE(target);
    EXPECT_NE(create_decoder(target), nullptr);
  }
}

TEST(DecodeFuzzTargetsTest, CreatesRegisteredGpuAliases) {
  constexpr std::string_view targets[] = {"gfx90a", "gfx942", "gfx950", "gfx1200", "gfx1201"};
  for (const std::string_view target : targets) {
    SCOPED_TRACE(target);
    EXPECT_NE(create_decoder(target), nullptr);
  }
}

TEST(DecodeFuzzTargetsTest, RejectsUnknownTarget) {
  EXPECT_THROW((void)create_decoder("unknown"), std::invalid_argument);
}

TEST_F(DecodeFuzzCoreTest, RejectsShortWindowBeforeDecode) {
  std::array<uint8_t, kWindowSize - 1> input{};
  EXPECT_THROW((void)decode_window(*decoder, input), std::invalid_argument);
}

TEST_F(DecodeFuzzCoreTest, TreatsInvalidInstructionAsNormalRejection) {
  const auto input = make_window(std::array<uint32_t, 1>{0xffffffffu});
  const DecodeRecord record = decode_window(*decoder, input);
  EXPECT_FALSE(record.valid);
  EXPECT_FALSE(record.rejection.empty());
}

TEST_F(DecodeFuzzCoreTest, DecodesFourByteInstruction) {
  const auto input = make_window(std::array<uint32_t, 1>{0xBF800000u});
  const DecodeRecord record = decode_window(*decoder, input);
  ASSERT_TRUE(record.valid);
  EXPECT_EQ(record.size, 4);
  EXPECT_EQ(record.mnemonic, "s_nop");
  EXPECT_EQ(record.disassembly, "s_nop 0");
}

TEST_F(DecodeFuzzCoreTest, PreservesTwelveByteLiteral) {
  const auto input = make_window(std::array<uint32_t, 3>{0x46040504u, 0x00000000u, 0xC1F00000u});
  const DecodeRecord record = decode_window(*decoder, input);
  ASSERT_TRUE(record.valid);
  EXPECT_EQ(record.size, 12);
  EXPECT_EQ(record.mnemonic, "v_fmamk_f64_e32");
  EXPECT_EQ(record.disassembly, "v_fmamk_f64_e32 v[2:3], v[4:5], 0xc1f0000000000000, v[2:3]");
}

TEST_F(DecodeFuzzCoreTest, DecodesSopTwoWithLiteralWithoutAborting) {
  const auto input =
      make_window(std::array<uint32_t, 4>{0x9842FFB4u, 0xFD9670E6u, 0x673F885Eu, 0x2036829Bu});
  const DecodeRecord record = decode_window(*decoder, input);
  ASSERT_TRUE(record.valid);
  EXPECT_EQ(record.size, 8);
  EXPECT_EQ(record.mnemonic, "s_cselect_b32");
}

TEST_F(DecodeFuzzCoreTest, PreservesSixteenByteInstruction) {
  const auto input =
      make_window(std::array<uint32_t, 4>{0xCC350000u, 0x04020900u, 0xCC330006u, 0x02026912u});
  const DecodeRecord record = decode_window(*decoder, input);
  ASSERT_TRUE(record.valid);
  EXPECT_EQ(record.size, 16);
  EXPECT_EQ(record.mnemonic, "v_wmma_scale_f32_16x16x128_f8f6f4");
  EXPECT_EQ(record.disassembly,
            "v_wmma_scale_f32_16x16x128_f8f6f4 v[6:13], v[18:33], v[52:67], 0, v0, v4");
}

TEST_F(DecodeFuzzCoreTest, AcceptsCompoundMnemonicDisassembly) {
  const auto input = make_window(std::array<uint32_t, 2>{0xCA500501u, 0x02000080u});
  const DecodeRecord record = decode_window(*decoder, input);
  ASSERT_TRUE(record.valid);
  EXPECT_EQ(record.size, 8);
  EXPECT_EQ(record.mnemonic, "v_dual_cndmask_b32 :: v_dual_mov_b32");
  EXPECT_EQ(record.disassembly, "v_dual_cndmask_b32 v2, v1, v2 :: v_dual_mov_b32 v1, 0");
}

TEST_F(DecodeFuzzCoreTest, ReusesDecoderAcrossManyIterations) {
  const auto input = make_window(std::array<uint32_t, 1>{0xBF800000u});
  for (int iteration = 0; iteration < 1000; ++iteration)
    ASSERT_TRUE(decode_window(*decoder, input).valid);
}

} // namespace
} // namespace rocjitsu::decode_fuzz

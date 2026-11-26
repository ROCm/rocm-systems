// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/legacy_memory_displacement_encoded_fields.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace rocjitsu {
namespace {

struct LegacyOffsetTarget {
  rj_code_target_id_t target;
  bool cdna;
  const char *name;
};
constexpr std::array<LegacyOffsetTarget, 3> kLegacyOffsetTargets = {{
    {.target = ROCJITSU_CODE_TARGET_GFX950, .cdna = true, .name = "Gfx950"},
    {.target = ROCJITSU_CODE_TARGET_GFX942, .cdna = true, .name = "Gfx942"},
    {.target = ROCJITSU_CODE_TARGET_GFX1100, .cdna = false, .name = "Gfx1100"},
}};

struct LegacyOffsetCase {
  std::array<uint32_t, 2> words;
  uint32_t expected_word0;
  std::string_view mnemonic;
  int32_t source_offset;
  uint32_t offset_mask;
  bool signed_offset;
};

std::array<LegacyOffsetCase, 4> legacy_offset_cases(bool cdna) {
  // Literal fixtures are independent of the generated builders. In particular,
  // the CDNA build_flat helper does not populate GLOBAL's thirteenth offset bit.
  if (cdna) {
    return {{
        {.words = {0xdd090004u, 0x067f0204u},
         .expected_word0 = 0xdd090008u,
         .mnemonic = "flat_atomic_add",
         .source_offset = 4,
         .offset_mask = 0xfffu,
         .signed_offset = false},
        {.words = {0xdd099ffcu, 0x067f0204u},
         .expected_word0 = 0xdd098008u,
         .mnemonic = "global_atomic_add",
         .source_offset = -4,
         .offset_mask = 0x1fffu,
         .signed_offset = true},
        {.words = {0xe108f004u, 0x80010204u},
         .expected_word0 = 0xe108f008u,
         .mnemonic = "buffer_atomic_add",
         .source_offset = 4,
         .offset_mask = 0xfffu,
         .signed_offset = false},
        {.words = {0xdc519ffcu, 0x057f0004u},
         .expected_word0 = 0xdc518008u,
         .mnemonic = "global_load_dword",
         .source_offset = -4,
         .offset_mask = 0x1fffu,
         .signed_offset = true},
    }};
  }
  return {{
      {.words = {0xdcd4e004u, 0x067c0204u},
       .expected_word0 = 0xdcd4e008u,
       .mnemonic = "flat_atomic_add_u32",
       .source_offset = 4,
       .offset_mask = 0xfffu,
       .signed_offset = false},
      {.words = {0xdcd6fffcu, 0x067c0204u},
       .expected_word0 = 0xdcd6e008u,
       .mnemonic = "global_atomic_add_u32",
       .source_offset = -4,
       .offset_mask = 0x1fffu,
       .signed_offset = true},
      {.words = {0xe0d47004u, 0x80c10204u},
       .expected_word0 = 0xe0d47008u,
       .mnemonic = "buffer_atomic_add_u32",
       .source_offset = 4,
       .offset_mask = 0xfffu,
       .signed_offset = false},
      {.words = {0xdc52fffcu, 0x057c0004u},
       .expected_word0 = 0xdc52e008u,
       .mnemonic = "global_load_b32",
       .source_offset = -4,
       .offset_mask = 0x1fffu,
       .signed_offset = true},
  }};
}

bool only_legacy_offset_changed(const std::array<uint32_t, 2> &source,
                                const std::array<uint32_t, 2> &replacement, uint32_t mask,
                                uint32_t expected_encoded) {
  return source[1] == replacement[1] && (source[0] & ~mask) == (replacement[0] & ~mask) &&
         (replacement[0] & mask) == expected_encoded;
}

void expect_legacy_operand_unchanged(const Operand *before, const Operand *after) {
  ASSERT_NE(before, nullptr);
  ASSERT_NE(after, nullptr);
  EXPECT_EQ(after->name(), before->name());
  EXPECT_EQ(after->encoding_value(), before->encoding_value());
  EXPECT_EQ(after->size_bits(), before->size_bits());
  EXPECT_EQ(after->decoded_vgpr(), before->decoded_vgpr());
  EXPECT_EQ(after->register_byte_mask(), before->register_byte_mask());
}

class LegacyMemoryDisplacementEncodedFields : public ::testing::TestWithParam<LegacyOffsetTarget> {
};

TEST_P(LegacyMemoryDisplacementEncodedFields, PreservesAndDecodesConcreteFlatGlobalAndBuffer) {
  const LegacyOffsetTarget &target = GetParam();
  for (const LegacyOffsetCase &fixture : legacy_offset_cases(target.cdna)) {
    SCOPED_TRACE(fixture.mnemonic);
    std::array<uint32_t, 2> source = fixture.words;
    EXPECT_EQ(amdgpu::read_legacy_memory_byte_offset(target.target, source), fixture.source_offset);
    const std::optional<std::array<uint32_t, 2>> result =
        amdgpu::replace_legacy_memory_byte_offset(target.target, source, 8);
    ASSERT_TRUE(result);
    const std::array<uint32_t, 2> expected = {fixture.expected_word0, fixture.words[1]};
    EXPECT_EQ(*result, expected);
    EXPECT_EQ(source, fixture.words);
    EXPECT_TRUE(only_legacy_offset_changed(source, *result, fixture.offset_mask, 8));

    std::unique_ptr<Decoder> decoder =
        Decoder::create(default_isa_target_registry(), target.target);
    ASSERT_TRUE(decoder);
    const DecodeResult before = decoder->decode_window(source);
    const DecodeResult after = decoder->decode_window(*result);
    ASSERT_TRUE(before.succeeded());
    ASSERT_TRUE(after.succeeded());
    EXPECT_EQ(before.value()->mnemonic(), fixture.mnemonic);
    EXPECT_EQ(after.value()->mnemonic(), fixture.mnemonic);
    EXPECT_EQ(before.value()->size(), 8);
    EXPECT_EQ(after.value()->size(), 8);
    EXPECT_EQ(before.value()->opcode(), after.value()->opcode());
    // RDNA3's formatter prints the raw13-bit value; address calculation sign extends it.
    // The signed byte offset itself is checked independently above.
    const int32_t displayed_offset = !target.cdna && fixture.signed_offset
                                         ? static_cast<int32_t>(source[0] & 0x1fffu)
                                         : fixture.source_offset;
    EXPECT_NE(before.value()->disassemble().find("offset:" + std::to_string(displayed_offset)),
              std::string::npos);
    EXPECT_NE(after.value()->disassemble().find("offset:8"), std::string::npos);
    ASSERT_EQ(before.value()->num_src_operands(), after.value()->num_src_operands());
    ASSERT_EQ(before.value()->num_dst_operands(), after.value()->num_dst_operands());
    for (int index = 0; index < before.value()->num_src_operands(); ++index)
      ASSERT_NO_FATAL_FAILURE(expect_legacy_operand_unchanged(before.value()->src_operand(index),
                                                              after.value()->src_operand(index)));
    for (int index = 0; index < before.value()->num_dst_operands(); ++index)
      ASSERT_NO_FATAL_FAILURE(expect_legacy_operand_unchanged(before.value()->dst_operand(index),
                                                              after.value()->dst_operand(index)));

    for (size_t word = 0; word < 2; ++word) {
      std::array<uint32_t, 2> corrupted = *result;
      corrupted[word] ^= 1u;
      EXPECT_FALSE(only_legacy_offset_changed(source, corrupted, fixture.offset_mask, 8));
    }
    std::array<uint32_t, 2> wrong_modifier = *result;
    wrong_modifier[0] ^= 0x2000u;
    EXPECT_FALSE(only_legacy_offset_changed(source, wrong_modifier, fixture.offset_mask, 8));
  }
}

TEST_P(LegacyMemoryDisplacementEncodedFields, ChecksBoundsAndPermitsRawNoOpAndUnalignedValues) {
  const LegacyOffsetTarget &target = GetParam();
  for (const LegacyOffsetCase &fixture : legacy_offset_cases(target.cdna)) {
    const std::array<int32_t, 6> values = {fixture.signed_offset ? -4096 : 0,
                                           fixture.signed_offset ? -1 : 1,
                                           0,
                                           1,
                                           fixture.source_offset,
                                           4095};
    for (int32_t value : values) {
      SCOPED_TRACE(fixture.mnemonic);
      SCOPED_TRACE(value);
      std::array<uint32_t, 2> source = fixture.words;
      const std::optional<std::array<uint32_t, 2>> result =
          amdgpu::replace_legacy_memory_byte_offset(target.target, source, value);
      ASSERT_TRUE(result);
      const uint32_t encoded = static_cast<uint32_t>(value < 0 ? value + 8192 : value);
      EXPECT_TRUE(only_legacy_offset_changed(source, *result, fixture.offset_mask, encoded));
      EXPECT_EQ(amdgpu::read_legacy_memory_byte_offset(target.target, *result), value);
      EXPECT_EQ(source, fixture.words);
      if (value == fixture.source_offset) {
        EXPECT_EQ(*result, source);
      }
    }
    for (int64_t invalid :
         {int64_t{4096}, fixture.signed_offset ? int64_t{-4097} : int64_t{-1},
          std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}) {
      std::array<uint32_t, 2> source = fixture.words;
      EXPECT_FALSE(amdgpu::replace_legacy_memory_byte_offset(target.target, source, invalid));
      EXPECT_EQ(source, fixture.words);
    }
  }
}

TEST_P(LegacyMemoryDisplacementEncodedFields, RejectsOtherTargetsFormatsExtentsAndSegments) {
  const LegacyOffsetTarget &target = GetParam();
  const std::array<uint32_t, 2> source = legacy_offset_cases(target.cdna).front().words;
  const std::array<uint32_t, 3> oversized = {source[0], source[1], 0xdeadbeefu};
  for (std::span<const uint32_t> invalid :
       {std::span<const uint32_t>(), std::span<const uint32_t>(source).first(1),
        std::span<const uint32_t>(oversized)}) {
    EXPECT_FALSE(amdgpu::read_legacy_memory_byte_offset(target.target, invalid));
    EXPECT_FALSE(amdgpu::replace_legacy_memory_byte_offset(target.target, invalid, 8));
  }
  for (rj_code_target_id_t invalid :
       {ROCJITSU_CODE_TARGET_INVALID, ROCJITSU_CODE_TARGET_GFX90A, ROCJITSU_CODE_TARGET_GFX1200,
        ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251}) {
    EXPECT_FALSE(amdgpu::read_legacy_memory_byte_offset(invalid, source));
    EXPECT_FALSE(amdgpu::replace_legacy_memory_byte_offset(invalid, source, 8));
  }
  for (uint32_t prefix : {0xd8000000u, 0xc4000000u, 0xe8000000u, 0xec000000u, 0u}) {
    const std::array<uint32_t, 2> wrong = {prefix | 4u, source[1]};
    EXPECT_FALSE(amdgpu::read_legacy_memory_byte_offset(target.target, wrong));
    EXPECT_FALSE(amdgpu::replace_legacy_memory_byte_offset(target.target, wrong, 8));
  }
  const uint32_t segment_shift = target.cdna ? 14 : 16;
  for (uint32_t segment : {1u, 3u}) {
    const std::array<uint32_t, 2> wrong = {
        (source[0] & ~(3u << segment_shift)) | (segment << segment_shift), source[1]};
    EXPECT_FALSE(amdgpu::read_legacy_memory_byte_offset(target.target, wrong));
    EXPECT_FALSE(amdgpu::replace_legacy_memory_byte_offset(target.target, wrong, 8));
  }
}

TEST_P(LegacyMemoryDisplacementEncodedFields, PreservesWholeRawFormatBeyondTopEightPrefixBits) {
  const LegacyOffsetTarget &target = GetParam();
  for (const LegacyOffsetCase &fixture : legacy_offset_cases(target.cdna)) {
    std::array<uint32_t, 2> source = fixture.words;
    // Bit25 is CDNA SC1 or an unused FLAT bit, and part of RDNA3 MUBUF's
    // opcode. Raw field editing must preserve it without a top-eight-bit gate.
    // GLOBAL bit13 is SVE on CDNA, adjacent to the signed immediate.
    const uint32_t preserved_bits =
        0x02000000u | (target.cdna && fixture.signed_offset ? 0x2000u : 0u);
    source[0] |= preserved_bits;
    source[1] = 0xa5b6c7d8u;
    const std::array<uint32_t, 2> pristine = source;
    const std::optional<std::array<uint32_t, 2>> result =
        amdgpu::replace_legacy_memory_byte_offset(target.target, source, 8);
    ASSERT_TRUE(result);
    EXPECT_EQ((*result)[0], fixture.expected_word0 | preserved_bits);
    EXPECT_EQ((*result)[1], 0xa5b6c7d8u);
    EXPECT_TRUE(only_legacy_offset_changed(source, *result, fixture.offset_mask, 8));
    EXPECT_EQ(source, pristine);
  }
}

TEST(LegacyMemoryDisplacementPadding, CdnaFlatBit12IsPreservedButGlobalBit12IsSignedOffset) {
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_TARGET_GFX950}) {
    // Equal low13 source bits mean unsigned4092 for FLAT, signed-4 for GLOBAL.
    constexpr std::array<uint32_t, 2> flat = {0xdd091ffcu, 0x067f0204u};
    constexpr std::array<uint32_t, 2> global = {0xdd099ffcu, 0x067f0204u};
    EXPECT_EQ(amdgpu::read_legacy_memory_byte_offset(target, flat), 4092);
    EXPECT_EQ(amdgpu::read_legacy_memory_byte_offset(target, global), -4);
    const std::optional<std::array<uint32_t, 2>> flat_result =
        amdgpu::replace_legacy_memory_byte_offset(target, flat, 8);
    const std::optional<std::array<uint32_t, 2>> global_result =
        amdgpu::replace_legacy_memory_byte_offset(target, global, 8);
    ASSERT_TRUE(flat_result);
    ASSERT_TRUE(global_result);
    const std::array<uint32_t, 2> expected_flat = {0xdd091008u, 0x067f0204u};
    const std::array<uint32_t, 2> expected_global = {0xdd098008u, 0x067f0204u};
    EXPECT_EQ(*flat_result, expected_flat);
    EXPECT_EQ(*global_result, expected_global);
    EXPECT_FALSE(amdgpu::replace_legacy_memory_byte_offset(target, flat, -4));
  }
}

TEST(LegacyMemoryDisplacementPadding, Rdna3FlatUsesUnsignedTwelveBitRange) {
  constexpr rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1100;
  constexpr std::array<uint32_t, 2> source = {0xdcd4e000u, 0x067c0204u};
  // RDNA3 Table 107 makes FLAT's thirteenth encoded bit ignored, while
  // GLOBAL uses it as the sign bit. The generic field operation keeps both
  // the unsigned FLAT bounds and its otherwise unused source bit intact.
  for (int64_t invalid : {int64_t{-4096}, int64_t{-4}, int64_t{-1}, int64_t{4096}}) {
    SCOPED_TRACE(invalid);
    EXPECT_FALSE(amdgpu::replace_legacy_memory_byte_offset(target, source, invalid));
  }
  constexpr std::array<uint32_t, 2> ignored_msb = {0xdcd4fffcu, 0x067c0204u};
  EXPECT_EQ(amdgpu::read_legacy_memory_byte_offset(target, ignored_msb), 4092);
  constexpr std::array<uint32_t, 2> global = {0xdcd6fffcu, 0x067c0204u};
  EXPECT_EQ(amdgpu::read_legacy_memory_byte_offset(target, global), -4);
  const std::optional<std::array<uint32_t, 2>> negative_global =
      amdgpu::replace_legacy_memory_byte_offset(target, global, -4096);
  ASSERT_TRUE(negative_global);
  constexpr std::array<uint32_t, 2> expected_global = {0xdcd6f000u, 0x067c0204u};
  EXPECT_EQ(*negative_global, expected_global);
}

TEST(LegacyMemoryDisplacementPadding, Rdna3FlatPreservesIgnoredBitTwelve) {
  constexpr rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1100;
  constexpr std::array<uint32_t, 2> source = {0xdcd4f008u, 0x067c0204u};
  const std::optional<std::array<uint32_t, 2>> unchanged =
      amdgpu::replace_legacy_memory_byte_offset(target, source, 8);
  ASSERT_TRUE(unchanged);
  EXPECT_EQ(*unchanged, source);
  const std::optional<std::array<uint32_t, 2>> changed =
      amdgpu::replace_legacy_memory_byte_offset(target, source, 12);
  ASSERT_TRUE(changed);
  constexpr std::array<uint32_t, 2> expected = {0xdcd4f00cu, 0x067c0204u};
  EXPECT_EQ(*changed, expected);
  EXPECT_EQ(amdgpu::read_legacy_memory_byte_offset(target, *changed), 12);
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  ASSERT_TRUE(decoder);
  const DecodeResult before = decoder->decode_window(source);
  const DecodeResult after = decoder->decode_window(*changed);
  ASSERT_TRUE(before.succeeded());
  ASSERT_TRUE(after.succeeded());
  EXPECT_EQ(before.value()->mnemonic(), "flat_atomic_add_u32");
  EXPECT_EQ(after.value()->mnemonic(), before.value()->mnemonic());
  EXPECT_EQ(after.value()->size(), 8);
}

INSTANTIATE_TEST_SUITE_P(Targets, LegacyMemoryDisplacementEncodedFields,
                         ::testing::ValuesIn(kLegacyOffsetTargets),
                         [](const ::testing::TestParamInfo<LegacyOffsetTarget> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu

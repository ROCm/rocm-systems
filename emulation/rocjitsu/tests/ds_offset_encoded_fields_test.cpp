// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ds_address_test_support.h"

#include "rocjitsu/isa/arch/amdgpu/shared/ds_offset_encoded_fields.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace rocjitsu {
namespace {

struct Target {
  rj_code_target_id_t id;
  bool cdna3_or_cdna4;
  const char *name;
};
constexpr std::array<Target, 7> kTargets = {{
    {.id = ROCJITSU_CODE_TARGET_GFX1100, .cdna3_or_cdna4 = false, .name = "Gfx1100"},
    {.id = ROCJITSU_CODE_TARGET_GFX1200, .cdna3_or_cdna4 = false, .name = "Gfx1200"},
    {.id = ROCJITSU_CODE_TARGET_GFX1201, .cdna3_or_cdna4 = false, .name = "Gfx1201"},
    {.id = ROCJITSU_CODE_TARGET_GFX942, .cdna3_or_cdna4 = true, .name = "Gfx942"},
    {.id = ROCJITSU_CODE_TARGET_GFX950, .cdna3_or_cdna4 = true, .name = "Gfx950"},
    {.id = ROCJITSU_CODE_TARGET_GFX1250, .cdna3_or_cdna4 = false, .name = "Gfx1250"},
    {.id = ROCJITSU_CODE_TARGET_GFX1251, .cdna3_or_cdna4 = false, .name = "Gfx1251"},
}};

bool only_offset0_changed(const std::array<uint32_t, 2> &source,
                          const std::array<uint32_t, 2> &replacement, uint32_t offset0) {
  return (source[0] & 0xffffff00u) == (replacement[0] & 0xffffff00u) &&
         source[1] == replacement[1] && (replacement[0] & 0xffu) == offset0;
}

class DsOffsetEncodedFields : public ::testing::TestWithParam<Target> {};

TEST_P(DsOffsetEncodedFields, ReplacesAtomicOffsetWithoutChangingDecodedOperands) {
  const Target &target = GetParam();
  // DS_ADD_RTN_U32 uses op[24:17] on CDNA3/4 and op[25:18] on RDNA3/4/CDNA5.
  // These independent complete encodings keep offset1=1, ADDR=v2, DATA0=v3, VDST=v5.
  const std::array<uint32_t, 2> source = {target.cdna3_or_cdna4 ? 0xd8400104u : 0xd8800104u,
                                          0x05000302u};
  const std::array<uint32_t, 2> pristine = source;
  const std::optional<uint32_t> original = amdgpu::read_ds_offset0_field(target.id, source);
  ASSERT_TRUE(original);
  EXPECT_EQ(*original, 4u);
  const std::optional<std::array<uint32_t, 2>> result =
      amdgpu::replace_ds_offset0_field(target.id, source, 8);
  ASSERT_TRUE(result);
  const std::array<uint32_t, 2> expected = {target.cdna3_or_cdna4 ? 0xd8400108u : 0xd8800108u,
                                            0x05000302u};
  EXPECT_EQ(*result, expected);
  EXPECT_EQ(source, pristine);
  EXPECT_TRUE(only_offset0_changed(source, *result, 8));
  std::array<uint32_t, 2> corrupted = *result;
  corrupted[0] ^= 0x100u;
  EXPECT_FALSE(only_offset0_changed(source, corrupted, 8));
  corrupted = *result;
  corrupted[1] ^= 0x100u;
  EXPECT_FALSE(only_offset0_changed(source, corrupted, 8));

  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.id);
  ASSERT_TRUE(decoder);
  const DecodeResult before = decoder->decode_window(source);
  const DecodeResult after = decoder->decode_window(*result);
  ASSERT_TRUE(before.succeeded());
  ASSERT_TRUE(after.succeeded());
  EXPECT_EQ(before.value()->mnemonic(), "ds_add_rtn_u32");
  EXPECT_EQ(after.value()->mnemonic(), before.value()->mnemonic());
  EXPECT_EQ(after.value()->opcode(), before.value()->opcode());
  EXPECT_EQ(before.value()->size(), 8);
  EXPECT_EQ(after.value()->size(), 8);
  ASSERT_NE(before.value()->raw_encoding(), nullptr);
  ASSERT_NE(after.value()->raw_encoding(), nullptr);
  EXPECT_EQ(before.value()->raw_encoding()[0] & 0xffffu, 260u);
  EXPECT_EQ(after.value()->raw_encoding()[0] & 0xffffu, 264u);
  ASSERT_EQ(after.value()->num_src_operands(), before.value()->num_src_operands());
  ASSERT_EQ(after.value()->num_dst_operands(), before.value()->num_dst_operands());
  for (int index = 0; index < before.value()->num_src_operands(); ++index) {
    SCOPED_TRACE(index);
    ASSERT_NO_FATAL_FAILURE(test_support::expect_same_ds_operand(
        before.value()->src_operand(index), after.value()->src_operand(index)));
  }
  for (int index = 0; index < before.value()->num_dst_operands(); ++index) {
    SCOPED_TRACE(index);
    ASSERT_NO_FATAL_FAILURE(test_support::expect_same_ds_operand(
        before.value()->dst_operand(index), after.value()->dst_operand(index)));
  }
}

TEST_P(DsOffsetEncodedFields, PreservesOtherFieldsAtBoundariesWithoutAlignmentPolicy) {
  const Target &target = GetParam();
  // Reserved/modifier bits and unused operands are preservation witnesses, not
  // a claim that this arbitrary opcode is legal or suitable for execution.
  constexpr std::array<uint32_t, 2> source = {0xdbffef04u, 0xa5b60302u};
  for (uint32_t value : {0u, 1u, 4u, 255u}) {
    SCOPED_TRACE(value);
    const std::optional<std::array<uint32_t, 2>> result =
        amdgpu::replace_ds_offset0_field(target.id, source, value);
    ASSERT_TRUE(result);
    EXPECT_EQ((*result)[0], 0xdbffef00u | value);
    EXPECT_EQ((*result)[1], 0xa5b60302u);
    EXPECT_TRUE(only_offset0_changed(source, *result, value));
    const std::optional<uint32_t> read_back = amdgpu::read_ds_offset0_field(target.id, *result);
    ASSERT_TRUE(read_back);
    EXPECT_EQ(*read_back, value);
  }
  const std::optional<std::array<uint32_t, 2>> unchanged =
      amdgpu::replace_ds_offset0_field(target.id, source, 4);
  ASSERT_TRUE(unchanged);
  EXPECT_EQ(*unchanged, source);
}

TEST_P(DsOffsetEncodedFields, RejectsInvalidInputsWithoutChangingSource) {
  const Target &target = GetParam();
  const std::array<uint32_t, 2> source = {target.cdna3_or_cdna4 ? 0xd8400104u : 0xd8800104u,
                                          0x05000302u};
  const std::array<uint32_t, 2> pristine = source;
  const std::array<uint32_t, 3> oversized = {source[0], source[1], 0};
  constexpr std::array<uint32_t, 2> non_ds = {0xbf820000u, 0x05000302u};
  for (std::span<const uint32_t> invalid :
       {std::span<const uint32_t>(), std::span<const uint32_t>(source).first(1),
        std::span<const uint32_t>(oversized), std::span<const uint32_t>(non_ds)}) {
    EXPECT_FALSE(amdgpu::read_ds_offset0_field(target.id, invalid));
    EXPECT_FALSE(amdgpu::replace_ds_offset0_field(target.id, invalid, 8));
  }
  for (rj_code_target_id_t unsupported :
       {ROCJITSU_CODE_TARGET_INVALID, ROCJITSU_CODE_TARGET_GFX90A}) {
    EXPECT_FALSE(amdgpu::read_ds_offset0_field(unsupported, source));
    EXPECT_FALSE(amdgpu::replace_ds_offset0_field(unsupported, source, 8));
  }
  for (uint32_t invalid : {256u, UINT32_MAX})
    EXPECT_FALSE(amdgpu::replace_ds_offset0_field(target.id, source, invalid));
  EXPECT_EQ(source, pristine);
}

TEST_P(DsOffsetEncodedFields, FormatOnlyDoesNotQualifyAnOffsetUsingOpcode) {
  const Target &target = GetParam();
  const std::array<uint32_t, 2> nop = {target.cdna3_or_cdna4 ? 0xd8280000u : 0xd8500000u, 0};
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.id);
  ASSERT_TRUE(decoder);
  const DecodeResult decoded = decoder->decode_window(nop);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->mnemonic(), "ds_nop");
  const std::optional<uint32_t> raw = amdgpu::read_ds_offset0_field(target.id, nop);
  ASSERT_TRUE(raw);
  EXPECT_EQ(*raw, 0u);
  const std::optional<std::array<uint32_t, 2>> result =
      amdgpu::replace_ds_offset0_field(target.id, nop, 1);
  ASSERT_TRUE(result);
  EXPECT_EQ((*result)[0], nop[0] | 1u);
  EXPECT_EQ((*result)[1], 0u);
}

INSTANTIATE_TEST_SUITE_P(Targets, DsOffsetEncodedFields, ::testing::ValuesIn(kTargets),
                         [](const ::testing::TestParamInfo<Target> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu

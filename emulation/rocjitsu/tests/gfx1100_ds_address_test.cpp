// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ds_address_test_support.h"
#include "rdna3_lds_fixture.h"

#include "rocjitsu/isa/arch/amdgpu/shared/ds_address_encoded_fields.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace rocjitsu {
namespace {

class Gfx1100DsAddress : public ::testing::TestWithParam<test::Rdna3LdsFixture> {};

TEST_P(Gfx1100DsAddress, ReplacesOnlyAddressAndMatchesIndependentAssemblerWords) {
  const test::Rdna3LdsFixture &fixture = GetParam();
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1100);
  ASSERT_TRUE(decoder);
  const DecodeResult before = decoder->decode_window(fixture.source);
  ASSERT_TRUE(before.succeeded());
  EXPECT_EQ(before.value()->mnemonic(), fixture.form);
  EXPECT_EQ(before.value()->opcode(), fixture.opcode);
  EXPECT_EQ(before.value()->disassemble(), fixture.assembly);

  const std::optional<std::array<uint32_t, 2>> replacement =
      amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, fixture.source, 6);
  ASSERT_TRUE(replacement);
  EXPECT_EQ(*replacement, fixture.expected);
  EXPECT_TRUE(test_support::only_ds_address_changed(fixture.source, *replacement, 6));
  const DecodeResult after = decoder->decode_window(*replacement);
  ASSERT_TRUE(after.succeeded());
  EXPECT_EQ(after.value()->mnemonic(), fixture.form);
  EXPECT_EQ(after.value()->opcode(), fixture.opcode);
  ASSERT_NO_FATAL_FAILURE(
      test_support::expect_ds_address_replacement(*before.value(), *after.value(), 2, 6));
}

TEST_P(Gfx1100DsAddress, PreservesGdsBitSeventeenAndReservedBitSixteenIndependently) {
  const test::Rdna3LdsFixture &fixture = GetParam();
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1100);
  ASSERT_TRUE(decoder);
  // These are raw-field preservation witnesses, including reserved-bit inputs;
  // the generic setter does not promise that they are legal to execute.
  for (uint32_t modifiers : {0u, 0x00010000u, 0x00020000u, 0x00030000u}) {
    SCOPED_TRACE(modifiers);
    std::array<uint32_t, 2> source = fixture.source;
    source[0] |= modifiers;
    std::array<uint32_t, 2> expected = fixture.expected;
    expected[0] |= modifiers;
    const std::optional<std::array<uint32_t, 2>> replacement =
        amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, source, 6);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(*replacement, expected);
    const DecodeResult decoded = decoder->decode_window(*replacement);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value()->mnemonic(), fixture.form);
    EXPECT_EQ(decoded.value()->opcode(), fixture.opcode);
    EXPECT_EQ(decoded.value()->disassemble().find(" gds") != std::string::npos,
              (modifiers & 0x00020000u) != 0);
  }
}

TEST_P(Gfx1100DsAddress, HandlesFullSelectorRangeAndRejectsMalformedInputs) {
  const test::Rdna3LdsFixture &fixture = GetParam();
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1100);
  ASSERT_TRUE(decoder);
  for (uint32_t address : {0u, 2u, 255u}) {
    SCOPED_TRACE(address);
    const std::optional<std::array<uint32_t, 2>> replacement =
        amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, fixture.source, address);
    ASSERT_TRUE(replacement);
    EXPECT_EQ((*replacement)[0], fixture.source[0]);
    EXPECT_EQ((*replacement)[1] & 0xffffff00u, fixture.source[1] & 0xffffff00u);
    const DecodeResult decoded = decoder->decode_window(*replacement);
    ASSERT_TRUE(decoded.succeeded());
    ASSERT_NE(decoded.value()->src_operand(0), nullptr);
    const std::optional<RegisterRef> register_ref = decoded.value()->src_operand(0)->decoded_vgpr();
    ASSERT_TRUE(register_ref);
    EXPECT_EQ(register_ref->index, address);
    if (address == 2) {
      EXPECT_EQ(*replacement, fixture.source);
    }
  }
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, {}, 6));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100,
                                                std::span(fixture.source).first(1), 6));
  const std::array<uint32_t, 3> extra_word = {fixture.source[0], fixture.source[1], 0xbf800000u};
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, extra_word, 6));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, fixture.source, 256));
  EXPECT_FALSE(
      amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, fixture.source, UINT32_MAX));
  const std::array<uint32_t, 2> wrong_format = {0xbf800000u, fixture.source[1]};
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, wrong_format, 6));
  // FLAT's prefix differs from DS by bit 26; omitting that mask bit would
  // accept this instruction even though the SOPP witness above still rejects.
  const std::array<uint32_t, 2> flat_load = {0xdc500000u, 0x0c7c0002u};
  const DecodeResult decoded_flat = decoder->decode_window(flat_load);
  ASSERT_TRUE(decoded_flat.succeeded());
  EXPECT_EQ(decoded_flat.value()->mnemonic(), "flat_load_b32");
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, flat_load, 6));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX90A, fixture.source, 6));
  EXPECT_TRUE(decoder->decode_window(std::span(fixture.source).first(1)).failed());
}

TEST(Gfx1100DsAddressControls, FormatSetterDoesNotQualifyAnAddressOperand) {
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1100);
  ASSERT_TRUE(decoder);
  // DS_NOP has no operands. ADDTID uses M0 and the lane ID, not the raw ADDR
  // field: its store's first source is DATA0 and its load's is implicit DSMEM.
  for (const std::array<uint32_t, 2> source : {
           std::array<uint32_t, 2>{0xd8500000u, 0},
           std::array<uint32_t, 2>{0xdac00000u, 0x00000800u},
           std::array<uint32_t, 2>{0xdac40000u, 0x0c000000u},
       }) {
    const DecodeResult before = decoder->decode_window(source);
    ASSERT_TRUE(before.succeeded());
    const std::optional<std::array<uint32_t, 2>> replacement =
        amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, source, 6);
    ASSERT_TRUE(replacement);
    const DecodeResult after = decoder->decode_window(*replacement);
    ASSERT_TRUE(after.succeeded());
    EXPECT_EQ(after.value()->disassemble(), before.value()->disassemble());
    ASSERT_EQ(after.value()->num_src_operands(), before.value()->num_src_operands());
    ASSERT_EQ(after.value()->num_dst_operands(), before.value()->num_dst_operands());
    for (int index = 0; index < before.value()->num_src_operands(); ++index)
      ASSERT_NO_FATAL_FAILURE(test_support::expect_same_ds_operand(
          before.value()->src_operand(index), after.value()->src_operand(index)));
    for (int index = 0; index < before.value()->num_dst_operands(); ++index)
      ASSERT_NO_FATAL_FAILURE(test_support::expect_same_ds_operand(
          before.value()->dst_operand(index), after.value()->dst_operand(index)));
  }
}

TEST(Gfx1100DsAddressControls, ConcreteDecoderRejectsOtherFamiliesTransposeOpcodes) {
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1100);
  ASSERT_TRUE(decoder);
  // RDNA3 ENC_DS opcode slots 227 (CDNA4 transpose), 252 and 253 (CDNA5
  // transpose) are absent from its XML and concrete generated decode table.
  for (const std::array<uint32_t, 2> source : {
           std::array<uint32_t, 2>{0xdb8c0004u, 0x0c000002u},
           std::array<uint32_t, 2>{0xdbf00004u, 0x0c000002u},
           std::array<uint32_t, 2>{0xdbf40004u, 0x0c000002u},
       }) {
    EXPECT_TRUE(decoder->decode_window(source).failed());
    // A raw format-only setter deliberately leaves opcode legality to decode.
    const std::optional<std::array<uint32_t, 2>> replacement =
        amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1100, source, 6);
    ASSERT_TRUE(replacement);
    EXPECT_EQ((*replacement)[0], source[0]);
    EXPECT_EQ((*replacement)[1], 0x0c000006u);
    EXPECT_TRUE(decoder->decode_window(*replacement).failed());
  }
}

INSTANTIATE_TEST_SUITE_P(ConcreteForms, Gfx1100DsAddress,
                         ::testing::ValuesIn(test::kRdna3LdsFixtures),
                         [](const ::testing::TestParamInfo<test::Rdna3LdsFixture> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu

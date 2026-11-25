// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ds_address_test_support.h"
#include "gfx1251_lds_fixture.h"

#include "rocjitsu/isa/arch/amdgpu/cdna5/encoded_fields.h"
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

class Gfx1251DsAddress : public testing::TestWithParam<test::Gfx1251LdsFixture> {};

TEST_P(Gfx1251DsAddress, PreservesConcreteInstructionAndOtherOperands) {
  const test::Gfx1251LdsFixture &fixture = GetParam();
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1251);
  ASSERT_NE(decoder, nullptr);
  const DecodeResult before = decoder->decode_window(fixture.source);
  ASSERT_TRUE(before.succeeded());
  EXPECT_EQ(before.value()->mnemonic(), fixture.form);

  const std::optional<std::array<uint32_t, 2>> replacement =
      amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, fixture.source, 6);
  ASSERT_TRUE(replacement);
  EXPECT_EQ(*replacement, fixture.expected);
  EXPECT_TRUE(test_support::only_ds_address_changed(fixture.source, *replacement, 6));
  const DecodeResult after = decoder->decode_window(*replacement);
  ASSERT_TRUE(after.succeeded());
  ASSERT_NO_FATAL_FAILURE(
      test_support::expect_ds_address_replacement(*before.value(), *after.value(), 2, 6));
  EXPECT_EQ(before.value()->src_operand(0)->vgpr_msb_role(), amdgpu::VgprMsbRole::Src0);
  EXPECT_EQ(after.value()->src_operand(0)->vgpr_msb_role(), amdgpu::VgprMsbRole::Src0);

  std::array<uint32_t, 2> corrupted = *replacement;
  corrupted[0] ^= 1u;
  EXPECT_FALSE(test_support::only_ds_address_changed(fixture.source, corrupted, 6));
  corrupted = *replacement;
  corrupted[1] ^= 1u << 8;
  EXPECT_FALSE(test_support::only_ds_address_changed(fixture.source, corrupted, 6));
}

TEST_P(Gfx1251DsAddress, PreservesUnusedFieldsAndEncodedSelectorBoundaries) {
  const test::Gfx1251LdsFixture &fixture = GetParam();
  // Format-only editing also preserves noncanonical padding and unused data
  // fields. These bit witnesses do not assert instruction legality.
  std::array<uint32_t, 2> source = fixture.source;
  source[0] |= 0x00030000u;
  source[1] |= 0xa5b60000u;
  for (uint32_t address : {0u, 2u, 6u, 255u}) {
    const std::optional<std::array<uint32_t, 2>> replacement =
        amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, source, address);
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(test_support::only_ds_address_changed(source, *replacement, address));
  }
  const std::optional<std::array<uint32_t, 2>> no_op =
      amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, fixture.source, 2);
  ASSERT_TRUE(no_op);
  EXPECT_EQ(*no_op, fixture.source);
}

TEST_P(Gfx1251DsAddress, RejectsWrongExtentUnencodableSelectorAndUnsupportedTarget) {
  const test::Gfx1251LdsFixture &fixture = GetParam();
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, {}, 6));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251,
                                                std::span(fixture.source).first(1), 6));
  const std::array<uint32_t, 3> oversized = {fixture.source[0], fixture.source[1], 0};
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, oversized, 6));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, fixture.source, 256));
  EXPECT_FALSE(
      amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, fixture.source, UINT32_MAX));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX90A, fixture.source, 6));
  const std::array<uint32_t, 2> non_ds = {0xbfb00000u, fixture.source[1]};
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, non_ds, 6));
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1251);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decoder->decode_window(std::span(fixture.source).first(1)).failed());
}

TEST(Gfx1251DsAddress, ImplicitAddressOperationsDoNotAcquireAnAddressOperand) {
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1251);
  ASSERT_NE(decoder, nullptr);
  for (const test::Gfx1251LdsFixture &fixture : test::kGfx1251ImplicitAddressLdsFixtures) {
    SCOPED_TRACE(fixture.form);
    const DecodeResult before = decoder->decode_window(fixture.source);
    ASSERT_TRUE(before.succeeded());
    EXPECT_EQ(before.value()->mnemonic(), fixture.form);
    EXPECT_EQ(before.value()->size(), 8);
    // The helper edits a format field, including an unused field. Its success
    // cannot establish that the instruction uses a VGPR address.
    const std::optional<std::array<uint32_t, 2>> replacement =
        amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, fixture.source, 6);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(*replacement, fixture.expected);
    const DecodeResult after = decoder->decode_window(*replacement);
    ASSERT_TRUE(after.succeeded());
    EXPECT_EQ(after.value()->mnemonic(), fixture.form);
    ASSERT_EQ(after.value()->num_src_operands(), before.value()->num_src_operands());
    ASSERT_EQ(after.value()->num_dst_operands(), before.value()->num_dst_operands());
    for (int index = 0; index < before.value()->num_src_operands(); ++index) {
      EXPECT_NE(before.value()->src_operand(index)->vgpr_msb_role(), amdgpu::VgprMsbRole::Src0);
      ASSERT_NO_FATAL_FAILURE(test_support::expect_same_ds_operand(
          before.value()->src_operand(index), after.value()->src_operand(index)));
    }
    for (int index = 0; index < before.value()->num_dst_operands(); ++index)
      ASSERT_NO_FATAL_FAILURE(test_support::expect_same_ds_operand(
          before.value()->dst_operand(index), after.value()->dst_operand(index)));
    EXPECT_FALSE(cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1251, fixture.source, 6));
  }
}

TEST(Gfx1251DsAddress, ConcreteTargetRetainsItsOwnFeaturesAndModeCapability) {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetBinding *gfx1250 = registry.find_target(ROCJITSU_CODE_TARGET_GFX1250);
  const IsaTargetBinding *gfx1251 = registry.find_target(ROCJITSU_CODE_TARGET_GFX1251);
  ASSERT_NE(gfx1250, nullptr);
  ASSERT_NE(gfx1251, nullptr);
  EXPECT_TRUE(gfx1250->capabilities.setreg_vgpr_msb_fixup);
  EXPECT_FALSE(gfx1251->capabilities.setreg_vgpr_msb_fixup);
  EXPECT_NE(gfx1250->capabilities.instruction_features, gfx1251->capabilities.instruction_features);
  std::unique_ptr<Decoder> gfx1250_decoder = Decoder::create(registry, gfx1250->public_id);
  std::unique_ptr<Decoder> gfx1251_decoder = Decoder::create(registry, gfx1251->public_id);
  ASSERT_NE(gfx1250_decoder, nullptr);
  ASSERT_NE(gfx1251_decoder, nullptr);
  // Public LLVM gfx1251_asm_vop3p.s: v_pk_add_f64 v[4:7], v[8:11], v[12:15].
  // Family decoding must not grant gfx1250 the gfx1251-only instruction.
  constexpr std::array<uint32_t, 2> packed_add = {0xcc4b4004u, 0x1a021908u};
  const DecodeResult accepted = gfx1251_decoder->decode_window(packed_add);
  ASSERT_TRUE(accepted.succeeded());
  EXPECT_EQ(accepted.value()->mnemonic(), "v_pk_add_f64");
  EXPECT_TRUE(gfx1250_decoder->decode_window(packed_add).failed());
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1251, packed_add, 6));
}

INSTANTIATE_TEST_SUITE_P(AllReadWriteForms, Gfx1251DsAddress,
                         testing::ValuesIn(test::kGfx1251LdsFixtures),
                         [](const testing::TestParamInfo<test::Gfx1251LdsFixture> &info) {
                           return std::string(info.param.form);
                         });

} // namespace
} // namespace rocjitsu

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/encoded_fields.h"
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

namespace rocjitsu {
namespace {

bool only_address_changed(const std::array<uint32_t, 2> &source,
                          const std::array<uint32_t, 2> &replacement, uint32_t address) {
  return source[0] == replacement[0] &&
         (source[1] & 0xffffff00u) == (replacement[1] & 0xffffff00u) &&
         (replacement[1] & 0xffu) == address;
}

TEST(Rdna4EncodedFields, PreservesDsStoreFieldsAndDecodesReplacement) {
  const std::array<uint32_t, 2> source =
      rdna4::build_vds(rdna4::kDsStoreB32Vds, {.offset0 = 4, .offset1 = 1, .addr = 2, .data0 = 3});
  const std::optional<std::array<uint32_t, 2>> replacement =
      rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1201, source, 6);
  ASSERT_TRUE(replacement);
  EXPECT_TRUE(only_address_changed(source, *replacement, 6));
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1201);
  ASSERT_TRUE(decoder);
  DecodeResult decoded = decoder->decode_window(*replacement);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->mnemonic(), "ds_store_b32");
  ASSERT_NE(decoded.value()->src_operand(0), nullptr);
  ASSERT_NE(decoded.value()->src_operand(1), nullptr);
  EXPECT_EQ(decoded.value()->src_operand(0)->encoding_value(), 6u);
  EXPECT_EQ(decoded.value()->src_operand(1)->encoding_value(), 3u);
  std::array<uint32_t, 2> corrupted = *replacement;
  corrupted[1] ^= 1u << 8;
  EXPECT_FALSE(only_address_changed(source, corrupted, 6));
}

TEST(Rdna4EncodedFields, RejectsUnsupportedTargetShapeAndRegister) {
  const std::array<uint32_t, 2> source =
      rdna4::build_vds(rdna4::kDsStoreB32Vds, {.addr = 2, .data0 = 3});
  EXPECT_FALSE(rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1100, source, 6));
  EXPECT_FALSE(rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1201,
                                                   std::span(source).first(1), 6));
  const std::array<uint32_t, 3> oversized = {source[0], source[1], 0};
  EXPECT_FALSE(rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1201, oversized, 6));
  EXPECT_FALSE(rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1201, source, 256));
  const std::array<uint32_t, 2> other = rdna4::build_vds(rdna4::kDsLoadB32Vds, {.addr = 2});
  EXPECT_FALSE(rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1201, other, 6));
  for (uint32_t value : {0u, 255u}) {
    const std::optional<std::array<uint32_t, 2>> replacement =
        rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1201, source, value);
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(only_address_changed(source, *replacement, value));
  }
}

TEST(Rdna4EncodedFields, Gfx1200PreservesSourceAndRejectsInvalidInputs) {
  // llvm-mc encodes ds_store_b32 v2, v3 offset:4 identically on gfx1200 and gfx1201.
  constexpr std::array<uint32_t, 2> source = {0xd8340004u, 0x00000302u};
  const std::optional<std::array<uint32_t, 2>> replacement =
      rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1200, source, 6);
  ASSERT_TRUE(replacement);
  EXPECT_TRUE(only_address_changed(source, *replacement, 6));
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1200);
  ASSERT_TRUE(decoder);
  DecodeResult decoded = decoder->decode_window(*replacement);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->mnemonic(), "ds_store_b32");
  ASSERT_NE(decoded.value()->src_operand(0), nullptr);
  ASSERT_NE(decoded.value()->src_operand(1), nullptr);
  EXPECT_EQ(decoded.value()->src_operand(0)->encoding_value(), 6u);
  EXPECT_EQ(decoded.value()->src_operand(1)->encoding_value(), 3u);
  std::array<uint32_t, 2> corrupted = *replacement;
  corrupted[1] ^= 1u << 8;
  EXPECT_FALSE(only_address_changed(source, corrupted, 6));
  EXPECT_FALSE(rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1200,
                                                   std::span(source).first(1), 6));
  const std::array<uint32_t, 3> oversized = {source[0], source[1], 0};
  EXPECT_FALSE(rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1200, oversized, 6));
  EXPECT_FALSE(rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1200, source, 256));
  const std::array<uint32_t, 2> other = rdna4::build_vds(rdna4::kDsLoadB32Vds, {.addr = 2});
  EXPECT_FALSE(rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1200, other, 6));
  for (uint32_t value : {0u, 255u}) {
    const std::optional<std::array<uint32_t, 2>> boundary =
        rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1200, source, value);
    ASSERT_TRUE(boundary);
    EXPECT_TRUE(only_address_changed(source, *boundary, value));
  }
}

TEST(Rdna4EncodedFields, PreservesLiteralPaddingAndUnusedFields) {
  // Bits 16 and 17 are padding in gfx1201 VDS; the remaining high bytes of
  // word one are data and destination fields, even when unused by this form.
  constexpr std::array<uint32_t, 2> source = {0xd8370104u, 0xa5b60302u};
  constexpr std::array<uint32_t, 2> expected = {0xd8370104u, 0xa5b60306u};
  const std::optional<std::array<uint32_t, 2>> replacement =
      rdna4::replace_ds_store_b32_address(ROCJITSU_CODE_TARGET_GFX1201, source, 6);
  ASSERT_TRUE(replacement);
  EXPECT_EQ(*replacement, expected);
}

TEST(Rdna4EncodedFields, LoadAddressPreservesSourceAndRejectsOtherForms) {
  // llvm-mc gfx1200/gfx1201: ds_load_b32 v5, v2 offset:4.
  constexpr std::array<uint32_t, 2> source = {0xd8d80004u, 0x05000002u};
  constexpr std::array<uint32_t, 2> expected = {0xd8d80004u, 0x05000006u};
  constexpr std::array<uint32_t, 2> dual = {0xd8dc0804u, 0x04000002u};
  constexpr std::array<uint32_t, 2> store = {0xd8340004u, 0x00000302u};
  constexpr std::array<uint32_t, 2> unselected_bits = {0xd8db0104u, 0xa5b60302u};
  constexpr std::array<uint32_t, 2> expected_bits = {0xd8db0104u, 0xa5b60306u};
  constexpr std::array<uint32_t, 3> oversized = {source[0], source[1], 0};
  EXPECT_FALSE(rdna4::replace_ds_load_b32_address(ROCJITSU_CODE_TARGET_GFX1100, source, 6));
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_TARGET_GFX1201}) {
    SCOPED_TRACE(static_cast<int>(target));
    const std::optional<std::array<uint32_t, 2>> result =
        rdna4::replace_ds_load_b32_address(target, source, 6);
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, expected);
    std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
    ASSERT_TRUE(decoder);
    DecodeResult decoded = decoder->decode_window(*result);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value()->mnemonic(), "ds_load_b32");
    ASSERT_NE(decoded.value()->src_operand(0), nullptr);
    ASSERT_NE(decoded.value()->dst_operand(0), nullptr);
    EXPECT_EQ(decoded.value()->src_operand(0)->encoding_value(), 6u);
    EXPECT_EQ(decoded.value()->dst_operand(0)->encoding_value(), 5u);
    std::array<uint32_t, 2> corrupted = *result;
    corrupted[1] ^= 1u << 24;
    EXPECT_FALSE(only_address_changed(source, corrupted, 6));
    EXPECT_FALSE(rdna4::replace_ds_load_b32_address(target, std::span(source).first(1), 6));
    EXPECT_FALSE(rdna4::replace_ds_load_b32_address(target, oversized, 6));
    EXPECT_FALSE(rdna4::replace_ds_load_b32_address(target, source, 256));
    EXPECT_FALSE(rdna4::replace_ds_load_b32_address(target, dual, 6));
    EXPECT_FALSE(rdna4::replace_ds_load_b32_address(target, store, 6));
    const std::optional<std::array<uint32_t, 2>> preserved =
        rdna4::replace_ds_load_b32_address(target, unselected_bits, 6);
    ASSERT_TRUE(preserved);
    EXPECT_EQ(*preserved, expected_bits);
    for (uint32_t value : {0u, 255u}) {
      const std::optional<std::array<uint32_t, 2>> boundary =
          rdna4::replace_ds_load_b32_address(target, source, value);
      ASSERT_TRUE(boundary);
      EXPECT_TRUE(only_address_changed(source, *boundary, value));
    }
  }
}

} // namespace
} // namespace rocjitsu

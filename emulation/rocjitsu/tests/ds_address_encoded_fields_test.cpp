// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ds_address_test_support.h"

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
#include <string_view>

namespace rocjitsu {
namespace {

struct Target {
  rj_code_target_id_t id;
  bool rdna;
  const char *name;
};
constexpr std::array<Target, 7> kTargets = {{
    {.id = ROCJITSU_CODE_TARGET_GFX1100, .rdna = true, .name = "Gfx1100"},
    {.id = ROCJITSU_CODE_TARGET_GFX1200, .rdna = true, .name = "Gfx1200"},
    {.id = ROCJITSU_CODE_TARGET_GFX1201, .rdna = true, .name = "Gfx1201"},
    {.id = ROCJITSU_CODE_TARGET_GFX942, .rdna = false, .name = "Gfx942"},
    {.id = ROCJITSU_CODE_TARGET_GFX950, .rdna = false, .name = "Gfx950"},
    {.id = ROCJITSU_CODE_TARGET_GFX1250, .rdna = true, .name = "Gfx1250"},
    {.id = ROCJITSU_CODE_TARGET_GFX1251, .rdna = true, .name = "Gfx1251"},
}};
struct Fixture {
  std::array<uint32_t, 2> source;
  std::array<uint32_t, 2> expected;
  std::string_view mnemonic;
};
// llvm-mc witnesses for each concrete target, byte-for-byte expected output.
constexpr std::array<Fixture, 4> kRdna = {{
    {.source = {0xd8d80004u, 0x01000002u},
     .expected = {0xd8d80004u, 0x01000006u},
     .mnemonic = "ds_load_b32"},
    {.source = {0xd8340004u, 0x00000302u},
     .expected = {0xd8340004u, 0x00000306u},
     .mnemonic = "ds_store_b32"},
    {.source = {0xd9dc0301u, 0x04000002u},
     .expected = {0xd9dc0301u, 0x04000006u},
     .mnemonic = "ds_load_2addr_b64"},
    {.source = {0xd8380502u, 0x00040302u},
     .expected = {0xd8380502u, 0x00040306u},
     .mnemonic = "ds_store_2addr_b32"},
}};
constexpr std::array<Fixture, 4> kCdna = {{
    {.source = {0xd86c0004u, 0x01000002u},
     .expected = {0xd86c0004u, 0x01000006u},
     .mnemonic = "ds_read_b32"},
    {.source = {0xd81a0004u, 0x00000302u},
     .expected = {0xd81a0004u, 0x00000306u},
     .mnemonic = "ds_write_b32"},
    {.source = {0xd8ee0301u, 0x04000002u},
     .expected = {0xd8ee0301u, 0x04000006u},
     .mnemonic = "ds_read2_b64"},
    {.source = {0xd89e0502u, 0x00060402u},
     .expected = {0xd89e0502u, 0x00060406u},
     .mnemonic = "ds_write2st64_b64"},
}};
constexpr std::array<Fixture, 4> kCdna5 = {{
    {.source = {0xdbf80004u, 0x04000002u},
     .expected = {0xdbf80004u, 0x04000006u},
     .mnemonic = "ds_load_b96"},
    {.source = {0xdb780004u, 0x00000402u},
     .expected = {0xdb780004u, 0x00000406u},
     .mnemonic = "ds_store_b96"},
    {.source = {0xd8e80004u, 0x01000002u},
     .expected = {0xd8e80004u, 0x01000006u},
     .mnemonic = "ds_load_u8"},
    {.source = {0xd8780004u, 0x00000302u},
     .expected = {0xd8780004u, 0x00000306u},
     .mnemonic = "ds_store_b8"},
}};
class DsAddressEncodedFields : public ::testing::TestWithParam<Target> {};

TEST_P(DsAddressEncodedFields, ReplacesSingleAndTwoAddressFieldsLosslessly) {
  const Target &target = GetParam();
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.id);
  ASSERT_TRUE(decoder);
  const std::array<Fixture, 4> &fixtures =
      (target.id == ROCJITSU_CODE_TARGET_GFX1250 || target.id == ROCJITSU_CODE_TARGET_GFX1251)
          ? kCdna5
      : target.rdna ? kRdna
                    : kCdna;
  for (const Fixture &fixture : fixtures) {
    SCOPED_TRACE(fixture.mnemonic);
    const std::optional<std::array<uint32_t, 2>> result =
        amdgpu::replace_ds_address_field(target.id, fixture.source, 6);
    EXPECT_TRUE(result);
    if (!result) {
      continue;
    }
    EXPECT_EQ(*result, fixture.expected);
    EXPECT_TRUE(test_support::only_ds_address_changed(fixture.source, *result, 6));
    const DecodeResult before = decoder->decode_window(fixture.source);
    const DecodeResult after = decoder->decode_window(*result);
    ASSERT_TRUE(before.succeeded());
    ASSERT_TRUE(after.succeeded());
    EXPECT_EQ(before.value()->mnemonic(), fixture.mnemonic);
    EXPECT_EQ(after.value()->mnemonic(), fixture.mnemonic);
    ASSERT_NO_FATAL_FAILURE(
        test_support::expect_ds_address_replacement(*before.value(), *after.value(), 2, 6));
  }
}

TEST_P(DsAddressEncodedFields, PreservesAllNonAddressBitsAndRejectsInvalidInputs) {
  const Target &target = GetParam();
  // Preserve both offset bytes, opcode, all other fields and reserved bits.
  // CDNA ACC/GDS bits and RDNA reserved bits are bit-preservation witnesses,
  // not a claim that this exact instruction is legal to execute.
  const std::array<uint32_t, 2> source = target.rdna
                                             ? std::array<uint32_t, 2>{0xdbffef01u, 0xa5b60302u}
                                             : std::array<uint32_t, 2>{0xdbffef01u, 0xa5b60402u};
  for (uint32_t value : {0u, 6u, 255u}) {
    const std::optional<std::array<uint32_t, 2>> result =
        amdgpu::replace_ds_address_field(target.id, source, value);
    ASSERT_TRUE(result);
    EXPECT_EQ((*result)[0], source[0]);
    EXPECT_EQ((*result)[1], (source[1] & 0xffffff00u) | value);
  }
  EXPECT_FALSE(amdgpu::replace_ds_address_field(target.id, {}, 6));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(target.id, std::span(source).first(1), 6));
  const std::array<uint32_t, 3> oversized = {source[0], source[1], 0};
  EXPECT_FALSE(amdgpu::replace_ds_address_field(target.id, oversized, 6));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(target.id, source, 256));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(target.id, source, UINT32_MAX));
  const std::array<uint32_t, 2> non_ds = {0xbf820000u, 0x01000302u};
  EXPECT_FALSE(amdgpu::replace_ds_address_field(target.id, non_ds, 6));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_INVALID, source, 6));
  EXPECT_FALSE(amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX90A, source, 6));
}

TEST_P(DsAddressEncodedFields, FormatOnlyDoesNotGuaranteeAddressOperand) {
  const Target &target = GetParam();
  // llvm-mc emits DS_NOP in this same family, but it has no ADDR operand.
  const std::array<uint32_t, 2> nop = target.rdna ? std::array<uint32_t, 2>{0xd8500000u, 0}
                                                  : std::array<uint32_t, 2>{0xd8280000u, 0};
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.id);
  ASSERT_TRUE(decoder);
  const DecodeResult decoded = decoder->decode_window(nop);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->mnemonic(), "ds_nop");
  EXPECT_EQ(decoded.value()->num_src_operands(), 0);
  const std::optional<std::array<uint32_t, 2>> result =
      amdgpu::replace_ds_address_field(target.id, nop, 6);
  ASSERT_TRUE(result);
  EXPECT_EQ((*result)[0], nop[0]);
  EXPECT_EQ((*result)[1], 6u);
}

TEST(DsAddressEncodedFieldsCdna5, TwoAddressFieldIsMechanicallyReplaced) {
  // llvm-mc gfx1250 ds_load_2addr_b64; field encoding is not FI admission.
  const Fixture &fixture = kRdna[2];
  const std::optional<std::array<uint32_t, 2>> result =
      amdgpu::replace_ds_address_field(ROCJITSU_CODE_TARGET_GFX1250, fixture.source, 6);
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, fixture.expected);
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1250);
  ASSERT_TRUE(decoder);
  const DecodeResult decoded = decoder->decode_window(*result);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->mnemonic(), "ds_load_2addr_b64");
  ASSERT_NE(decoded.value()->src_operand(0), nullptr);
  EXPECT_EQ(decoded.value()->src_operand(0)->encoding_value(), 6);
}

INSTANTIATE_TEST_SUITE_P(Targets, DsAddressEncodedFields, ::testing::ValuesIn(kTargets),
                         [](const ::testing::TestParamInfo<Target> &info) {
                           return info.param.name;
                         });
} // namespace
} // namespace rocjitsu

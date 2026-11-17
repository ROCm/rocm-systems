// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ds_address_test_support.h"

#include "rocjitsu/isa/arch/amdgpu/cdna5/encoded_fields.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
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

class Cdna5EncodedFields : public ::testing::TestWithParam<rj_code_target_id_t> {};

TEST_P(Cdna5EncodedFields, PreservesStoreAndLoadAndDecodesReplacement) {
  // llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250/gfx1251 --show-encoding:
  // ds_store_b32 v2, v3 offset:260 / ds_load_b32 v5, v2 offset:260.
  struct Fixture {
    std::array<uint32_t, 2> source;
    std::array<uint32_t, 2> expected;
    const char *mnemonic;
    bool is_store;
  };
  const std::array<Fixture, 2> fixtures = {{
      {.source = {0xd8340104u, 0x00000302u},
       .expected = {0xd8340104u, 0x00000306u},
       .mnemonic = "ds_store_b32",
       .is_store = true},
      {.source = {0xd8d80104u, 0x05000002u},
       .expected = {0xd8d80104u, 0x05000006u},
       .mnemonic = "ds_load_b32",
       .is_store = false},
  }};
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), GetParam());
  ASSERT_TRUE(decoder);
  for (const Fixture &fixture : fixtures) {
    SCOPED_TRACE(fixture.mnemonic);
    const std::optional<std::array<uint32_t, 2>> replacement =
        cdna5::replace_ds_b32_address(GetParam(), fixture.source, 6);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(*replacement, fixture.expected);
    EXPECT_TRUE(test_support::only_ds_address_changed(fixture.source, *replacement, 6));
    DecodeResult decoded = decoder->decode_window(*replacement);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value()->mnemonic(), fixture.mnemonic);
    ASSERT_NE(decoded.value()->src_operand(0), nullptr);
    EXPECT_EQ(decoded.value()->src_operand(0)->encoding_value(), 6u);
    if (fixture.is_store) {
      ASSERT_NE(decoded.value()->src_operand(1), nullptr);
      EXPECT_EQ(decoded.value()->src_operand(1)->encoding_value(), 3u);
    } else {
      ASSERT_NE(decoded.value()->dst_operand(0), nullptr);
      EXPECT_EQ(decoded.value()->dst_operand(0)->encoding_value(), 5u);
    }
    std::array<uint32_t, 2> corrupted = *replacement;
    corrupted[1] ^= 1u << 8;
    EXPECT_FALSE(test_support::only_ds_address_changed(fixture.source, corrupted, 6));
    corrupted = *replacement;
    corrupted[0] ^= 1u;
    EXPECT_FALSE(test_support::only_ds_address_changed(fixture.source, corrupted, 6));
  }
}

TEST_P(Cdna5EncodedFields, RejectsUnsupportedTargetFormExtentAndRegister) {
  const std::array<uint32_t, 2> source = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2});
  EXPECT_FALSE(cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1100, source, 6));
  EXPECT_FALSE(cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1201, source, 6));
  EXPECT_FALSE(cdna5::replace_ds_b32_address(GetParam(), std::span(source).first(1), 6));
  const std::array<uint32_t, 3> oversized = {source[0], source[1], 0};
  EXPECT_FALSE(cdna5::replace_ds_b32_address(GetParam(), oversized, 6));
  EXPECT_FALSE(cdna5::replace_ds_b32_address(GetParam(), source, 256));
  for (uint16_t opcode : {cdna5::kDsLoad2addrB32Vds, cdna5::kDsLoad2addrStride64B32Vds,
                          cdna5::kDsStore2addrB32Vds, cdna5::kDsStoreB64Vds}) {
    const std::array<uint32_t, 2> other = cdna5::build_vds(opcode, {.addr = 2});
    EXPECT_FALSE(cdna5::replace_ds_b32_address(GetParam(), other, 6));
  }
  const std::array<uint32_t, 2> invalid = {0, source[1]};
  EXPECT_FALSE(cdna5::replace_ds_b32_address(GetParam(), invalid, 6));
  for (uint32_t value : {0u, 255u}) {
    const std::optional<std::array<uint32_t, 2>> replacement =
        cdna5::replace_ds_b32_address(GetParam(), source, value);
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(test_support::only_ds_address_changed(source, *replacement, value));
  }
}

TEST_P(Cdna5EncodedFields, PreservesPaddingAndUnusedFields) {
  constexpr std::array<uint32_t, 2> store = {0xd8370104u, 0xa5b60302u};
  constexpr std::array<uint32_t, 2> load = {0xd8db0104u, 0xa5b60302u};
  for (const std::array<uint32_t, 2> &source : {store, load}) {
    const std::optional<std::array<uint32_t, 2>> replacement =
        cdna5::replace_ds_b32_address(GetParam(), source, 6);
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(test_support::only_ds_address_changed(source, *replacement, 6));
    EXPECT_EQ((*replacement)[1], 0xa5b60306u);
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, Cdna5EncodedFields,
                         ::testing::Values(ROCJITSU_CODE_TARGET_GFX1250,
                                           ROCJITSU_CODE_TARGET_GFX1251));

} // namespace
} // namespace rocjitsu

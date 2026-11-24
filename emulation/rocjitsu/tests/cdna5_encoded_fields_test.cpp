// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

bool only_address_changed(const std::array<uint32_t, 2> &source,
                          const std::array<uint32_t, 2> &replacement, uint32_t address) {
  return source[0] == replacement[0] &&
         (source[1] & 0xffffff00u) == (replacement[1] & 0xffffff00u) &&
         (replacement[1] & 0xffu) == address;
}

TEST(Cdna5EncodedFields, PreservesStoreAndLoadAndDecodesReplacement) {
  // llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 --show-encoding:
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
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1250);
  ASSERT_TRUE(decoder);
  for (const Fixture &fixture : fixtures) {
    SCOPED_TRACE(fixture.mnemonic);
    const std::optional<std::array<uint32_t, 2>> replacement =
        cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1250, fixture.source, 6);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(*replacement, fixture.expected);
    EXPECT_TRUE(only_address_changed(fixture.source, *replacement, 6));
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
    EXPECT_FALSE(only_address_changed(fixture.source, corrupted, 6));
    corrupted = *replacement;
    corrupted[0] ^= 1u;
    EXPECT_FALSE(only_address_changed(fixture.source, corrupted, 6));
  }
}

TEST(Cdna5EncodedFields, RejectsUnsupportedTargetFormExtentAndRegister) {
  const std::array<uint32_t, 2> source = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2});
  EXPECT_FALSE(cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1251, source, 6));
  EXPECT_FALSE(cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1201, source, 6));
  EXPECT_FALSE(
      cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1250, std::span(source).first(1), 6));
  const std::array<uint32_t, 3> oversized = {source[0], source[1], 0};
  EXPECT_FALSE(cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1250, oversized, 6));
  EXPECT_FALSE(cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1250, source, 256));
  for (uint16_t opcode : {cdna5::kDsLoad2addrB32Vds, cdna5::kDsLoad2addrStride64B32Vds,
                          cdna5::kDsStore2addrB32Vds, cdna5::kDsStoreB64Vds}) {
    const std::array<uint32_t, 2> other = cdna5::build_vds(opcode, {.addr = 2});
    EXPECT_FALSE(cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1250, other, 6));
  }
  const std::array<uint32_t, 2> invalid = {0, source[1]};
  EXPECT_FALSE(cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1250, invalid, 6));
  for (uint32_t value : {0u, 255u}) {
    const std::optional<std::array<uint32_t, 2>> replacement =
        cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1250, source, value);
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(only_address_changed(source, *replacement, value));
  }
}

TEST(Cdna5EncodedFields, PreservesPaddingAndUnusedFields) {
  constexpr std::array<uint32_t, 2> store = {0xd8370104u, 0xa5b60302u};
  constexpr std::array<uint32_t, 2> load = {0xd8db0104u, 0xa5b60302u};
  for (const std::array<uint32_t, 2> &source : {store, load}) {
    const std::optional<std::array<uint32_t, 2>> replacement =
        cdna5::replace_ds_b32_address(ROCJITSU_CODE_TARGET_GFX1250, source, 6);
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(only_address_changed(source, *replacement, 6));
    EXPECT_EQ((*replacement)[1], 0xa5b60306u);
  }
}

} // namespace
} // namespace rocjitsu

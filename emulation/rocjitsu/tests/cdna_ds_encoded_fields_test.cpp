// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_ds_encoded_fields.h"
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

struct StoreCase {
  std::array<uint32_t, 2> source;
  std::array<uint32_t, 2> expected;
  std::string_view mnemonic;
  uint16_t data_index;
  uint8_t data_width;
};

TEST(CdnaDsEncodedFields, PreservesConcreteStoreFieldsAndDecodedOperands) {
  // Words are independent llvm-mc gfx942/gfx950 witnesses, including the ACC
  // data selector. The address operand remains an ordinary encoded VGPR.
  constexpr std::array<StoreCase, 4> cases = {{
      {.source = {0xd81a0004u, 0x00000302u},
       .expected = {0xd81a0004u, 0x00000306u},
       .mnemonic = "ds_write_b32",
       .data_index = 3,
       .data_width = 1},
      {.source = {0xda1a0004u, 0x00000302u},
       .expected = {0xda1a0004u, 0x00000306u},
       .mnemonic = "ds_write_b32",
       .data_index = 259,
       .data_width = 1},
      {.source = {0xd9be0010u, 0x00000402u},
       .expected = {0xd9be0010u, 0x00000406u},
       .mnemonic = "ds_write_b128",
       .data_index = 4,
       .data_width = 4},
      {.source = {0xdbbe0010u, 0x00000402u},
       .expected = {0xdbbe0010u, 0x00000406u},
       .mnemonic = "ds_write_b128",
       .data_index = 260,
       .data_width = 4},
  }};
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_TARGET_GFX950}) {
    std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
    ASSERT_TRUE(decoder);
    for (const StoreCase &fixture : cases) {
      SCOPED_TRACE(static_cast<int>(target));
      SCOPED_TRACE(fixture.mnemonic);
      const std::optional<std::array<uint32_t, 2>> replacement =
          amdgpu::replace_cdna_ds_write_address(target, fixture.source, 6);
      ASSERT_TRUE(replacement);
      EXPECT_EQ(*replacement, fixture.expected);
      const DecodeResult decoded = decoder->decode_window(*replacement);
      ASSERT_TRUE(decoded.succeeded());
      EXPECT_EQ(decoded.value()->mnemonic(), fixture.mnemonic);
      ASSERT_NE(decoded.value()->src_operand(0), nullptr);
      ASSERT_NE(decoded.value()->src_operand(1), nullptr);
      const std::optional<RegisterRef> address = decoded.value()->src_operand(0)->decoded_vgpr();
      const std::optional<RegisterRef> data = decoded.value()->src_operand(1)->decoded_vgpr();
      ASSERT_TRUE(address);
      ASSERT_TRUE(data);
      EXPECT_EQ(address->index, 6u);
      EXPECT_EQ(address->width, 1u);
      EXPECT_EQ(data->index, fixture.data_index);
      EXPECT_EQ(data->width, fixture.data_width);
    }
  }
}

TEST(CdnaDsEncodedFields, PreservesGdsAccAndEveryUnselectedBit) {
  // Nonzero GDS here is a bit-preservation witness, not an assertion that the
  // instruction executes legally on either target.
  constexpr std::array<uint32_t, 2> source = {0xdbbfef01u, 0xa5b60402u};
  constexpr std::array<uint32_t, 2> expected = {0xdbbfef01u, 0xa5b604feu};
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_TARGET_GFX950}) {
    const std::optional<std::array<uint32_t, 2>> replacement =
        amdgpu::replace_cdna_ds_write_address(target, source, 254);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(*replacement, expected);
    std::array<uint32_t, 2> corrupted = *replacement;
    corrupted[1] ^= 1u << 8;
    EXPECT_NE(corrupted, expected);
  }
}

TEST(CdnaDsEncodedFields, RejectsOtherTargetsFormsExtentsAndOverflow) {
  constexpr std::array<uint32_t, 2> source = {0xd81a0004u, 0x00000302u};
  constexpr std::array<uint32_t, 2> load = {0xd86c0004u, 0x05000002u};
  constexpr std::array<uint32_t, 2> other_encoding = {0x981a0004u, 0x00000302u};
  constexpr std::array<uint32_t, 3> oversized = {0xd81a0004u, 0x00000302u, 0};
  EXPECT_FALSE(amdgpu::replace_cdna_ds_write_address(ROCJITSU_CODE_TARGET_GFX1201, source, 6));
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_TARGET_GFX950}) {
    EXPECT_FALSE(amdgpu::replace_cdna_ds_write_address(target, std::span<const uint32_t>(), 6));
    EXPECT_FALSE(amdgpu::replace_cdna_ds_write_address(target, std::span(source).first(1), 6));
    EXPECT_FALSE(amdgpu::replace_cdna_ds_write_address(target, oversized, 6));
    EXPECT_FALSE(amdgpu::replace_cdna_ds_write_address(target, source, 256));
    EXPECT_FALSE(amdgpu::replace_cdna_ds_write_address(target, source, UINT32_MAX));
    EXPECT_FALSE(amdgpu::replace_cdna_ds_write_address(target, load, 6));
    EXPECT_FALSE(amdgpu::replace_cdna_ds_write_address(target, other_encoding, 6));
    for (uint32_t value : {0u, 255u}) {
      const std::optional<std::array<uint32_t, 2>> replacement =
          amdgpu::replace_cdna_ds_write_address(target, source, value);
      ASSERT_TRUE(replacement);
      EXPECT_EQ((*replacement)[0], 0xd81a0004u);
      EXPECT_EQ((*replacement)[1], 0x00000300u | value);
    }
  }
}

} // namespace
} // namespace rocjitsu

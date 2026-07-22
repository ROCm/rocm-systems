// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/gfx1250_instrumentation_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace rocjitsu {
namespace {

TEST(Gfx1250InstructionBuilder, BuildSWaitXcnt0) {
  const auto word = build_gfx1250_s_wait_xcnt0(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0xBFC50000u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "s_wait_xcnt");
  EXPECT_EQ(inst->disassemble(), "s_wait_xcnt 0");

  EXPECT_FALSE(build_gfx1250_s_wait_xcnt0(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(Gfx1250InstructionBuilder, BuildSSetVgprMsb) {
  EXPECT_EQ(build_gfx1250_s_set_vgpr_msb(/*mode=*/0x00, ROCJITSU_CODE_ARCH_GFX1250), 0xBF860000u);
  EXPECT_EQ(build_gfx1250_s_set_vgpr_msb(/*mode=*/0x40, ROCJITSU_CODE_ARCH_GFX1250), 0xBF860040u);
  EXPECT_EQ(build_gfx1250_s_set_vgpr_msb(/*mode=*/0x100, ROCJITSU_CODE_ARCH_GFX1250), 0xBF860100u);
  EXPECT_FALSE(build_gfx1250_s_set_vgpr_msb(/*mode=*/0x40, ROCJITSU_CODE_ARCH_RDNA4));

  EXPECT_EQ(build_gfx1250_s_set_vgpr_msb_transition(
                /*previous_mode=*/0x44, /*new_mode=*/0x08, ROCJITSU_CODE_ARCH_GFX1250),
            0xBF864408u);
  EXPECT_FALSE(build_gfx1250_s_set_vgpr_msb_transition(
      /*previous_mode=*/0x44, /*new_mode=*/0x08, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(Gfx1250InstructionBuilder, BuildSCallI64) {
  const auto word =
      build_gfx1250_s_call_i64(/*sdst=*/24, /*simm16=*/-3, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0xBA18FFFDu);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "s_call_i64");
  EXPECT_EQ(inst->branch_offset_bytes(), -12);

  EXPECT_FALSE(build_gfx1250_s_call_i64(/*sdst=*/23, 0, ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_FALSE(build_gfx1250_s_call_i64(/*sdst=*/24, 0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(Gfx1250InstructionBuilder, BuildVCmpNeU16Vcc) {
  const auto word = build_gfx1250_v_cmp_ne_u16_vcc(vector_source_vgpr(1), /*vsrc1=*/2,
                                                   ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x7C7A0501u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "v_cmp_ne_u16_e32");

  EXPECT_FALSE(
      build_gfx1250_v_cmp_ne_u16_vcc(/*src0=*/512, /*vsrc1=*/2, ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_FALSE(build_gfx1250_v_cmp_ne_u16_vcc(vector_source_vgpr(1), /*vsrc1=*/256,
                                              ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_FALSE(
      build_gfx1250_v_cmp_ne_u16_vcc(vector_source_vgpr(1), /*vsrc1=*/2, ROCJITSU_CODE_ARCH_CDNA4));
}

} // namespace
} // namespace rocjitsu

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/rdna4_instrumentation_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>

namespace rocjitsu {
namespace {

TEST(InstructionBuilder, BuildAddressFreeScratchB32) {
  const auto store = build_address_free_scratch_store_b32(
      /*vsrc=*/5, /*byte_offset=*/16, ROCJITSU_CODE_ARCH_RDNA4);
  const auto load = build_address_free_scratch_load_b32(
      /*vdst=*/5, /*byte_offset=*/16, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(store);
  ASSERT_TRUE(load);
  EXPECT_EQ(*store, (std::array<uint32_t, 3>{0xed06807cu, 0x02800000u, 0x00001000u}));
  EXPECT_EQ(*load, (std::array<uint32_t, 3>{0xed05007cu, 0x00000005u, 0x00001000u}));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> store_inst(decoder->decode(store->data()));
  std::unique_ptr<Instruction> load_inst(decoder->decode(load->data()));
  ASSERT_NE(store_inst, nullptr);
  ASSERT_NE(load_inst, nullptr);
  EXPECT_EQ(store_inst->mnemonic(), "scratch_store_b32");
  EXPECT_EQ(load_inst->mnemonic(), "scratch_load_b32");
  EXPECT_EQ(store_inst->size(), 12u);
  EXPECT_EQ(load_inst->size(), 12u);
}

TEST(InstructionBuilder, BuildGfx1250AddressFreeScratchB32) {
  const auto store = build_address_free_scratch_store_b32(
      /*vsrc=*/5, /*byte_offset=*/16, ROCJITSU_CODE_ARCH_GFX1250);
  const auto load = build_address_free_scratch_load_b32(
      /*vdst=*/5, /*byte_offset=*/16, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(store);
  ASSERT_TRUE(load);
  EXPECT_EQ(*store, (std::array<uint32_t, 3>{0xed06807cu, 0x02800000u, 0x00001000u}));
  EXPECT_EQ(*load, (std::array<uint32_t, 3>{0xed05007cu, 0x00000005u, 0x00001000u}));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> store_inst(decoder->decode(store->data()));
  std::unique_ptr<Instruction> load_inst(decoder->decode(load->data()));
  ASSERT_NE(store_inst, nullptr);
  ASSERT_NE(load_inst, nullptr);
  EXPECT_EQ(store_inst->mnemonic(), "scratch_store_b32");
  EXPECT_EQ(load_inst->mnemonic(), "scratch_load_b32");
  EXPECT_EQ(store_inst->size(), 12u);
  EXPECT_EQ(load_inst->size(), 12u);
}

TEST(InstructionBuilder, AddressFreeScratchOffsetBoundary) {
  const auto store = build_address_free_scratch_store_b32(
      /*vsrc=*/255, kMaxAddressFreeScratchDwordOffset, ROCJITSU_CODE_ARCH_RDNA4);
  const auto load = build_address_free_scratch_load_b32(
      /*vdst=*/255, kMaxAddressFreeScratchDwordOffset, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(store);
  ASSERT_TRUE(load);
  EXPECT_EQ(*store, (std::array<uint32_t, 3>{0xed06807cu, 0x7f800000u, 0x7ffffc00u}));
  EXPECT_EQ(*load, (std::array<uint32_t, 3>{0xed05007cu, 0x000000ffu, 0x7ffffc00u}));

  EXPECT_FALSE(build_address_free_scratch_store_b32(0, kMaxAddressFreeScratchPrivateBytes,
                                                    ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_address_free_scratch_load_b32(0, kMaxAddressFreeScratchPrivateBytes,
                                                   ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_address_free_scratch_store_b32(0, 2, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_address_free_scratch_load_b32(0, 2, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_address_free_scratch_store_b32(0, 0, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildSplitScratchWaits) {
  const auto store_wait = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto store_ds_wait = build_s_wait_storecnt_dscnt0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto load_wait = build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(store_wait);
  ASSERT_TRUE(store_ds_wait);
  ASSERT_TRUE(load_wait);
  EXPECT_EQ(*store_wait, 0xbfc10000u);
  EXPECT_EQ(*store_ds_wait, 0xbfc90000u);
  EXPECT_EQ(*load_wait, 0xbfc00000u);
  EXPECT_FALSE(build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_s_wait_storecnt_dscnt0(ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildScalarDestinationDependencyWaits) {
  const auto salu = build_s_wait_alu_sa_sdst0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto valu = build_s_wait_alu_va_sdst0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto gfx1250_salu = build_s_wait_alu_sa_sdst0(ROCJITSU_CODE_ARCH_GFX1250);
  const auto gfx1250_valu = build_s_wait_alu_va_sdst0(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(salu);
  ASSERT_TRUE(valu);
  ASSERT_TRUE(gfx1250_salu);
  ASSERT_TRUE(gfx1250_valu);
  EXPECT_EQ(*salu, 0xbf88ff9eu);
  // va_sdst=0 with every other depctr field at its no-wait maximum.
  EXPECT_EQ(*valu, 0xbf88f19fu);
  EXPECT_EQ(*gfx1250_salu, 0xbf88ff9eu);
  EXPECT_EQ(*gfx1250_valu, 0xbf88f19fu);
  EXPECT_FALSE(build_s_wait_alu_sa_sdst0(ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_s_wait_alu_va_sdst0(ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildVLshrrevB32E32) {
  const auto word = build_v_lshrrev_b32_e32(/*vdst=*/10, scalar_positive_inline_u32(2),
                                            /*vsrc1=*/3, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x32000000u | (10u << 17) | (3u << 9) | scalar_positive_inline_u32(2));

  EXPECT_FALSE(build_v_lshrrev_b32_e32(/*vdst=*/256, scalar_positive_inline_u32(2), /*vsrc1=*/3,
                                       ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_v_lshrrev_b32_e32(/*vdst=*/10, /*src0=*/512, /*vsrc1=*/3, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_lshrrev_b32_e32(/*vdst=*/10, scalar_positive_inline_u32(2), /*vsrc1=*/256,
                                       ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(InstructionBuilder, BuildVLshlrevB32E32) {
  const auto word = build_v_lshlrev_b32_e32(/*vdst=*/10, scalar_positive_inline_u32(16),
                                            /*vsrc1=*/10, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x30141490u);

  EXPECT_FALSE(build_v_lshlrev_b32_e32(/*vdst=*/256, scalar_positive_inline_u32(16),
                                       /*vsrc1=*/10, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_v_lshlrev_b32_e32(/*vdst=*/10, /*src0=*/512, /*vsrc1=*/10, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_lshlrev_b32_e32(/*vdst=*/10, scalar_positive_inline_u32(16),
                                       /*vsrc1=*/256, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_lshlrev_b32_e32(/*vdst=*/10, scalar_positive_inline_u32(16),
                                       /*vsrc1=*/10, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildVReadfirstlaneB32) {
  const auto word = build_v_readfirstlane_b32(/*sdst=*/20, /*vsrc=*/8, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x7E280508u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "v_readfirstlane_b32_e32");

  EXPECT_FALSE(build_v_readfirstlane_b32(/*sdst=*/128, /*vsrc=*/8, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_readfirstlane_b32(/*sdst=*/20, /*vsrc=*/256, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_readfirstlane_b32(/*sdst=*/20, /*vsrc=*/8, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildRdna4FixedLaneScalarTransfers) {
  const auto write =
      build_rdna4_v_writelane_b32(/*vdst=*/40, /*ssrc=*/18, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto read =
      build_rdna4_v_readlane_b32(/*sdst=*/3, /*vsrc=*/40, /*lane=*/1, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(write);
  ASSERT_TRUE(read);
  EXPECT_EQ(*write, (std::array<uint32_t, 2>{0xD7610028u, 0x02010012u}));
  EXPECT_EQ(*read, (std::array<uint32_t, 2>{0xD7600003u, 0x02010328u}));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> write_inst(decoder->decode(write->data()));
  std::unique_ptr<Instruction> read_inst(decoder->decode(read->data()));
  ASSERT_NE(write_inst, nullptr);
  ASSERT_NE(read_inst, nullptr);
  EXPECT_EQ(std::string_view(write_inst->mnemonic()), "v_writelane_b32");
  EXPECT_EQ(std::string_view(read_inst->mnemonic()), "v_readlane_b32");

  EXPECT_FALSE(
      build_rdna4_v_writelane_b32(/*vdst=*/256, /*ssrc=*/18, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_rdna4_v_writelane_b32(/*vdst=*/40, /*ssrc=*/106, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_rdna4_v_writelane_b32(/*vdst=*/40, /*ssrc=*/18, /*lane=*/64, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_rdna4_v_readlane_b32(/*sdst=*/106, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_rdna4_v_readlane_b32(/*sdst=*/3, /*vsrc=*/256, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_rdna4_v_readlane_b32(/*sdst=*/3, /*vsrc=*/40, /*lane=*/64, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_rdna4_v_writelane_b32(/*vdst=*/40, /*ssrc=*/18, /*lane=*/0,
                                           ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_FALSE(
      build_rdna4_v_readlane_b32(/*sdst=*/3, /*vsrc=*/40, /*lane=*/1, ROCJITSU_CODE_ARCH_GFX1250));
}

TEST(InstructionBuilder, BuildSGetregB32) {
  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  EXPECT_EQ(*hwreg, 0x4817u);

  const auto word = build_s_getreg_b32(/*sdst=*/20, *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0xB8944817u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "s_getreg_b32");

  EXPECT_FALSE(build_hwreg_imm(/*reg_id=*/64, /*offset=*/0, /*size_bits=*/10));
  EXPECT_FALSE(build_hwreg_imm(/*reg_id=*/23, /*offset=*/32, /*size_bits=*/10));
  EXPECT_FALSE(build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/0));
  EXPECT_FALSE(build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/33));
  EXPECT_FALSE(build_s_getreg_b32(/*sdst=*/128, *hwreg, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(InstructionBuilder, BuildVMbcntLaneIdSequence) {
  const auto low = build_v_mbcnt_lo_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(low);
  EXPECT_EQ((*low)[0], 0xD71F000Du);
  // Pinned against LLVM's gfx1201 assembler. Reserved bit 25 must remain zero;
  // setting it produces an illegal instruction on RDNA4.
  EXPECT_EQ((*low)[1], 0x000100C1u);

  const auto high = build_v_mbcnt_hi_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, vector_source_vgpr(13), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(high);
  EXPECT_EQ((*high)[0], 0xD720000Du);
  EXPECT_EQ((*high)[1], 0x00021AC1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> low_inst(decoder->decode(low->data()));
  ASSERT_NE(low_inst, nullptr);
  EXPECT_EQ(std::string_view(low_inst->mnemonic()), "v_mbcnt_lo_u32_b32");
  std::unique_ptr<Instruction> high_inst(decoder->decode(high->data()));
  ASSERT_NE(high_inst, nullptr);
  EXPECT_EQ(std::string_view(high_inst->mnemonic()), "v_mbcnt_hi_u32_b32");

  EXPECT_FALSE(build_v_mbcnt_lo_u32_b32(/*vdst=*/256, /*src0=*/0xC1, scalar_positive_inline_u32(0),
                                        ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_mbcnt_lo_u32_b32(/*vdst=*/13, /*src0=*/512, scalar_positive_inline_u32(0),
                                        ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_v_mbcnt_hi_u32_b32(/*vdst=*/13, /*src0=*/0xC1, /*src1=*/512, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_mbcnt_hi_u32_b32(/*vdst=*/13, /*src0=*/0xC1, vector_source_vgpr(13),
                                        ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildVCmpEqU32E32Vcc) {
  const auto word = build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), /*vsrc1=*/13,
                                               ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x7C941A80u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "v_cmp_eq_u32_e32");

  EXPECT_FALSE(build_v_cmp_eq_u32_e32_vcc(/*src0=*/512, /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), /*vsrc1=*/256,
                                          ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), /*vsrc1=*/13,
                                          ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildVAddNcU32E32) {
  const auto word = build_v_add_nc_u32_e32(/*vdst=*/13, vector_source_vgpr(13), /*vsrc1=*/14,
                                           ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x4A1A1D0Du);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "v_add_nc_u32_e32");

  EXPECT_FALSE(build_v_add_nc_u32_e32(/*vdst=*/256, vector_source_vgpr(13), /*vsrc1=*/14,
                                      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_v_add_nc_u32_e32(/*vdst=*/13, /*src0=*/512, /*vsrc1=*/14, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_add_nc_u32_e32(/*vdst=*/13, vector_source_vgpr(13), /*vsrc1=*/256,
                                      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_add_nc_u32_e32(/*vdst=*/13, vector_source_vgpr(13), /*vsrc1=*/14,
                                      ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildVAndB32E32) {
  const auto word =
      build_v_and_b32_e32(/*vdst=*/3, vector_source_vgpr(4), /*vsrc1=*/5, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x36060B04u);

  const auto literal = build_v_and_b32_e32_literal(/*vdst=*/3, /*literal=*/0x1ff8, /*vsrc1=*/4,
                                                   ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(literal);
  EXPECT_EQ((*literal)[0], 0x360608FFu);
  EXPECT_EQ((*literal)[1], 0x00001FF8u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "v_and_b32_e32");
  std::unique_ptr<Instruction> literal_inst(decoder->decode(literal->data()));
  ASSERT_NE(literal_inst, nullptr);
  EXPECT_EQ(std::string_view(literal_inst->mnemonic()), "v_and_b32_e32");

  EXPECT_FALSE(build_v_and_b32_e32(/*vdst=*/256, vector_source_vgpr(4), /*vsrc1=*/5,
                                   ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_v_and_b32_e32(/*vdst=*/3, /*src0=*/512, /*vsrc1=*/5, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_and_b32_e32(/*vdst=*/3, vector_source_vgpr(4), /*vsrc1=*/256,
                                   ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_and_b32_e32(/*vdst=*/3, vector_source_vgpr(4), /*vsrc1=*/5,
                                   ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_v_and_b32_e32_literal(/*vdst=*/256, /*literal=*/0x1ff8, /*vsrc1=*/4,
                                           ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_and_b32_e32_literal(/*vdst=*/3, /*literal=*/0x1ff8, /*vsrc1=*/256,
                                           ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_and_b32_e32_literal(/*vdst=*/3, /*literal=*/0x1ff8, /*vsrc1=*/4,
                                           ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildVCmpGtU32E32Vcc) {
  const auto word = build_v_cmp_gt_u32_e32_vcc(scalar_positive_inline_u32(4), /*vsrc1=*/10,
                                               ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x7C981484u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "v_cmp_gt_u32_e32");

  EXPECT_FALSE(build_v_cmp_gt_u32_e32_vcc(/*src0=*/512, /*vsrc1=*/10, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_cmp_gt_u32_e32_vcc(scalar_positive_inline_u32(4), /*vsrc1=*/256,
                                          ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_cmp_gt_u32_e32_vcc(scalar_positive_inline_u32(4), /*vsrc1=*/10,
                                          ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildVCmpNeU32E32Vcc) {
  const auto word =
      build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(1), /*vsrc1=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x7C9A0501u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "v_cmp_ne_u32_e32");

  EXPECT_FALSE(build_v_cmp_ne_u32_e32_vcc(/*src0=*/512, /*vsrc1=*/2, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(1), /*vsrc1=*/256, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(1), /*vsrc1=*/2, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildVCmpNeU16E32Vcc) {
  const auto word =
      build_v_cmp_ne_u16_e32_vcc(vector_source_vgpr(1), /*vsrc1=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0x7C7A0501u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "v_cmp_ne_u16_e32");

  EXPECT_FALSE(build_v_cmp_ne_u16_e32_vcc(/*src0=*/512, /*vsrc1=*/2, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_v_cmp_ne_u16_e32_vcc(vector_source_vgpr(1), /*vsrc1=*/256, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_v_cmp_ne_u16_e32_vcc(vector_source_vgpr(1), /*vsrc1=*/2, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildExecNarrowingScalarOps) {
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/106, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_exec);
  EXPECT_EQ(*save_exec, 0xBE9E216Au);

  const auto restore_exec = build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(restore_exec);
  EXPECT_EQ(*restore_exec, 0xBEFE011Eu);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> save_inst(decoder->decode(&*save_exec));
  ASSERT_NE(save_inst, nullptr);
  EXPECT_EQ(std::string_view(save_inst->mnemonic()), "s_and_saveexec_b64");
  std::unique_ptr<Instruction> restore_inst(decoder->decode(&*restore_exec));
  ASSERT_NE(restore_inst, nullptr);
  EXPECT_EQ(std::string_view(restore_inst->mnemonic()), "s_mov_b64");

  EXPECT_FALSE(build_s_and_saveexec_b64(/*sdst=*/127, /*ssrc0=*/106, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/256, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/106, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_s_mov_b64(/*sdst=*/127, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/256, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildWavePartitionMaskRemoval) {
  const auto remaining = build_s_xor_b64(
      /*sdst=*/30, /*ssrc0=*/32, /*ssrc1=*/34, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(remaining);
  EXPECT_EQ(*remaining, 0x8D9E2220u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*remaining));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "s_xor_b64");

  EXPECT_FALSE(build_s_xor_b64(/*sdst=*/127, /*ssrc0=*/32, /*ssrc1=*/34, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_s_xor_b64(/*sdst=*/30, /*ssrc0=*/255, /*ssrc1=*/34, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_s_xor_b64(/*sdst=*/30, /*ssrc0=*/32, /*ssrc1=*/255, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_s_xor_b64(/*sdst=*/30, /*ssrc0=*/32, /*ssrc1=*/34, ROCJITSU_CODE_ARCH_CDNA4));

  const auto repeat = build_s_cbranch_execnz(/*offset_dwords=*/-4, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(repeat);
  EXPECT_EQ(*repeat, 0xBFA6FFFCu);
  std::unique_ptr<Instruction> branch_inst(decoder->decode(&*repeat));
  ASSERT_NE(branch_inst, nullptr);
  EXPECT_EQ(std::string_view(branch_inst->mnemonic()), "s_cbranch_execnz");
  EXPECT_FALSE(build_s_cbranch_execnz(/*offset_dwords=*/-4, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildSccSnapshotAndRestoreOps) {
  const auto save_scc = build_rdna4_s_cselect_b32(
      /*sdst=*/20, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_scc);
  EXPECT_EQ(*save_scc, 0x98148081u);

  const auto restore_scc = build_rdna4_s_cmp_lg_u32(
      /*ssrc0=*/20, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(restore_scc);
  EXPECT_EQ(*restore_scc, 0xBF078014u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> save_inst(decoder->decode(&*save_scc));
  ASSERT_NE(save_inst, nullptr);
  EXPECT_EQ(std::string_view(save_inst->mnemonic()), "s_cselect_b32");
  std::unique_ptr<Instruction> restore_inst(decoder->decode(&*restore_scc));
  ASSERT_NE(restore_inst, nullptr);
  EXPECT_EQ(std::string_view(restore_inst->mnemonic()), "s_cmp_lg_u32");

  EXPECT_FALSE(build_rdna4_s_cselect_b32(
      /*sdst=*/128, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_rdna4_s_cselect_b32(
      /*sdst=*/20, /*ssrc0=*/256, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_rdna4_s_cselect_b32(
      /*sdst=*/20, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_rdna4_s_cmp_lg_u32(/*ssrc0=*/256, scalar_positive_inline_u32(0),
                                        ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_rdna4_s_cmp_lg_u32(/*ssrc0=*/20, scalar_positive_inline_u32(0),
                                        ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildSCbranchVccz) {
  const auto word = build_s_cbranch_vccz(/*offset_dwords=*/3, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0xBFA30003u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "s_cbranch_vccz");

  const auto backwards = build_s_cbranch_vccz(/*offset_dwords=*/-1, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(backwards);
  EXPECT_EQ(*backwards, 0xBFA3FFFFu);
  EXPECT_FALSE(build_s_cbranch_vccz(/*offset_dwords=*/3, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildSCbranchVccnz) {
  const auto word = build_s_cbranch_vccnz(/*offset_dwords=*/3, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(word);
  EXPECT_EQ(*word, 0xBFA40003u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(&*word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "s_cbranch_vccnz");

  const auto backwards = build_s_cbranch_vccnz(/*offset_dwords=*/-7, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(backwards);
  EXPECT_EQ(*backwards, 0xBFA4FFF9u);
  EXPECT_FALSE(build_s_cbranch_vccnz(/*offset_dwords=*/3, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildRdna4WaveUniformControlFlow) {
  const auto compare = build_s_cmp_eq_u32(/*ssrc0=*/126, /*ssrc1=*/42, ROCJITSU_CODE_ARCH_RDNA4);
  const auto forward = build_s_cbranch_scc0(/*offset_dwords=*/17, ROCJITSU_CODE_ARCH_RDNA4);
  const auto backward = build_s_cbranch_scc0(/*offset_dwords=*/-3, ROCJITSU_CODE_ARCH_RDNA4);
  const auto continue_loop = build_s_cbranch_scc1(/*offset_dwords=*/-23, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(compare);
  ASSERT_TRUE(forward);
  ASSERT_TRUE(backward);
  ASSERT_TRUE(continue_loop);
  EXPECT_EQ(*compare, pack_sopc(/*s_cmp_eq_u32=*/6, /*ssrc0=*/126, /*ssrc1=*/42));
  EXPECT_EQ(*forward, pack_sopp(/*s_cbranch_scc0=*/33, /*simm16=*/17));
  EXPECT_EQ(*backward, pack_sopp(/*s_cbranch_scc0=*/33, /*simm16=*/0xfffd));
  EXPECT_EQ(*continue_loop, pack_sopp(/*s_cbranch_scc1=*/34, /*simm16=*/0xffe9));
  EXPECT_FALSE(build_s_cmp_eq_u32(/*ssrc0=*/256, /*ssrc1=*/42, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_s_cbranch_scc0(/*offset_dwords=*/1, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_s_cbranch_scc1(/*offset_dwords=*/1, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildGfx1250MoiBarrierRecordRecipeEncodings) {
  constexpr auto kArch = ROCJITSU_CODE_ARCH_GFX1250;

  const auto mbcnt_low = build_v_mbcnt_lo_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, scalar_positive_inline_u32(0), kArch);
  const auto mbcnt_high = build_v_mbcnt_hi_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, vector_source_vgpr(13), kArch);
  const auto cmp_eq =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), /*vsrc1=*/13, kArch);
  const auto cmp_gt =
      build_v_cmp_gt_u32_e32_vcc(scalar_positive_inline_u32(4), /*vsrc1=*/10, kArch);
  const auto mov_literal = build_v_mov_b32_e64_literal(/*vdst=*/13, 0x12345678u, kArch);
  const auto flat_store = build_flat_store_b32_vaddr_vsrc(/*vaddr=*/8, /*vsrc=*/10, kArch);
  const auto store_wait = build_s_wait_storecnt0(kArch);
  const auto load_wait = build_s_wait_loadcnt0(kArch);
  const auto atomic_add = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true, /*scope=*/2, kArch);
  const auto save_exec = build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/106, kArch);
  const auto restore_exec = build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/30, kArch);
  const auto save_scc = build_rdna4_s_cselect_b32(
      /*sdst=*/20, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), kArch);
  const auto restore_scc =
      build_rdna4_s_cmp_lg_u32(/*ssrc0=*/20, scalar_positive_inline_u32(0), kArch);
  const auto capacity_skip = build_s_cbranch_vccz(/*offset_dwords=*/3, kArch);

  ASSERT_TRUE(mbcnt_low);
  ASSERT_TRUE(mbcnt_high);
  ASSERT_TRUE(cmp_eq);
  ASSERT_TRUE(cmp_gt);
  ASSERT_TRUE(mov_literal);
  ASSERT_TRUE(flat_store);
  ASSERT_TRUE(store_wait);
  ASSERT_TRUE(load_wait);
  ASSERT_TRUE(atomic_add);
  ASSERT_TRUE(save_exec);
  ASSERT_TRUE(restore_exec);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(restore_scc);
  ASSERT_TRUE(capacity_skip);
  EXPECT_EQ(*mbcnt_low, (std::array<uint32_t, 2>{0xD71F000Du, 0x000100C1u}));
  EXPECT_EQ(*mbcnt_high, (std::array<uint32_t, 2>{0xD720000Du, 0x00021AC1u}));
  EXPECT_EQ(*cmp_eq, 0x7C941A80u);
  EXPECT_EQ(*cmp_gt, 0x7C981484u);
  EXPECT_EQ(*mov_literal, (std::array<uint32_t, 3>{0xD581000Du, 0x000000FFu, 0x12345678u}));
  EXPECT_EQ(*flat_store, (std::array<uint32_t, 3>{0xEC06807Cu, 0x05000000u, 0x00000008u}));
  EXPECT_EQ(*store_wait, 0xBFC10000u);
  EXPECT_EQ(*load_wait, 0xBFC00000u);
  EXPECT_EQ(*atomic_add, (std::array<uint32_t, 3>{0xEC0D407Cu, 0x0518000Au, 0x00000008u}));
  EXPECT_EQ(*save_exec, 0xBE9E216Au);
  EXPECT_EQ(*restore_exec, 0xBEFE011Eu);
  EXPECT_EQ(*save_scc, 0x98148081u);
  EXPECT_EQ(*restore_scc, 0xBF078014u);
  EXPECT_EQ(*capacity_skip, 0xBFA30003u);

  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> save_exec_inst(decoder->decode(&*save_exec));
  std::unique_ptr<Instruction> mbcnt_inst(decoder->decode(mbcnt_low->data()));
  std::unique_ptr<Instruction> store_inst(decoder->decode(flat_store->data()));
  std::unique_ptr<Instruction> atomic_inst(decoder->decode(atomic_add->data()));
  ASSERT_NE(save_exec_inst, nullptr);
  ASSERT_NE(mbcnt_inst, nullptr);
  ASSERT_NE(store_inst, nullptr);
  ASSERT_NE(atomic_inst, nullptr);
  EXPECT_EQ(std::string_view(save_exec_inst->mnemonic()), "s_and_saveexec_b64");
  EXPECT_EQ(std::string_view(mbcnt_inst->mnemonic()), "v_mbcnt_lo_u32_b32");
  EXPECT_EQ(std::string_view(store_inst->mnemonic()), "flat_store_b32");
  EXPECT_EQ(std::string_view(atomic_inst->mnemonic()), "flat_atomic_add_u32");
}

TEST(InstructionBuilder, BuildFlatStoreB32) {
  const auto words =
      build_flat_store_b32_vaddr_vsrc(/*vaddr=*/8, /*vsrc=*/10, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(words);

  EXPECT_EQ((*words)[0], 0xEC06807Cu);
  EXPECT_EQ((*words)[1], 0x05000000u);
  EXPECT_EQ((*words)[2], 0x00000008u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words->data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "flat_store_b32");

  EXPECT_FALSE(
      build_flat_store_b32_vaddr_vsrc(/*vaddr=*/256, /*vsrc=*/10, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_flat_store_b32_vaddr_vsrc(/*vaddr=*/8, /*vsrc=*/256, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_store_b32_vaddr_vsrc(/*vaddr=*/8, /*vsrc=*/10, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildWideLdsStoreAndLiteralBoundsCompare) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_GFX1250}) {
    const auto store = build_ds_store_b64(/*vaddr=*/3, /*vdata=*/4, /*byte_offset=*/16, arch);
    const auto store_quad = build_ds_store_b128(/*vaddr=*/3, /*vdata=*/4, /*byte_offset=*/16, arch);
    const auto compare = build_v_cmp_gt_u32_e32_vcc_literal(/*literal=*/13080u, /*vsrc1=*/3, arch);
    ASSERT_TRUE(store);
    ASSERT_TRUE(store_quad);
    ASSERT_TRUE(compare);
    EXPECT_EQ(*store, (std::array<uint32_t, 2>{0xD9340010u, 0x00000403u}));
    EXPECT_EQ(*store_quad, (std::array<uint32_t, 2>{0xDB7C0010u, 0x00000403u}));
    EXPECT_EQ(*compare, (std::array<uint32_t, 2>{0x7C9806FFu, 0x00003318u}));

    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> store_inst(decoder->decode(store->data()));
    std::unique_ptr<Instruction> store_quad_inst(decoder->decode(store_quad->data()));
    std::unique_ptr<Instruction> compare_inst(decoder->decode(compare->data()));
    ASSERT_NE(store_inst, nullptr);
    ASSERT_NE(store_quad_inst, nullptr);
    ASSERT_NE(compare_inst, nullptr);
    EXPECT_EQ(std::string_view(store_inst->mnemonic()), "ds_store_b64");
    EXPECT_EQ(std::string_view(store_quad_inst->mnemonic()), "ds_store_b128");
    EXPECT_EQ(std::string_view(compare_inst->mnemonic()), "v_cmp_gt_u32_e32");
    EXPECT_EQ(store_inst->size(), 8u);
    EXPECT_EQ(store_quad_inst->size(), 8u);
    EXPECT_EQ(compare_inst->size(), 8u);
  }

  EXPECT_FALSE(build_ds_store_b64(/*vaddr=*/3, /*vdata=*/255, /*byte_offset=*/0,
                                  ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_FALSE(build_ds_store_b128(/*vaddr=*/3, /*vdata=*/253, /*byte_offset=*/0,
                                   ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_FALSE(
      build_ds_store_b64(/*vaddr=*/3, /*vdata=*/4, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(
      build_ds_store_b128(/*vaddr=*/3, /*vdata=*/4, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(build_v_cmp_gt_u32_e32_vcc_literal(
      /*literal=*/13080u, /*vsrc1=*/3, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildFlatLoadB32) {
  const auto words =
      build_flat_load_b32_vaddr_vdst(/*vaddr=*/8, /*vdst=*/10, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(words);

  EXPECT_EQ((*words)[0], 0xEC05007Cu);
  EXPECT_EQ((*words)[1], 0x0000000Au);
  EXPECT_EQ((*words)[2], 0x00000008u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words->data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "flat_load_b32");

  EXPECT_FALSE(
      build_flat_load_b32_vaddr_vdst(/*vaddr=*/256, /*vdst=*/10, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_load_b32_vaddr_vdst(/*vaddr=*/8, /*vdst=*/256, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_load_b32_vaddr_vdst(/*vaddr=*/8, /*vdst=*/10, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildGfx1250FlatLoadB32) {
  const auto words =
      build_flat_load_b32_vaddr_vdst(/*vaddr=*/8, /*vdst=*/10, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(words);

  EXPECT_EQ(*words, (std::array<uint32_t, 3>{0xEC05007Cu, 0x0000000Au, 0x00000008u}));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words->data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "flat_load_b32");
  EXPECT_EQ(inst->size(), 12u);
}

TEST(InstructionBuilder, BuildGfx1250SampledPublicationOperations) {
  constexpr auto expected_cmpswap = gfx1250::build_vflat(
      gfx1250::kFlatAtomicCmpswapB32Vflat, {.saddr = static_cast<uint8_t>(gfx1250::OPR_SREG_NULL),
                                            .vdst = 10,
                                            .scope = 2,
                                            .th = 1,
                                            .vsrc = 10,
                                            .vaddr = 8});
  constexpr auto expected_swap = gfx1250::build_vflat(
      gfx1250::kFlatAtomicSwapB64Vflat, {.saddr = static_cast<uint8_t>(gfx1250::OPR_SREG_NULL),
                                         .vdst = 10,
                                         .scope = 2,
                                         .th = 1,
                                         .vsrc = 10,
                                         .vaddr = 8});
  constexpr auto expected_add = gfx1250::build_vflat(
      gfx1250::kFlatAtomicAddU64Vflat, {.saddr = static_cast<uint8_t>(gfx1250::OPR_SREG_NULL),
                                        .vdst = 10,
                                        .scope = 2,
                                        .th = 1,
                                        .vsrc = 10,
                                        .vaddr = 8});

  EXPECT_EQ(
      build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(8, 10, 10, true, 2, ROCJITSU_CODE_ARCH_GFX1250),
      expected_cmpswap);
  EXPECT_EQ(
      build_flat_atomic_swap_b64_vaddr_vsrc_vdst(8, 10, 10, true, 2, ROCJITSU_CODE_ARCH_GFX1250),
      expected_swap);
  EXPECT_EQ(
      build_flat_atomic_add_u64_vaddr_vsrc_vdst(8, 10, 10, true, 2, ROCJITSU_CODE_ARCH_GFX1250),
      expected_add);

  const auto address_add = build_v_add_u64_vgpr_offset(/*address_vgpr=*/8, /*offset_vgpr=*/10,
                                                       ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(address_add);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> low(decoder->decode(address_add->data()));
  std::unique_ptr<Instruction> wait(decoder->decode(address_add->data() + 2));
  std::unique_ptr<Instruction> high(decoder->decode(address_add->data() + 3));
  ASSERT_NE(low, nullptr);
  ASSERT_NE(wait, nullptr);
  ASSERT_NE(high, nullptr);
  EXPECT_EQ(std::string_view(low->mnemonic()), "v_add_co_u32");
  EXPECT_EQ(std::string_view(wait->mnemonic()), "s_wait_alu");
  EXPECT_EQ(std::string_view(high->mnemonic()), "v_add_co_ci_u32");
  EXPECT_EQ((*address_add)[0], 0xd7006a08u);
  EXPECT_EQ((*address_add)[1], 0x0202110au);

  const auto readfirstlane =
      build_v_readfirstlane_b32(/*sdst=*/78, /*vsrc=*/90, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(readfirstlane);
  constexpr auto expected_readfirstlane = gfx1250::build_vop1(
      gfx1250::kVReadfirstlaneB32Vop1, {.src0 = vector_source_vgpr(90), .vdst = 78});
  EXPECT_EQ(*readfirstlane, expected_readfirstlane[0]);

  const auto cell_offset = build_v_add_u64_signed_i24(/*address_vgpr=*/8, /*displacement=*/16,
                                                      ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(cell_offset);
  std::unique_ptr<Instruction> offset_low(decoder->decode(cell_offset->data()));
  std::unique_ptr<Instruction> offset_wait(decoder->decode(cell_offset->data() + 3));
  std::unique_ptr<Instruction> offset_high(decoder->decode(cell_offset->data() + 4));
  ASSERT_NE(offset_low, nullptr);
  ASSERT_NE(offset_wait, nullptr);
  ASSERT_NE(offset_high, nullptr);
  EXPECT_EQ(std::string_view(offset_low->mnemonic()), "v_add_co_u32");
  constexpr auto expected_offset_low = gfx1250::build_vop3_sdst_enc(
      gfx1250::kVAddCoU32Vop3SdstEnc, {.vdst = 8,
                                       .sdst = 106,
                                       .src0 = scalar_positive_inline_u32(16),
                                       .src1 = vector_source_vgpr(8),
                                       .src2 = scalar_positive_inline_u32(0)});
  EXPECT_EQ((*cell_offset)[0], expected_offset_low[0]);
  EXPECT_EQ((*cell_offset)[1], expected_offset_low[1]);
  EXPECT_EQ((*cell_offset)[2], build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_EQ(std::string_view(offset_wait->mnemonic()), "s_wait_alu");
  EXPECT_EQ(std::string_view(offset_high->mnemonic()), "v_add_co_ci_u32");
}

TEST(InstructionBuilder, BuildGfx1250SignedI24AddPinsInlineAndLiteralBoundaries) {
  constexpr uint16_t kScalarInlineNegativeBase = 192;
  constexpr uint16_t kScalarInlineNegativeOne = 193;
  struct DisplacementCase {
    std::string_view label;
    int32_t displacement;
    uint16_t low_source;
    uint16_t high_source;
    std::array<uint32_t, 6> expected;
  };
  constexpr std::array<DisplacementCase, 7> kCases = {{
      {"signed-24 minimum",
       -(1 << 23),
       kVopLiteralSource,
       kScalarInlineNegativeOne,
       {0xd7006a08u, 0x020210ffu, 0xff800000u, 0xbf88fffdu, 0xd5206a09u, 0x01aa12c1u}},
      {"first negative literal",
       -17,
       kVopLiteralSource,
       kScalarInlineNegativeOne,
       {0xd7006a08u, 0x020210ffu, 0xffffffefu, 0xbf88fffdu, 0xd5206a09u, 0x01aa12c1u}},
      {"last negative inline",
       -16,
       kScalarInlineNegativeBase - (-16),
       kScalarInlineNegativeOne,
       {0xd7006a08u, 0x020210d0u, 0xbf800000u, 0xbf88fffdu, 0xd5206a09u, 0x01aa12c1u}},
      {"representative negative inline",
       -4,
       kScalarInlineNegativeBase - (-4),
       kScalarInlineNegativeOne,
       {0xd7006a08u, 0x020210c4u, 0xbf800000u, 0xbf88fffdu, 0xd5206a09u, 0x01aa12c1u}},
      {"last positive inline",
       64,
       scalar_positive_inline_u32(64),
       scalar_positive_inline_u32(0),
       {0xd7006a08u, 0x020210c0u, 0xbf800000u, 0xbf88fffdu, 0xd5206a09u, 0x01aa1280u}},
      {"first positive literal",
       65,
       kVopLiteralSource,
       scalar_positive_inline_u32(0),
       {0xd7006a08u, 0x020210ffu, 0x00000041u, 0xbf88fffdu, 0xd5206a09u, 0x01aa1280u}},
      {"signed-24 maximum",
       (1 << 23) - 1,
       kVopLiteralSource,
       scalar_positive_inline_u32(0),
       {0xd7006a08u, 0x020210ffu, 0x007fffffu, 0xbf88fffdu, 0xd5206a09u, 0x01aa1280u}},
  }};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  for (const DisplacementCase &test_case : kCases) {
    SCOPED_TRACE(test_case.label);
    const auto words = build_v_add_u64_signed_i24(
        /*address_vgpr=*/8, test_case.displacement, ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_TRUE(words);
    EXPECT_EQ(*words, test_case.expected);
    EXPECT_EQ((*words)[1] & 0x1ffu, test_case.low_source);
    EXPECT_EQ((*words)[5] & 0x1ffu, test_case.high_source);

    std::unique_ptr<Instruction> low(decoder->decode(words->data()));
    std::unique_ptr<Instruction> wait(decoder->decode(words->data() + 3));
    std::unique_ptr<Instruction> high(decoder->decode(words->data() + 4));
    ASSERT_NE(low, nullptr);
    ASSERT_NE(wait, nullptr);
    ASSERT_NE(high, nullptr);
    EXPECT_EQ(std::string_view(low->mnemonic()), "v_add_co_u32");
    EXPECT_EQ(std::string_view(wait->mnemonic()), "s_wait_alu");
    EXPECT_EQ(std::string_view(high->mnemonic()), "v_add_co_ci_u32");
  }
}

TEST(InstructionBuilder, BuildFlatAtomicAddU32ReturnDeviceScope) {
  const auto words = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(words);

  EXPECT_EQ((*words)[0], 0xEC0D407Cu);
  EXPECT_EQ((*words)[1], 0x0518000Au);
  EXPECT_EQ((*words)[2], 0x00000008u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words->data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "flat_atomic_add_u32");

  EXPECT_FALSE(build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/256, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/256, /*vdst=*/10, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/256, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true, /*scope=*/4,
      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildFlatAtomicOrU32ReturnDeviceScope) {
  const auto words = build_flat_atomic_or_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(words);
  EXPECT_EQ(*words, (std::array<uint32_t, 3>{0xEC0F407Cu, 0x0518000Au, 0x00000008u}));

  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(decoder);
  std::unique_ptr<Instruction> inst(decoder->decode(words->data()));
  ASSERT_TRUE(inst);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "flat_atomic_or_b32");

  EXPECT_FALSE(
      build_flat_atomic_or_u32_vaddr_vsrc_vdst(0, 0, 0, true, 4, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_flat_atomic_or_u32_vaddr_vsrc_vdst(0, 0, 0, true, 2, ROCJITSU_CODE_ARCH_RDNA3));
}

TEST(InstructionBuilder, BuildFlatAtomicCompareSwapB64ReturnDeviceScope) {
  const auto words = build_flat_atomic_cmpswap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(words);
  EXPECT_EQ(*words, (std::array<uint32_t, 3>{0xEC10807Cu, 0x0518000Au, 0x00000008u}));

  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(decoder);
  std::unique_ptr<Instruction> inst(decoder->decode(words->data()));
  ASSERT_TRUE(inst);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "flat_atomic_cmpswap_b64");

  EXPECT_FALSE(
      build_flat_atomic_cmpswap_b64_vaddr_vsrc_vdst(0, 253, 0, true, 2, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(
      build_flat_atomic_cmpswap_b64_vaddr_vsrc_vdst(0, 0, 0, true, 2, ROCJITSU_CODE_ARCH_RDNA3));
}

TEST(InstructionBuilder, BuildFlatAtomicSwapB64ReturnDeviceScope) {
  const auto words = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/12, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(words);

  EXPECT_EQ((*words)[0], 0xEC10407Cu);
  EXPECT_EQ((*words)[1], 0x0518000Cu);
  EXPECT_EQ((*words)[2], 0x00000008u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words->data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "flat_atomic_swap_b64");

  EXPECT_FALSE(build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/255, /*vsrc=*/10, /*vdst=*/12, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/255, /*vdst=*/12, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/255, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/12, /*return_old_value=*/true, /*scope=*/4,
      ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/12, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_CDNA4));
}

} // namespace
} // namespace rocjitsu

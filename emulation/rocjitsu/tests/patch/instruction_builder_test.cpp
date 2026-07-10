// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instruction_builder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu {
namespace {

// SOPP semantics under test:
//   target = branch_pc + 4 + simm16 * 4
// Inverted:
//   simm16 = (target - (branch_pc + 4)) / 4
//
// The helper must return nullopt when the delta is not dword-aligned, when it
// would not fit in a signed 16-bit dword field, or when branch_pc/target are so
// large that the signed int64 intermediate (branch_pc + 4) would overflow.

TEST(ComputeSoppBranchSimm16, SelfBranchIsMinusOne) {
  auto simm = compute_sopp_branch_simm16(/*branch_pc=*/0x100, /*target=*/0x100);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, -1);
}

TEST(ComputeSoppBranchSimm16, FallThroughIsZero) {
  auto simm = compute_sopp_branch_simm16(/*branch_pc=*/0x100, /*target=*/0x104);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, 0);
}

TEST(ComputeSoppBranchSimm16, SmallForwardBranch) {
  auto simm = compute_sopp_branch_simm16(0x1000, 0x1100);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, 63);
}

TEST(ComputeSoppBranchSimm16, SmallBackwardBranch) {
  auto simm = compute_sopp_branch_simm16(0x1100, 0x1000);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, -65);
}

TEST(ComputeSoppBranchSimm16, MaxPositiveSimm16) {
  constexpr uint64_t pc = 0x10000;
  constexpr int64_t kMaxDelta = static_cast<int64_t>(std::numeric_limits<int16_t>::max()) * 4;
  auto simm = compute_sopp_branch_simm16(pc, pc + 4 + kMaxDelta);
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, std::numeric_limits<int16_t>::max());
}

TEST(ComputeSoppBranchSimm16, MaxNegativeSimm16) {
  constexpr uint64_t pc = 0x10'0000;
  constexpr int64_t kMinDelta = static_cast<int64_t>(std::numeric_limits<int16_t>::min()) * 4;
  auto simm = compute_sopp_branch_simm16(
      pc, static_cast<uint64_t>(static_cast<int64_t>(pc) + 4 + kMinDelta));
  ASSERT_TRUE(simm.has_value());
  EXPECT_EQ(*simm, std::numeric_limits<int16_t>::min());
}

TEST(ComputeSoppBranchSimm16, PositiveOverflowFails) {
  constexpr uint64_t pc = 0x10000;
  constexpr int64_t kJustOver = (static_cast<int64_t>(std::numeric_limits<int16_t>::max()) + 1) * 4;
  EXPECT_FALSE(compute_sopp_branch_simm16(pc, pc + 4 + kJustOver).has_value());
}

TEST(ComputeSoppBranchSimm16, NegativeOverflowFails) {
  constexpr uint64_t pc = 0x10'0000;
  constexpr int64_t kJustUnder =
      (static_cast<int64_t>(std::numeric_limits<int16_t>::min()) - 1) * 4;
  EXPECT_FALSE(compute_sopp_branch_simm16(
                   pc, static_cast<uint64_t>(static_cast<int64_t>(pc) + 4 + kJustUnder))
                   .has_value());
}

TEST(ComputeSoppBranchSimm16, NonDwordAlignedTargetFails) {
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1000, 0x1002).has_value());
}

TEST(ComputeSoppBranchSimm16, NonDwordAlignedBranchPcFails) {
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1002, 0x1100).has_value());
}

// branch_pc and target are misaligned by the same amount, so the delta is
// dword-aligned (0 and 4 here). A delta-only check would accept these; the
// branch_pc/target alignment checks must still reject them.
TEST(ComputeSoppBranchSimm16, EquallyMisalignedPcsFailEvenWhenDeltaAligned) {
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1002, 0x1006).has_value()); // delta 0
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1002, 0x100a).has_value()); // delta 4
}

// C++20 specifies truncated-toward-zero integer division/modulo, so
// `(-258) % 4 == -2 != 0`. This pins that semantic: a negative delta that
// is not a multiple of 4 must be rejected (not silently rounded).
TEST(ComputeSoppBranchSimm16, NegativeUnalignedDeltaFails) {
  // branch_pc = 0x1100, target = 0x1002 →
  //   delta = 0x1002 - 0x1100 - 4 = -0x102 = -258 bytes (not /4).
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1100, 0x1002).has_value());
}

TEST(ComputeSoppBranchSimm16, BranchPcNearInt64MaxFails) {
  constexpr uint64_t kHugePc = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  EXPECT_FALSE(compute_sopp_branch_simm16(kHugePc, kHugePc).has_value());
}

TEST(ComputeSoppBranchSimm16, TargetNearUint64MaxFails) {
  constexpr uint64_t kHugeTarget = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1;
  EXPECT_FALSE(compute_sopp_branch_simm16(0x1000, kHugeTarget).has_value());
}

TEST(InstructionBuilder, BuildSEndpgm) {
  // calculate with SOPP prefix (0x17F) << 23 | opcode 0x1 << 16
  constexpr uint32_t SOPP_S_ENDPGM_CDNA4 = 0xBF810000u;
  EXPECT_EQ(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), SOPP_S_ENDPGM_CDNA4);
  // calculate with SOPP prefix (0x17F) << 23 | opcode 0x30 << 16
  constexpr uint32_t SOPP_S_ENDPGM_RDNA4 = 0xBFB00000u;
  EXPECT_EQ(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4), SOPP_S_ENDPGM_RDNA4);
}

TEST(InstructionBuilder, BuildSMovB32UsesRdna1AndRdna2Opcodes) {
  constexpr uint16_t kDst = 4;
  constexpr uint16_t kSrc = 8;

  const uint32_t rdna1_word = build_s_mov_b32(kDst, kSrc, ROCJITSU_CODE_ARCH_RDNA1);
  const uint32_t rdna2_word = build_s_mov_b32(kDst, kSrc, ROCJITSU_CODE_ARCH_RDNA2);

  // RDNA1/2 assign s_mov_b32 opcode 3 rather than the opcode 0 used by CDNA
  // and newer RDNA targets. This is an intentional correctness fix over the
  // old architecture-agnostic builder and must not be treated as NFC.
  EXPECT_EQ((rdna1_word >> 8) & 0xFFu, rdna1::kSMovB32Sop1);
  EXPECT_EQ((rdna2_word >> 8) & 0xFFu, rdna2::kSMovB32Sop1);
  EXPECT_EQ(rdna1::kSMovB32Sop1, 3u);
  EXPECT_EQ(rdna2::kSMovB32Sop1, 3u);
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
  EXPECT_FALSE(build_s_getreg_b32(/*sdst=*/20, *hwreg, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstructionBuilder, BuildVMbcntLaneIdSequence) {
  const auto low = build_v_mbcnt_lo_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(low);
  EXPECT_EQ((*low)[0], 0xD71F000Du);
  EXPECT_EQ((*low)[1], 0x020100C1u);

  const auto high = build_v_mbcnt_hi_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, vector_source_vgpr(13), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(high);
  EXPECT_EQ((*high)[0], 0xD720000Du);
  EXPECT_EQ((*high)[1], 0x02021AC1u);

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

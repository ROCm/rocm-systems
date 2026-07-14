// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/builders.h"

#include <gtest/gtest.h>

#include <array>
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

TEST(GeneratedInstructionBuilder, PacksCdna3FormatsFromXmlLayouts) {
  // Pin both a one-word scalar format and representative two-word VALU/LDS
  // formats. These exact words were previously produced by local MachineInst
  // bitfield initialization in the CDNA4-to-CDNA3 semantic rules.
  constexpr auto sopp = cdna3::build_sopp(/*op=*/12, {.simm16 = 0xC07F});
  EXPECT_EQ(sopp, (std::array<uint32_t, 1>{0xBF8CC07Fu}));

  constexpr auto vop3 = cdna3::build_vop3(
      /*op=*/321, {.vdst = 7, .src0 = 256 + 8, .src1 = 256 + 9, .src2 = 128});
  EXPECT_EQ(vop3, (std::array<uint32_t, 2>{0xD1410007u, 0x02021308u}));

  constexpr auto ds = cdna3::build_ds(
      /*op=*/54, {.offset0 = 3, .offset1 = 5, .addr = 10, .data0 = 11, .data1 = 12, .vdst = 13});
  EXPECT_EQ(ds, (std::array<uint32_t, 2>{0xD86C0503u, 0x0D0C0B0Au}));
}

TEST(GeneratedInstructionBuilder, Gfx1250ScalarPathsUseGeneratedLayouts) {
  constexpr uint16_t kSimm16 = 0xC07F;
  constexpr uint16_t kSdst = 7;
  constexpr uint16_t kSsrc0 = 8;
  constexpr uint16_t kSsrc1 = 9;

  constexpr auto sopp = gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = kSimm16});
  EXPECT_EQ(build_sopp_encoding(ROCJITSU_CODE_ARCH_GFX1250, gfx1250::kSBranchSopp, kSimm16),
            sopp[0]);

  constexpr auto sop1 =
      gfx1250::build_sop1(gfx1250::kSMovB32Sop1, {.ssrc0 = kSsrc0, .sdst = kSdst});
  EXPECT_EQ(build_sop1_encoding(ROCJITSU_CODE_ARCH_GFX1250, gfx1250::kSMovB32Sop1, kSdst, kSsrc0),
            sop1[0]);

  constexpr auto sop2 = gfx1250::build_sop2(gfx1250::kSLshlB32Sop2,
                                            {.ssrc0 = kSsrc0, .ssrc1 = kSsrc1, .sdst = kSdst});
  EXPECT_EQ(build_sop2_encoding(ROCJITSU_CODE_ARCH_GFX1250, gfx1250::kSLshlB32Sop2, kSdst, kSsrc0,
                                kSsrc1),
            sop2[0]);
}

TEST(InstructionBuilder, BuildSGetpcB64) {
  // SOP1; opcode is arch-specific: 0x1C (GFX9), 0x1F (GFX10), 0x47 (GFX11+).
  EXPECT_EQ(build_s_getpc_b64(/*sdst_pair_base=*/0, ROCJITSU_CODE_ARCH_CDNA2), 0xBE801C00u);
  EXPECT_EQ(build_s_getpc_b64(0, ROCJITSU_CODE_ARCH_RDNA2), 0xBE801F00u);
  EXPECT_EQ(build_s_getpc_b64(0, ROCJITSU_CODE_ARCH_RDNA3), 0xBE804700u);
  EXPECT_EQ(build_s_getpc_b64(0, ROCJITSU_CODE_ARCH_RDNA4), 0xBE804700u);
  // gfx1250 renames the family (s_get_pc_i64) but keeps opcode 0x47.
  EXPECT_EQ(build_s_getpc_b64(0, ROCJITSU_CODE_ARCH_GFX1250), 0xBE804700u);
}

TEST(InstructionBuilder, BuildSAddU32) {
  // SOP2 opcode 0 (all gens); ssrc1 = inline literal 16.
  EXPECT_EQ(build_s_add_u32(/*sdst=*/0, /*ssrc0=*/0, scalar_positive_inline_u32(16),
                            ROCJITSU_CODE_ARCH_CDNA2),
            0x80009000u);
}

TEST(InstructionBuilder, BuildSAddcU32) {
  // SOP2 opcode 4 (all gens); ssrc1 = inline literal 0.
  EXPECT_EQ(build_s_addc_u32(/*sdst=*/1, /*ssrc0=*/1, scalar_positive_inline_u32(0),
                             ROCJITSU_CODE_ARCH_CDNA2),
            0x82018001u);
}

TEST(InstructionBuilder, BuildSSwappcB64) {
  // SOP1; opcode is arch-specific: 0x1E (GFX9), 0x21 (GFX10), 0x49 (GFX11+).
  // link pair s[30:31], target pair s[0:1].
  EXPECT_EQ(
      build_s_swappc_b64(/*sdst_link_base=*/30, /*ssrc0_target_base=*/0, ROCJITSU_CODE_ARCH_CDNA2),
      0xBE9E1E00u);
  EXPECT_EQ(build_s_swappc_b64(30, 0, ROCJITSU_CODE_ARCH_RDNA2), 0xBE9E2100u);
  EXPECT_EQ(build_s_swappc_b64(30, 0, ROCJITSU_CODE_ARCH_RDNA3), 0xBE9E4900u);
  EXPECT_EQ(build_s_swappc_b64(30, 0, ROCJITSU_CODE_ARCH_RDNA4), 0xBE9E4900u);
  // gfx1250: s_swap_pc_i64, opcode 0x49
  EXPECT_EQ(build_s_swappc_b64(30, 0, ROCJITSU_CODE_ARCH_GFX1250), 0xBE9E4900u);
}

TEST(InstructionBuilder, BuildSCselectB32) {
  // SOP2; opcode 0x0A on GFX9/GFX10, 0x30 on GFX11+. ssrc0 = inline 1, ssrc1 = inline 0.
  EXPECT_EQ(build_s_cselect_b32(/*sdst=*/2, scalar_positive_inline_u32(1),
                                scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA2),
            0x85028081u);
  EXPECT_EQ(build_s_cselect_b32(2, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
                                ROCJITSU_CODE_ARCH_RDNA3),
            0x98028081u);
  EXPECT_EQ(build_s_cselect_b32(2, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
                                ROCJITSU_CODE_ARCH_RDNA4),
            0x98028081u);
  // gfx1250 uses the GFX11+ opcode 0x30.
  EXPECT_EQ(build_s_cselect_b32(2, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
                                ROCJITSU_CODE_ARCH_GFX1250),
            0x98028081u);
}

TEST(InstructionBuilder, BuildSCmpLgU32) {
  // SOPC prefix (0x17E) << 23 | opcode 7 << 16; ssrc1 = inline literal 0.
  EXPECT_EQ(
      build_s_cmp_lg_u32(/*ssrc0=*/2, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA2),
      0xBF078002u);
  // Opcode 7 is invariant across gens, including gfx1250.
  EXPECT_EQ(build_s_cmp_lg_u32(2, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
            0xBF078002u);
  EXPECT_EQ(build_s_cmp_lg_u32(2, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_GFX1250),
            0xBF078002u);
}

} // namespace
} // namespace rocjitsu

// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"
#include "util/except.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace {

constexpr std::array<rj_code_arch_t, 10> kAmdGpuArchs{
    ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
    ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
    ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
    ROCJITSU_CODE_ARCH_RDNA4};

TEST(VectorBuilder, BuildVMovB32Imm) {
  // VOP1: prefix 63<<25 = 0x7E000000; src0 bits[8:0] = 255 (literal constant);
  // op bits[16:9] = 1; vdst bits[24:17]. The literal follows in word1.
  //
  // v_mov_b32 is opcode 1 on every generation and the VOP1 fields this uses sit
  // at the same offsets throughout, so all ten encode identically. The per-arch
  // dispatch is what keeps that a fact rather than an assumption: each case goes
  // through its own generation's generated builder and opcode table.
  for (const rj_code_arch_t arch : kAmdGpuArchs) {
    EXPECT_EQ(build_v_mov_b32_imm(/*vdst=*/3, /*imm=*/0xDEADBEEFu, arch),
              (std::array<uint32_t, 2>{0x7E0602FFu, 0xDEADBEEFu}))
        << "arch " << static_cast<int>(arch);
    // v0 clears the vdst field, so a nonzero one cannot come from a stray bit.
    EXPECT_EQ(build_v_mov_b32_imm(0, 0, arch), (std::array<uint32_t, 2>{0x7E0002FFu, 0u}))
        << "arch " << static_cast<int>(arch);
    // The high vdst bit is reachable: v255 fills the field.
    EXPECT_EQ(build_v_mov_b32_imm(255, 1, arch), (std::array<uint32_t, 2>{0x7FFE02FFu, 1u}))
        << "arch " << static_cast<int>(arch);
  }
}

// The RISC-V targets share the arch enum but have no VOP1 encoding, so they are
// the non-AMDGPU half of the dispatch rather than an unimplemented AMDGPU case.
TEST(VectorBuilder, BuildVMovB32ImmRejectsNonAmdGpuArch) {
  EXPECT_THROW((void)build_v_mov_b32_imm(0, 0, ROCJITSU_CODE_ARCH_RV32I), util::UnimplementedInst);
  EXPECT_THROW((void)build_v_mov_b32_imm(0, 0, ROCJITSU_CODE_ARCH_RV64I), util::UnimplementedInst);
  EXPECT_THROW((void)build_v_mov_b32_imm(0, 0, ROCJITSU_CODE_ARCH_INVALID),
               util::UnimplementedInst);
}

TEST(VectorBuilder, BuildVMovB32Src) {
  // Same VOP1 layout as the literal form, with src0 carrying the register code
  // instead of 255 and no second word. An SGPR's src0 code is its index.
  for (const rj_code_arch_t arch : kAmdGpuArchs) {
    // v_mov_b32 v3, s30: vdst 3 << 17, op 1 << 9, src0 30.
    EXPECT_EQ(build_v_mov_b32_src(/*vdst=*/3, /*src0=*/30, arch), 0x7E06021Eu)
        << "arch " << static_cast<int>(arch);
    // v0, s0 clears both fields, so a nonzero one cannot come from a stray bit.
    EXPECT_EQ(build_v_mov_b32_src(0, 0, arch), 0x7E000200u) << "arch " << static_cast<int>(arch);
    // The high vdst bit is reachable: v255 fills the field.
    EXPECT_EQ(build_v_mov_b32_src(255, 1, arch), 0x7FFE0201u)
        << "arch " << static_cast<int>(arch);
  }
}

// A literal source needs a trailing word this encoder cannot return, so it is
// rejected rather than encoded into an instruction that reads whatever follows.
TEST(VectorBuilder, BuildVMovB32SrcRejectsLiteralSource) {
  for (const rj_code_arch_t arch : kAmdGpuArchs)
    EXPECT_THROW((void)build_v_mov_b32_src(0, kVectorSrcLiteral, arch), util::InvalidInst)
        << "arch " << static_cast<int>(arch);
}

TEST(VectorBuilder, BuildVMovB32SrcRejectsNonAmdGpuArch) {
  EXPECT_THROW((void)build_v_mov_b32_src(0, 0, ROCJITSU_CODE_ARCH_RV32I), util::UnimplementedInst);
  EXPECT_THROW((void)build_v_mov_b32_src(0, 0, ROCJITSU_CODE_ARCH_RV64I), util::UnimplementedInst);
  EXPECT_THROW((void)build_v_mov_b32_src(0, 0, ROCJITSU_CODE_ARCH_INVALID),
               util::UnimplementedInst);
}

// kVectorSrcLiteral is declared once from cdna1's table; this pins that every
// other generation's table still agrees, so the single constant stays correct.
TEST(VectorBuilder, SrcLiteralCodeMatchesGeneratedTables) {
  EXPECT_EQ(kVectorSrcLiteral, 255);
  EXPECT_EQ(kVectorSrcLiteral, cdna1::OPR_SRC_SRC_LITERAL);
  EXPECT_EQ(kVectorSrcLiteral, cdna2::OPR_SRC_SRC_LITERAL);
  EXPECT_EQ(kVectorSrcLiteral, cdna3::OPR_SRC_SRC_LITERAL);
  EXPECT_EQ(kVectorSrcLiteral, cdna4::OPR_SRC_SRC_LITERAL);
  EXPECT_EQ(kVectorSrcLiteral, cdna5::OPR_SRC_SRC_LITERAL);
  EXPECT_EQ(kVectorSrcLiteral, rdna1::OPR_SRC_SRC_LITERAL);
  EXPECT_EQ(kVectorSrcLiteral, rdna2::OPR_SRC_SRC_LITERAL);
  EXPECT_EQ(kVectorSrcLiteral, rdna3::OPR_SRC_SRC_LITERAL);
  EXPECT_EQ(kVectorSrcLiteral, rdna3_5::OPR_SRC_SRC_LITERAL);
  EXPECT_EQ(kVectorSrcLiteral, rdna4::OPR_SRC_SRC_LITERAL);
}

} // namespace
} // namespace rocjitsu

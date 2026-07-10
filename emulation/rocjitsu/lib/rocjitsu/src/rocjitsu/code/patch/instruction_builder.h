// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_builder.h
/// @brief ISA-parameterized instruction encoding helpers for code patching.
/// @brief Builder functions for constructing common AMDGPU instructions.
///
/// @details Provides helpers for encoding frequently-used instructions
/// (s_branch, s_nop) in the DBT and DBI layers. The SOPP encoding
/// format is identical across all AMDGPU ISA generations:
///   bits[31:23] = SOPP encoding selector
///   bits[22:16] = op (7-bit opcode)
///   bits[15:0]  = simm16 (16-bit signed/unsigned immediate)
///
/// IMPORTANT: While the SOPP *format* is consistent across ISAs, the
/// *opcodes* for specific instructions differ between generations.
/// s_branch is opcode 2 on GFX9 (CDNA1-4) but opcode 32 on GFX12 (RDNA4).
/// These builders are parameterized by target ISA via rj_code_arch_t.

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/opcodes.h"

namespace rocjitsu {

class Instruction;

/// @brief SOPP encoding prefix, consistent across all AMDGPU ISA generations.
inline constexpr uint32_t kSoppEncodingPrefix = cdna4::encoding::kSopp;
inline constexpr uint32_t kSop1EncodingPrefix = cdna4::encoding::kSop1;
// SOP2 stores only a two-bit fixed prefix in MachineInst::encoding. Generated
// encoding::kSop2 is instead the wider primary-decode selector (word0 >> 23),
// so using it directly here would conflate two different representations.
inline constexpr uint32_t kSop2EncodingPrefix = 0x2;
inline constexpr uint32_t kSopkEncodingPrefix = 0xB;
inline constexpr uint16_t kScalarPositiveInlineBase = 128;
inline constexpr uint16_t kDelayAluSaluDep1 = 9;
inline constexpr uint16_t kVectorSourceVgprBase = 256;
inline constexpr uint16_t kVopLiteralSource = 255;

/// @brief Pack a SOPP instruction word from its constituent fields.
///
/// @param op      7-bit SOPP opcode.
/// @param simm16  16-bit immediate field.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t pack_sopp(uint32_t op, uint16_t simm16) {
  return (kSoppEncodingPrefix << 23) | (op << 16) | simm16;
}

/// @brief Pack a SOP1 instruction word from its constituent fields.
[[nodiscard]] inline constexpr uint32_t pack_sop1(uint32_t op, uint32_t sdst, uint32_t ssrc0) {
  return (kSop1EncodingPrefix << 23) | ((sdst & 0x7Fu) << 16) | ((op & 0xFFu) << 8) |
         (ssrc0 & 0xFFu);
}

/// @brief Pack a SOP2 instruction word from its constituent fields.
[[nodiscard]] inline constexpr uint32_t pack_sop2(uint32_t op, uint32_t sdst, uint32_t ssrc0,
                                                  uint32_t ssrc1) {
  return (kSop2EncodingPrefix << 30) | ((op & 0x7Fu) << 23) | ((sdst & 0x7Fu) << 16) |
         ((ssrc1 & 0xFFu) << 8) | (ssrc0 & 0xFFu);
}

/// @brief Pack a SOPK instruction word from its constituent fields.
[[nodiscard]] inline constexpr uint32_t pack_sopk(uint32_t op, uint32_t sdst, uint16_t simm16) {
  return (kSopkEncodingPrefix << 28) | ((op & 0x1Fu) << 23) | ((sdst & 0x7Fu) << 16) | simm16;
}

/// @brief Pack a VOP2 instruction word from its constituent fields.
[[nodiscard]] inline constexpr uint32_t pack_vop2(uint32_t op, uint32_t vdst, uint32_t src0,
                                                  uint32_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
}

/// @brief Scalar source operand encoding for a non-negative inline integer.
[[nodiscard]] inline constexpr uint16_t scalar_positive_inline_u32(uint16_t value) {
  return static_cast<uint16_t>(kScalarPositiveInlineBase + value);
}

/// @brief Vector source operand encoding for a VGPR.
[[nodiscard]] inline constexpr uint16_t vector_source_vgpr(uint16_t vgpr) {
  return static_cast<uint16_t>(kVectorSourceVgprBase + vgpr);
}

/// @brief Compute the SOPP simm16 dword field for a branch from @p branch_pc
///        to @p target under SOPP semantics: target = branch_pc + 4 + simm16*4.
///
/// Returns std::nullopt if @p branch_pc or @p target is not dword-aligned, if
/// the resulting delta does not fit in a signed 16-bit dword field, or if
/// @p branch_pc / @p target are large enough that the signed int64 intermediate
/// would overflow.
///
/// Shared by DBT cave-entry/return branches and the DBI relocation trampoline
/// so both paths fail closed on the same range.
[[nodiscard]] inline constexpr std::optional<int16_t> compute_sopp_branch_simm16(uint64_t branch_pc,
                                                                                 uint64_t target) {
  constexpr int64_t kBranchPcBiasBytes = static_cast<int64_t>(sizeof(uint32_t));
  constexpr uint64_t kMaxSignedTarget = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  constexpr uint64_t kMaxSignedBranchPc =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - kBranchPcBiasBytes);
  if (branch_pc > kMaxSignedBranchPc || target > kMaxSignedTarget)
    return std::nullopt;

  // The SOPP immediate is a signed *dword* offset, so both the branch base
  // (branch_pc + 4) and the target must be dword-aligned.
  if (branch_pc % sizeof(uint32_t) != 0 || target % sizeof(uint32_t) != 0)
    return std::nullopt;

  const int64_t delta_bytes =
      static_cast<int64_t>(target) - (static_cast<int64_t>(branch_pc) + kBranchPcBiasBytes);
  const int64_t delta_dwords = delta_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if (delta_dwords < std::numeric_limits<int16_t>::min() ||
      delta_dwords > std::numeric_limits<int16_t>::max())
    return std::nullopt;

  return static_cast<int16_t>(delta_dwords);
}

/// @brief Get the s_branch opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_branch(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSBranchSopp;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSBranchSopp;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSBranchSopp;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSBranchSopp;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSBranchSopp;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSBranchSopp;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSBranchSopp;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSBranchSopp;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSBranchSopp;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSBranchSopp;
  default:
    return cdna4::kSBranchSopp;
  }
}

/// @brief Get the s_endpgm opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_endpgm(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSEndpgmSopp;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSEndpgmSopp;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSEndpgmSopp;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSEndpgmSopp;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSEndpgmSopp;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSEndpgmSopp;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSEndpgmSopp;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSEndpgmSopp;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSEndpgmSopp;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSEndpgmSopp;
  default:
    return cdna4::kSEndpgmSopp;
  }
}
/// @brief Get the s_nop opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_nop(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSNopSopp;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSNopSopp;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSNopSopp;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSNopSopp;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSNopSopp;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSNopSopp;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSNopSopp;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSNopSopp;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSNopSopp;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSNopSopp;
  default:
    return cdna4::kSNopSopp;
  }
}

/// @brief Get the s_sleep opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_sleep(rj_code_arch_t arch) {
  // GFX9/RDNA1/RDNA2: opcode 14; GFX12-class ISA: opcode 3.
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 3;
  default:
    return 14;
  }
}

/// @brief Get the s_sleep_var opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop1_op_sleep_var([[maybe_unused]] rj_code_arch_t arch) {
  return 0x58;
}

/// @brief Get the s_lshl_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop2_op_lshl_b32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSLshlB32Sop2;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSLshlB32Sop2;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSLshlB32Sop2;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSLshlB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSLshlB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSLshlB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSLshlB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSLshlB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSLshlB32Sop2;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSLshlB32Sop2;
  default:
    return cdna4::kSLshlB32Sop2;
  }
}

/// @brief Get the s_lshr_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop2_op_lshr_b32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSLshrB32Sop2;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSLshrB32Sop2;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSLshrB32Sop2;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSLshrB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSLshrB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSLshrB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSLshrB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSLshrB32Sop2;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSLshrB32Sop2;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSLshrB32Sop2;
  default:
    return cdna4::kSLshrB32Sop2;
  }
}

/// @brief Get the s_delay_alu opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_delay_alu(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSDelayAluSopp;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSDelayAluSopp;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSDelayAluSopp;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSDelayAluSopp;
  default:
    return rdna4::kSDelayAluSopp;
  }
}

/// @brief Get the s_mov_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop1_op_mov_b32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSMovB32Sop1;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSMovB32Sop1;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSMovB32Sop1;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSMovB32Sop1;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSMovB32Sop1;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSMovB32Sop1;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSMovB32Sop1;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSMovB32Sop1;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSMovB32Sop1;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSMovB32Sop1;
  default:
    return cdna4::kSMovB32Sop1;
  }
}

/// @brief Get the VOP2 v_lshrrev_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr std::optional<uint32_t> vop2_op_lshrrev_b32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return 16;
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return 22;
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 25;
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return std::nullopt;
  }
  return std::nullopt;
}

/// @brief Get the VOP2 v_lshlrev_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr std::optional<uint32_t> vop2_op_lshlrev_b32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 24;
  default:
    return std::nullopt;
  }
}

/// @brief Get the VOP2 v_add_nc_u32 opcode for a target ISA.
[[nodiscard]] inline constexpr std::optional<uint32_t> vop2_op_add_nc_u32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 37;
  default:
    return std::nullopt;
  }
}

/// @brief Get the VOP2 v_and_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr std::optional<uint32_t> vop2_op_and_b32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 27;
  default:
    return std::nullopt;
  }
}

/// @brief Encode an s_branch instruction for the given target ISA.
///
/// @param offset_dwords  Signed offset in dwords from (PC + 4).
/// @param arch           Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t build_s_branch(int16_t offset_dwords, rj_code_arch_t arch) {
  return pack_sopp(sopp_op_branch(arch), static_cast<uint16_t>(offset_dwords));
}

/// @brief Patch an emitted direct PC-relative branch instruction in-place.
///
/// @details @p words points into the translated output buffer. @p delta_bytes is
/// relative to the instruction's branch base. For AMDGPU SOPP direct branches
/// and SOPK `s_call_b64`, the base is the next instruction and the immediate is
/// a signed dword offset. The function replaces bits [15:0] of word 0. It
/// returns false when @p inst has no decoded PC-relative branch offset, the
/// buffer is empty, or the delta is not representable by a signed 16-bit dword
/// immediate.
[[nodiscard]] bool patch_pcrel_branch_offset(const Instruction &inst, std::span<uint32_t> words,
                                             int64_t delta_bytes, rj_code_arch_t arch);

/// @brief Append a canonical PC-relative target builder for a recovered branch.
///
/// @details The original getpc remains in the instruction stream and initializes
/// @p pc_sreg / @p pc_sreg+1. This helper appends the smallest positive or
/// negative scalar add/sub sequence needed to turn that pair into the final
/// relocated target. Static PC recovery only records address-builder ranges that
/// have enough instruction words for this replacement to be written in place.
[[nodiscard]] bool append_pc_delta_builder(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                           uint16_t pc_sreg, int64_t delta);

/// @brief Encode an s_nop instruction for the given target ISA.
///
/// @param cycles  Number of additional stall cycles (0-based).
/// @param arch    Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t
build_s_nop(uint16_t cycles = 0, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  return pack_sopp(sopp_op_nop(arch), cycles);
}

/// @brief Encode an s_sleep instruction for the given target ISA.
///
/// @param delay  ISA-defined sleep delay immediate.
/// @param arch   Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t
build_s_sleep(uint16_t delay, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  return pack_sopp(sopp_op_sleep(arch), delay);
}

/// @brief Encode an s_sleep_var instruction for the given target ISA.
///
/// @param ssrc0  Scalar source operand encoding that supplies the delay.
/// @param arch   Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t
build_s_sleep_var(uint16_t ssrc0, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  return pack_sop1(sop1_op_sleep_var(arch), 0, ssrc0);
}

/// @brief Build the SOPK hwreg immediate used by s_getreg/s_setreg instructions.
[[nodiscard]] inline constexpr std::optional<uint16_t>
build_hwreg_imm(uint16_t reg_id, uint16_t offset, uint16_t size_bits) {
  if (reg_id > 63 || offset > 31 || size_bits == 0 || size_bits > 32)
    return std::nullopt;
  return static_cast<uint16_t>(reg_id | (offset << 6u) | ((size_bits - 1u) << 11u));
}

/// @brief Encode RDNA4-class `s_getreg_b32 sdst, hwreg`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_getreg_b32(uint16_t sdst, uint16_t hwreg, rj_code_arch_t arch) {
  if ((arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_GFX1250) || sdst > 127)
    return std::nullopt;
  return pack_sopk(/*op=*/17, sdst, hwreg);
}

/// @brief Encode v_mov_b32_e32 for the given target ISA.
///
/// @param vdst  Destination VGPR.
/// @param src0  VOP1 source operand encoding.
/// @param arch  Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t build_v_mov_b32_e32(uint16_t vdst, uint16_t src0,
                                                            [[maybe_unused]] rj_code_arch_t arch) {
  return (0x3Fu << 25) | (static_cast<uint32_t>(vdst) << 17) | (1u << 9) | (src0 & 0x1FFu);
}

/// @brief Encode VOP2 `v_lshrrev_b32 vdst, src0, vsrc1`.
///
/// @details The operation is `vdst = vsrc1 >> src0`. @p src0 is a VOP source
/// operand encoding, so inline constants such as `scalar_positive_inline_u32(2)`
/// are allowed. @p vsrc1 is a VGPR number.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_lshrrev_b32_e32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_lshrrev_b32(arch);
  if (!op)
    return std::nullopt;
  return pack_vop2(*op, vdst, src0, vsrc1);
}

/// @brief Encode VOP2 `v_lshlrev_b32 vdst, src0, vsrc1`.
///
/// @details The operation is `vdst = vsrc1 << src0`. @p src0 is a VOP source
/// operand encoding, so inline constants such as `scalar_positive_inline_u32(16)`
/// are allowed. @p vsrc1 is a VGPR number.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_lshlrev_b32_e32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_lshlrev_b32(arch);
  if (!op)
    return std::nullopt;
  return pack_vop2(*op, vdst, src0, vsrc1);
}

/// @brief Encode VOP2 `v_add_nc_u32 vdst, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_add_nc_u32_e32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_add_nc_u32(arch);
  if (!op)
    return std::nullopt;
  return pack_vop2(*op, vdst, src0, vsrc1);
}

/// @brief Encode VOP2 `v_and_b32 vdst, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_and_b32_e32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_and_b32(arch);
  if (!op)
    return std::nullopt;
  return pack_vop2(*op, vdst, src0, vsrc1);
}

/// @brief Encode VOP2 `v_and_b32 vdst, literal, vsrc1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_and_b32_e32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_and_b32(arch);
  if (!op)
    return std::nullopt;
  return std::array<uint32_t, 2>{pack_vop2(*op, vdst, kVopLiteralSource, vsrc1), literal};
}

/// @brief Encode RDNA4 `v_readfirstlane_b32 sdst, vsrc`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_readfirstlane_b32(uint16_t sdst, uint16_t vsrc, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || sdst > 127 || vsrc > 255)
    return std::nullopt;
  return (0x3Fu << 25) | (static_cast<uint32_t>(sdst) << 17) | (2u << 9) | vector_source_vgpr(vsrc);
}

/// @brief Encode RDNA4 `v_mbcnt_lo_u32_b32 vdst, src0, src1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_mbcnt_lo_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD71F0000u | static_cast<uint32_t>(vdst),
                                 0x02000000u | (static_cast<uint32_t>(src1) << 9u) | src0};
}

/// @brief Encode RDNA4 `v_mbcnt_hi_u32_b32 vdst, src0, src1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_mbcnt_hi_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD7200000u | static_cast<uint32_t>(vdst),
                                 0x02000000u | (static_cast<uint32_t>(src1) << 9u) | src0};
}

/// @brief Encode RDNA4 `v_cmp_eq_u32_e32 vcc_lo, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_eq_u32_e32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return 0x7C940000u | (static_cast<uint32_t>(vsrc1) << 9u) | src0;
}

/// @brief Encode RDNA4 `v_cmp_ne_u32_e32 vcc_lo, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_ne_u32_e32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return 0x7C9A0000u | (static_cast<uint32_t>(vsrc1) << 9u) | src0;
}

/// @brief Encode RDNA4 `v_cmp_gt_u32_e32 vcc_lo, src0, vsrc1`.
///
/// Useful as `src0=capacity, vsrc1=slot`, which tests `slot < capacity` while
/// keeping the immediate-like operand in the source position this encoding
/// accepts.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_gt_u32_e32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return 0x7C980000u | (static_cast<uint32_t>(vsrc1) << 9u) | src0;
}

/// @brief Encode RDNA4 `v_mov_b32` in VOP3 form with a literal source.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_v_mov_b32_e64_literal(uint16_t vdst, uint32_t literal, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || vdst > 255)
    return std::nullopt;
  return std::array<uint32_t, 3>{0xD5810000u | static_cast<uint32_t>(vdst), 0x000000FFu, literal};
}

/// @brief Encode RDNA4 `flat_store_b32 v[vaddr:vaddr+1], vsrc`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_store_b32_vaddr_vsrc(uint16_t vaddr, uint16_t vsrc, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || vaddr > 255 || vsrc > 255)
    return std::nullopt;
  constexpr uint32_t kRdna4FlatNoSaddr = 0x7C;
  return std::array<uint32_t, 3>{0xEC068000u | kRdna4FlatNoSaddr,
                                 static_cast<uint32_t>(vsrc) << 23u, static_cast<uint32_t>(vaddr)};
}

/// @brief Encode RDNA4 `flat_load_b32 vdst, v[vaddr:vaddr+1]`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_load_b32_vaddr_vdst(uint16_t vaddr, uint16_t vdst, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || vaddr > 255 || vdst > 255)
    return std::nullopt;
  constexpr uint32_t kRdna4FlatNoSaddr = 0x7C;
  return std::array<uint32_t, 3>{0xEC050000u | kRdna4FlatNoSaddr, static_cast<uint32_t>(vdst),
                                 static_cast<uint32_t>(vaddr)};
}

/// @brief Encode RDNA4 `flat_atomic_add_u32 vdst, v[vaddr:vaddr+1], vsrc`.
///
/// @details Uses the no-SADDR flat form. `return_old_value=true` encodes the
/// GFX12 atomic-return TH value so the old memory value is written to @p vdst.
/// @p scope is the two-bit RDNA4 SCOPE field; device scope is 2.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_atomic_add_u32_vaddr_vsrc_vdst(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                          bool return_old_value, uint8_t scope,
                                          rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || vaddr > 255 || vsrc > 255 || vdst > 255 || scope > 3)
    return std::nullopt;
  constexpr uint32_t kRdna4FlatNoSaddr = 0x7C;
  constexpr uint32_t kRdna4AtomicReturnTh = 1;
  const uint32_t th = return_old_value ? kRdna4AtomicReturnTh : 0u;
  return std::array<uint32_t, 3>{0xEC0D4000u | kRdna4FlatNoSaddr,
                                 static_cast<uint32_t>(vdst) |
                                     (static_cast<uint32_t>(scope) << 18u) | (th << 20u) |
                                     (static_cast<uint32_t>(vsrc) << 23u),
                                 static_cast<uint32_t>(vaddr)};
}

/// @brief Encode RDNA4 `flat_atomic_swap_b64 v[vdst:vdst+1], v[vaddr:vaddr+1],
///        v[vsrc:vsrc+1]`.
///
/// @details Uses the no-SADDR flat form. `return_old_value=true` encodes the
/// GFX12 atomic-return TH value so the old 64-bit memory value is written to
/// @p vdst/@p vdst+1. @p scope is the two-bit RDNA4 SCOPE field; device scope
/// is 2.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_atomic_swap_b64_vaddr_vsrc_vdst(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                           bool return_old_value, uint8_t scope,
                                           rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || vaddr > 254 || vsrc > 254 || vdst > 254 || scope > 3)
    return std::nullopt;
  constexpr uint32_t kRdna4FlatNoSaddr = 0x7C;
  constexpr uint32_t kRdna4AtomicReturnTh = 1;
  const uint32_t th = return_old_value ? kRdna4AtomicReturnTh : 0u;
  return std::array<uint32_t, 3>{0xEC104000u | kRdna4FlatNoSaddr,
                                 static_cast<uint32_t>(vdst) |
                                     (static_cast<uint32_t>(scope) << 18u) | (th << 20u) |
                                     (static_cast<uint32_t>(vsrc) << 23u),
                                 static_cast<uint32_t>(vaddr)};
}

/// @brief Encode an s_endpgm instruction for the given target ISA.
///
/// @param arch    Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t build_s_endpgm(rj_code_arch_t arch) {
  return pack_sopp(sopp_op_endpgm(arch), 0);
}

/// @brief Encode s_delay_alu for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_delay_alu(uint16_t simm16, rj_code_arch_t arch) {
  return pack_sopp(sopp_op_delay_alu(arch), simm16);
}

/// @brief Encode s_mov_b32 for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_mov_b32(uint16_t sdst, uint16_t ssrc0,
                                                        rj_code_arch_t arch) {
  return pack_sop1(sop1_op_mov_b32(arch), sdst, ssrc0);
}

/// @brief Encode RDNA4 `s_mov_b64 s[sdst:sdst+1], s[ssrc0:ssrc0+1]`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_mov_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || sdst > 126 || ssrc0 > 255)
    return std::nullopt;
  return pack_sop1(1, sdst, ssrc0);
}

/// @brief Encode RDNA4 `s_and_saveexec_b64 s[sdst:sdst+1], s[ssrc0:ssrc0+1]`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_and_saveexec_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || sdst > 126 || ssrc0 > 255)
    return std::nullopt;
  return pack_sop1(0x21, sdst, ssrc0);
}

/// @brief Encode RDNA4 `s_cbranch_vccz`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_vccz(int16_t offset_dwords,
                                                                            rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4)
    return std::nullopt;
  constexpr uint32_t kRdna4SoppCbranchVccz = 35;
  return pack_sopp(kRdna4SoppCbranchVccz, static_cast<uint16_t>(offset_dwords));
}

/// @brief Encode s_lshl_b32 for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_lshl_b32(uint16_t sdst, uint16_t ssrc0,
                                                         uint16_t ssrc1, rj_code_arch_t arch) {
  return pack_sop2(sop2_op_lshl_b32(arch), sdst, ssrc0, ssrc1);
}

/// @brief Encode s_lshr_b32 for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_lshr_b32(uint16_t sdst, uint16_t ssrc0,
                                                         uint16_t ssrc1, rj_code_arch_t arch) {
  return pack_sop2(sop2_op_lshr_b32(arch), sdst, ssrc0, ssrc1);
}

} // namespace rocjitsu

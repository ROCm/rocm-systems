// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_builder.h
/// @brief ISA-parameterized instruction encoding helpers for code patching.
/// @brief Builder functions for constructing common AMDGPU instructions.
///
/// @details Provides helpers for encoding frequently-used instructions
/// (s_branch, s_nop) in the DBT and DBI layers. The SOPP encoding
/// format is identical across all AMDGPU ISA generations:
///   bits[31:23] = 0x17F (SOPP encoding prefix)
///   bits[22:16] = op (7-bit opcode)
///   bits[15:0]  = simm16 (16-bit signed/unsigned immediate)
///
/// IMPORTANT: While the SOPP *format* is consistent across ISAs, the
/// *opcodes* for specific instructions differ between generations.
/// s_branch is opcode 2 on GFX9 (CDNA1-4) but opcode 32 on GFX12 (RDNA4).
/// These builders are parameterized by target ISA via rj_code_arch_t.

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

class Instruction;

/// @brief SOPP encoding prefix, consistent across all AMDGPU ISA generations.
inline constexpr uint32_t kSoppEncodingPrefix = 0x17F;
inline constexpr uint32_t kSop1EncodingPrefix = 0x17D;
inline constexpr uint32_t kSop2EncodingPrefix = 0x2;
inline constexpr uint16_t kScalarPositiveInlineBase = 128;
inline constexpr uint16_t kDelayAluSaluDep1 = 9;

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

/// @brief Scalar source operand encoding for a non-negative inline integer.
[[nodiscard]] inline constexpr uint16_t scalar_positive_inline_u32(uint16_t value) {
  return static_cast<uint16_t>(kScalarPositiveInlineBase + value);
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
  // GFX9 (CDNA1-4): opcode 2; GFX12 (RDNA3/3.5/4): opcode 32
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 32;
  default:
    return 2;
  }
}

/// @brief Get the s_endpgm opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_endpgm(rj_code_arch_t arch) {
  // GFX9 (CDNA1-4): opcode 1; GFX12 (RDNA3/3.5/4, gfx1250): opcode 48
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 48;
  default:
    return 1;
  }
}
/// @brief Get the s_nop opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_nop([[maybe_unused]] rj_code_arch_t arch) {
  return 0; // s_nop is opcode 0 on all ISAs
}

/// @brief Get the s_lshl_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop2_op_lshl_b32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 8;
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return 30;
  default:
    return 28;
  }
}

/// @brief Get the s_lshr_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop2_op_lshr_b32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 10;
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return 32;
  default:
    return 30;
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

/// @brief Encode an s_endpgm instruction for the given target ISA.
///
/// @param arch    Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t build_s_endpgm(rj_code_arch_t arch) {
  return pack_sopp(sopp_op_endpgm(arch), 0);
}

/// @brief Encode s_delay_alu for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_delay_alu(uint16_t simm16, rj_code_arch_t) {
  constexpr uint8_t kSoppDelayAlu = 7;
  return pack_sopp(kSoppDelayAlu, simm16);
}

/// @brief Encode s_mov_b32 for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_mov_b32(uint16_t sdst, uint16_t ssrc0,
                                                        rj_code_arch_t) {
  return pack_sop1(0, sdst, ssrc0);
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

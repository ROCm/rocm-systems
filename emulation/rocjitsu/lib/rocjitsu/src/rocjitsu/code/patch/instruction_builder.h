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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"

namespace rocjitsu {

class Instruction;

/// @brief SOPP encoding prefix, consistent across all AMDGPU ISA generations.
inline constexpr uint32_t kSoppEncodingPrefix = 0x17F;
inline constexpr uint32_t kSop1EncodingPrefix = 0x17D;
inline constexpr uint32_t kSop2EncodingPrefix = 0x2;
inline constexpr uint32_t kSopkEncodingPrefix = 0xB;
inline constexpr uint16_t kScalarPositiveInlineBase = 128;
inline constexpr uint16_t kDelayAluSaluDep1 = 9;
inline constexpr size_t kMaxRecoveredIndirectTransferWords = 6;

/// @brief CDNA3 memory instruction encoder used by handwritten DBT lowerings.
///
/// @details The semantic DBT rules are allowed to decide *which* memory
/// operation they need, but the raw bitfield construction goes through this
/// target-specific builder. Keeping that boundary explicit matters for LDS
/// virtualization: once a translated kernel is in virtual-LDS mode, this is the
/// place where DS-style LDS accesses can be redirected to the global-memory
/// scratchpad form without spreading encoding decisions across rule bodies.
class Cdna3MemoryInstructionBuilder {
public:
  using WordPair = std::pair<uint32_t, uint32_t>;

  /// @brief Source-independent fields preserved when rebuilding a CDNA3 MUBUF.
  ///
  /// @details The LDS-destination MUBUF lowering decodes a CDNA4 source
  /// instruction, issues an ordinary CDNA3 global load, then stores the result
  /// to LDS explicitly. The source architecture should not leak into this
  /// builder, so callers copy only the fields that the target MUBUF encoding
  /// needs.
  struct MubufOperands {
    uint16_t offset = 0;
    bool offen = false;
    bool idxen = false;
    bool sc0 = false;
    bool sc1 = false;
    bool nt = false;
    uint8_t vaddr = 0;
    uint8_t srsrc = 0;
    uint8_t soffset = 0;
  };

  /// @brief Common addressing fields for CDNA3 FLAT instructions in GLOBAL segment mode.
  ///
  /// @details GLOBAL segment mode addresses memory as:
  /// `saddr[63:0] + sign_extend(vgpr_addr[31:0]) + signed_offset13`.
  /// Virtual LDS lowering uses this form with `saddr` holding the backing-buffer
  /// base and `addr` holding the per-lane LDS byte offset.
  struct FlatGlobalOperands {
    uint16_t signed_offset13 = 0;
    bool sc0 = false;
    bool sc1 = false;
    bool nt = false;
    uint8_t addr = 0;
    uint8_t saddr = 0;
    bool acc = false;
  };

  /// @brief Encode a CDNA3 DS instruction.
  [[nodiscard]] static WordPair ds(uint8_t op, uint8_t vdst, uint8_t addr, uint8_t data0 = 0,
                                   uint8_t data1 = 0, uint8_t offset0 = 0, uint8_t offset1 = 0) {
    cdna3::DsMachineInst dst{};
    dst.encoding = 0x36;
    dst.op = op;
    dst.offset0 = offset0;
    dst.offset1 = offset1;
    dst.addr = addr;
    dst.data0 = data0;
    dst.data1 = data1;
    dst.vdst = vdst;
    return encode_pair(dst);
  }

  /// @brief Encode a CDNA3 non-LDS MUBUF instruction.
  [[nodiscard]] static WordPair mubuf(const MubufOperands &src, uint8_t op, uint8_t vdata) {
    cdna3::MubufMachineInst dst{};
    dst.encoding = 0x38;
    dst.op = op;
    dst.offset = src.offset;
    dst.offen = src.offen ? 1 : 0;
    dst.idxen = src.idxen ? 1 : 0;
    dst.sc0 = src.sc0 ? 1 : 0;
    dst.sc1 = src.sc1 ? 1 : 0;
    dst.lds = 0;
    dst.nt = src.nt ? 1 : 0;
    dst.vaddr = src.vaddr;
    dst.vdata = vdata;
    dst.srsrc = src.srsrc;
    dst.acc = 0;
    dst.soffset = src.soffset;
    return encode_pair(dst);
  }

  /// @brief Encode a CDNA3 flat/global load using a 64-bit SGPR base plus VGPR offset.
  [[nodiscard]] static WordPair flat_global_load(const FlatGlobalOperands &src, uint8_t op,
                                                 uint8_t vdst) {
    cdna3::FlatMachineInst dst{};
    dst.encoding = 0x37;
    dst.offset = src.signed_offset13 & 0x0FFF;
    dst.pad_12 = (src.signed_offset13 >> 12) & 0x1;
    dst.lds = 0;
    dst.seg = 2;
    dst.sc0 = src.sc0 ? 1 : 0;
    dst.nt = src.nt ? 1 : 0;
    dst.op = op;
    dst.sc1 = src.sc1 ? 1 : 0;
    dst.addr = src.addr;
    dst.saddr = src.saddr;
    dst.acc = src.acc ? 1 : 0;
    dst.vdst = vdst;
    return encode_pair(dst);
  }

  /// @brief Encode a CDNA3 flat/global store using a 64-bit SGPR base plus VGPR offset.
  [[nodiscard]] static WordPair flat_global_store(const FlatGlobalOperands &src, uint8_t op,
                                                  uint8_t data) {
    cdna3::FlatMachineInst dst{};
    dst.encoding = 0x37;
    dst.offset = src.signed_offset13 & 0x0FFF;
    dst.pad_12 = (src.signed_offset13 >> 12) & 0x1;
    dst.lds = 0;
    dst.seg = 2;
    dst.sc0 = src.sc0 ? 1 : 0;
    dst.nt = src.nt ? 1 : 0;
    dst.op = op;
    dst.sc1 = src.sc1 ? 1 : 0;
    dst.addr = src.addr;
    dst.data = data;
    dst.saddr = src.saddr;
    dst.acc = src.acc ? 1 : 0;
    return encode_pair(dst);
  }

  /// @brief Encode a CDNA3 flat_scratch dword load/store used for semantic spills.
  ///
  /// @details This is the scratch segment form with no VGPR address operand and
  /// no SGPR offset operand. The per-lane spill slot is carried in the signed
  /// 13-bit immediate field, matching the previous local encoder behavior.
  [[nodiscard]] static WordPair flat_scratch_dword(uint8_t op, uint8_t vgpr, uint32_t byte_offset,
                                                   bool is_load) {
    cdna3::FlatScratchMachineInst dst{};
    dst.encoding = 0x37;
    dst.op = op;
    dst.offset = byte_offset & 0x1FFF;
    dst.seg = 1;
    dst.sve = 0;
    dst.saddr = 0x7F;
    if (is_load)
      dst.vdst = vgpr;
    else
      dst.data = vgpr;
    return encode_pair(dst);
  }

private:
  template <typename MachineInst>
  [[nodiscard]] static WordPair encode_pair(const MachineInst &inst) {
    static_assert(sizeof(MachineInst) == sizeof(uint32_t) * 2,
                  "CDNA3 memory instruction encodings are 64-bit");
    uint32_t words[2]{};
    std::memcpy(words, &inst, sizeof(inst));
    return {words[0], words[1]};
  }
};

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

/// @brief Get the s_getpc_b64 SOP1 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop1_op_getpc_b64(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return 0x1f;
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return 0x47;
  default:
    return 0x1c;
  }
}

/// @brief Get the s_setpc_b64 SOP1 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop1_op_setpc_b64(rj_code_arch_t arch) {
  return sop1_op_getpc_b64(arch) + 1;
}

/// @brief Get the s_swappc_b64 SOP1 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop1_op_swappc_b64(rj_code_arch_t arch) {
  return sop1_op_getpc_b64(arch) + 2;
}

/// @brief Get the s_call_b64 SOPK opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopk_op_call_b64(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return 0x16;
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 0x14;
  default:
    return 0x15;
  }
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

/// @brief Encode an s_getpc_b64 instruction for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_getpc_b64(uint16_t sdst, rj_code_arch_t arch) {
  return pack_sop1(sop1_op_getpc_b64(arch), sdst, 0);
}

/// @brief Encode an s_setpc_b64 instruction for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_setpc_b64(uint16_t ssrc0, rj_code_arch_t arch) {
  return pack_sop1(sop1_op_setpc_b64(arch), 0, ssrc0);
}

/// @brief Encode an s_swappc_b64 instruction for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_swappc_b64(uint16_t sdst, uint16_t ssrc0,
                                                           rj_code_arch_t arch) {
  return pack_sop1(sop1_op_swappc_b64(arch), sdst, ssrc0);
}

/// @brief Encode an s_call_b64 instruction for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_call_b64(uint16_t sdst, int16_t offset_dwords,
                                                         rj_code_arch_t arch) {
  return pack_sopk(sopk_op_call_b64(arch), sdst, static_cast<uint16_t>(offset_dwords));
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

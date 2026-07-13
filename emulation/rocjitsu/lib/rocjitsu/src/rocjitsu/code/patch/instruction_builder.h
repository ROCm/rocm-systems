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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/builders.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/builders.h"
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
// SOPK has the same representation split: its machine field stores the low
// fixed selector, while generated encoding IDs describe primary decode.
inline constexpr uint32_t kSopkEncodingPrefix = 0xB;
inline constexpr uint16_t kScalarPositiveInlineBase = 128;
inline constexpr uint16_t kDelayAluSaluDep1 = 9;
inline constexpr size_t kMaxRecoveredIndirectTransferWords = 6;
inline constexpr size_t kMaxDirectBranchTransferWords = 7;

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
  /// @details GLOBAL segment mode consumes a 64-bit VGPR address pair starting at
  /// `addr`, plus the 64-bit scalar `saddr` base and the signed immediate offset.
  /// Virtual LDS lowering uses either a scalar backing-buffer base with a
  /// zero-extended VGPR LDS offset, or `saddr=null` with the full 64-bit backing
  /// address already materialized in VGPRs when no scalar base can remain live.
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
  [[nodiscard]] static WordPair ds(uint16_t op, uint8_t vdst, uint8_t addr, uint8_t data0 = 0,
                                   uint8_t data1 = 0, uint8_t offset0 = 0, uint8_t offset1 = 0) {
    cdna3::DsMachineInst dst{};
    dst.encoding = 0x36;
    dst.op = op & 0xFF;
    dst.offset0 = offset0;
    dst.offset1 = offset1;
    dst.addr = addr;
    dst.data0 = data0;
    dst.data1 = data1;
    dst.vdst = vdst;
    return encode_pair(dst);
  }

  /// @brief Encode a CDNA3 non-LDS MUBUF instruction.
  [[nodiscard]] static WordPair mubuf(const MubufOperands &src, uint16_t op, uint8_t vdata) {
    cdna3::MubufMachineInst dst{};
    dst.encoding = 0x38;
    dst.op = op & 0x7F;
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
  [[nodiscard]] static WordPair flat_global_load(const FlatGlobalOperands &src, uint16_t op,
                                                 uint8_t vdst) {
    cdna3::FlatMachineInst dst{};
    dst.encoding = 0x37;
    dst.offset = src.signed_offset13 & 0x0FFF;
    dst.pad_12 = (src.signed_offset13 >> 12) & 0x1;
    dst.lds = 0;
    dst.seg = 2;
    dst.sc0 = src.sc0 ? 1 : 0;
    dst.nt = src.nt ? 1 : 0;
    dst.op = op & 0x7F;
    dst.sc1 = src.sc1 ? 1 : 0;
    dst.addr = src.addr;
    dst.saddr = src.saddr;
    dst.acc = src.acc ? 1 : 0;
    dst.vdst = vdst;
    return encode_pair(dst);
  }

  /// @brief Encode a CDNA3 flat/global store using a 64-bit SGPR base plus VGPR offset.
  [[nodiscard]] static WordPair flat_global_store(const FlatGlobalOperands &src, uint16_t op,
                                                  uint8_t data) {
    cdna3::FlatMachineInst dst{};
    dst.encoding = 0x37;
    dst.offset = src.signed_offset13 & 0x0FFF;
    dst.pad_12 = (src.signed_offset13 >> 12) & 0x1;
    dst.lds = 0;
    dst.seg = 2;
    dst.sc0 = src.sc0 ? 1 : 0;
    dst.nt = src.nt ? 1 : 0;
    dst.op = op & 0x7F;
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
  [[nodiscard]] static WordPair flat_scratch_dword(uint16_t op, uint8_t vgpr, uint32_t byte_offset,
                                                   bool is_load) {
    cdna3::FlatScratchMachineInst dst{};
    dst.encoding = 0x37;
    dst.op = op & 0x7F;
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

  /// @brief Encode `s_load_dwordx2` with an immediate byte offset.
  ///
  /// @details CDNA3 SMEM encodes the scalar base register as an SGPR pair index:
  /// `s[4:5]` is encoded as SBASE=2. Virtual-LDS entry prologues use this to
  /// load the 64-bit original kernarg pointer and virtual-LDS runtime state
  /// fields from DBT's kernarg wrapper.
  [[nodiscard]] static WordPair smem_load_dwordx2(uint8_t dst_sgpr, uint8_t sbase_sgpr,
                                                  uint32_t byte_offset) {
    cdna3::SmemMachineInst dst{};
    dst.encoding = 0x30;
    dst.op = cdna3::kSLoadDwordx2Smem;
    dst.sbase = (sbase_sgpr / 2) & 0x3F;
    dst.sdata = dst_sgpr & 0x7F;
    dst.imm = 1;
    dst.offset = byte_offset & 0x1FFFFF;
    return encode_pair(dst);
  }

  /// @brief Encode `s_load_dword` with an immediate byte offset.
  ///
  /// @details Virtual-LDS entry prologues use scalar loads for compact
  /// per-dispatch state such as per-workgroup byte strides. Keep this beside
  /// the `s_load_dwordx2` builder so SMEM bitfield details stay target-local.
  [[nodiscard]] static WordPair smem_load_dword(uint8_t dst_sgpr, uint8_t sbase_sgpr,
                                                uint32_t byte_offset) {
    cdna3::SmemMachineInst dst{};
    dst.encoding = 0x30;
    dst.op = cdna3::kSLoadDwordSmem;
    dst.sbase = (sbase_sgpr / 2) & 0x3F;
    dst.sdata = dst_sgpr & 0x7F;
    dst.imm = 1;
    dst.offset = byte_offset & 0x1FFFFF;
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

/// @brief Encode a CDNA3 VOP2 instruction with a 32-bit literal source.
///
/// @details The source operand is forced to the hardware literal-constant
/// selector (`0xff`) and the literal itself is carried in the second dword.
/// Virtual-LDS address materialization uses this for `v_add_u32_e32` so large DS
/// byte offsets do not require an extra scalar register.
[[nodiscard]] inline std::pair<uint32_t, uint32_t>
build_cdna3_vop2_literal(uint16_t op, uint8_t vdst, uint8_t vsrc1, uint32_t literal) {
  cdna3::Vop2InstLiteralMachineInst dst{};
  dst.src0 = 0xFF;
  dst.vsrc1 = vsrc1;
  dst.vdst = vdst;
  dst.op = op & 0x3F;
  dst.encoding = 0;
  dst.simm32 = literal;

  uint32_t words[2]{};
  std::memcpy(words, &dst, sizeof(dst));
  return {words[0], words[1]};
}

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

/// @brief Build a SOPP word using the generated layout for @p arch.
[[nodiscard]] inline constexpr uint32_t build_sopp_encoding(rj_code_arch_t arch, uint16_t op,
                                                            uint16_t simm16) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_sopp(op, {.simm16 = simm16})[0];
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_sopp(op, {.simm16 = simm16})[0];
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_sopp(op, {.simm16 = simm16})[0];
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_sopp(op, {.simm16 = simm16})[0];
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_sopp(op, {.simm16 = simm16})[0];
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_sopp(op, {.simm16 = simm16})[0];
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_sopp(op, {.simm16 = simm16})[0];
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_sopp(op, {.simm16 = simm16})[0];
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_sopp(op, {.simm16 = simm16})[0];
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::build_sopp(op, {.simm16 = simm16})[0];
  default:
    return pack_sopp(op, simm16);
  }
}

/// @brief Build a SOP1 word using the generated layout for @p arch.
[[nodiscard]] inline constexpr uint32_t build_sop1_encoding(rj_code_arch_t arch, uint16_t op,
                                                            uint16_t sdst, uint16_t ssrc0) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::build_sop1(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
  default:
    return pack_sop1(op, sdst, ssrc0);
  }
}

/// @brief Build a SOP2 word using the generated layout for @p arch.
[[nodiscard]] inline constexpr uint32_t build_sop2_encoding(rj_code_arch_t arch, uint16_t op,
                                                            uint16_t sdst, uint16_t ssrc0,
                                                            uint16_t ssrc1) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                  .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                  .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                  .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                  .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                  .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                  .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                  .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                    .ssrc1 = static_cast<uint8_t>(ssrc1),
                                    .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                  .sdst = static_cast<uint8_t>(sdst)})[0];
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                    .ssrc1 = static_cast<uint8_t>(ssrc1),
                                    .sdst = static_cast<uint8_t>(sdst)})[0];
  default:
    return pack_sop2(op, sdst, ssrc0, ssrc1);
  }
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
[[nodiscard]] inline constexpr std::optional<int16_t> compute_sopp_branch_simm16(uint64_t branch_pc,
                                                                                 uint64_t target) {
  constexpr int64_t kBranchPcBiasBytes = static_cast<int64_t>(sizeof(uint32_t));
  constexpr uint64_t kMaxSignedTarget = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  constexpr uint64_t kMaxSignedBranchPc =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - kBranchPcBiasBytes);
  if (branch_pc > kMaxSignedBranchPc || target > kMaxSignedTarget)
    return std::nullopt;

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

/// @brief Encode an s_branch instruction for the given target ISA.
///
/// @param offset_dwords  Signed offset in dwords from (PC + 4).
/// @param arch           Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t build_s_branch(int16_t offset_dwords, rj_code_arch_t arch) {
  return build_sopp_encoding(arch, sopp_op_branch(arch), static_cast<uint16_t>(offset_dwords));
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
/// relative to the instruction's branch base. For AMDGPU SOPP direct branches,
/// the base is the next instruction and the immediate is a signed dword offset.
/// The function handles unconditional and conditional SOPP direct branches by
/// replacing bits [15:0] of word 0. It returns false when @p inst is not a
/// decoded direct branch, the buffer is empty, or the delta is not representable
/// by SOPP's signed 16-bit dword immediate.
[[nodiscard]] bool patch_pcrel_branch_offset(const Instruction &inst, std::span<uint32_t> words,
                                             int64_t delta_bytes, rj_code_arch_t arch);

/// @brief Build a PC-relative long branch through s_getpc_b64/s_setpc_b64.
///
/// @details This clobbers @p sgpr_pair and @p sgpr_pair + 1, so callers must
/// only use a pair known to be dead at the branch site.
[[nodiscard]] inline std::vector<uint32_t>
build_s_setpc_long_branch(uint64_t getpc_pc, uint64_t target, uint16_t sgpr_pair) {
  if (sgpr_pair >= 127 ||
      getpc_pc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - 4) ||
      target > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return {};

  const int64_t delta = static_cast<int64_t>(target) - (static_cast<int64_t>(getpc_pc) + 4);
  const auto delta_bits = static_cast<uint64_t>(delta);
  const uint32_t delta_lo = static_cast<uint32_t>(delta_bits & 0xFFFF'FFFFu);
  const uint32_t delta_hi = static_cast<uint32_t>(delta_bits >> 32);

  constexpr uint8_t kOpSAddCoU32 = 0;
  constexpr uint8_t kOpSAddCoCiU32 = 4;
  constexpr uint8_t kOpSGetPcB64 = 71;
  constexpr uint8_t kOpSSetPcB64 = 72;
  return {
      pack_sop1(kOpSGetPcB64, sgpr_pair, 0),
      pack_sop2(kOpSAddCoU32, sgpr_pair, sgpr_pair, 255),
      delta_lo,
      pack_sop2(kOpSAddCoCiU32, static_cast<uint16_t>(sgpr_pair + 1u),
                static_cast<uint16_t>(sgpr_pair + 1u), 255),
      delta_hi,
      pack_sop1(kOpSSetPcB64, 0, sgpr_pair),
  };
}

/// @brief Build a PC-relative long branch that preserves the incoming SCC.
///
/// @details The ordinary long branch uses scalar add-with-carry instructions to
/// materialize the target PC, which clobber SCC. Use this form when a cave body
/// must return with branch-like semantics, preserving the SCC value produced by
/// the cave body for any later scalar condition or carry consumer.
[[nodiscard]] inline std::vector<uint32_t>
build_s_setpc_long_branch_preserving_scc(uint64_t getpc_pc, uint64_t target, uint16_t sgpr_pair,
                                         uint16_t scc_sgpr) {
  if (sgpr_pair >= 127 || scc_sgpr >= 127 || scc_sgpr == sgpr_pair || scc_sgpr == sgpr_pair + 1u ||
      getpc_pc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - 8) ||
      target > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return {};

  const uint64_t actual_getpc_pc = getpc_pc + sizeof(uint32_t);
  const int64_t delta = static_cast<int64_t>(target) - (static_cast<int64_t>(actual_getpc_pc) + 4);
  const auto delta_bits = static_cast<uint64_t>(delta);
  const uint32_t delta_lo = static_cast<uint32_t>(delta_bits & 0xFFFF'FFFFu);
  const uint32_t delta_hi = static_cast<uint32_t>(delta_bits >> 32);

  constexpr uint8_t kOpSAddCoU32 = 0;
  constexpr uint8_t kOpSAddCoCiU32 = 4;
  constexpr uint8_t kOpSCselectB32 = 48;
  constexpr uint8_t kOpSCmpLgU32 = 7;
  constexpr uint8_t kOpSGetPcB64 = 71;
  constexpr uint8_t kOpSSetPcB64 = 72;
  return {
      pack_sop2(kOpSCselectB32, scc_sgpr, scalar_positive_inline_u32(1),
                scalar_positive_inline_u32(0)),
      pack_sop1(kOpSGetPcB64, sgpr_pair, 0),
      pack_sop2(kOpSAddCoU32, sgpr_pair, sgpr_pair, 255),
      delta_lo,
      pack_sop2(kOpSAddCoCiU32, static_cast<uint16_t>(sgpr_pair + 1u),
                static_cast<uint16_t>(sgpr_pair + 1u), 255),
      delta_hi,
      pack_sopc(kOpSCmpLgU32, scc_sgpr, scalar_positive_inline_u32(0)),
      pack_sop1(kOpSSetPcB64, 0, sgpr_pair),
  };
}

/// @brief Encode an s_nop instruction for the given target ISA.
///
/// @param cycles  Number of additional stall cycles (0-based).
/// @param arch    Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t
build_s_nop(uint16_t cycles = 0, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  return build_sopp_encoding(arch, sopp_op_nop(arch), cycles);
}

/// @brief Encode an s_endpgm instruction for the given target ISA.
///
/// @param arch    Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t build_s_endpgm(rj_code_arch_t arch) {
  return build_sopp_encoding(arch, sopp_op_endpgm(arch), 0);
}

/// @brief Encode an s_endpgm instruction for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_endpgm(rj_code_arch_t arch) {
  return pack_sopp(sopp_op_endpgm(arch), 0);
}

/// @brief Encode an s_trap instruction for the given target ISA.
///
/// @details The immediate is a trap code, not a printable message. Runtime DBT
/// uses a rocjitsu-specific value for skipped-kernel stubs so a surfaced trap
/// code can be distinguished from guest code traps.
[[nodiscard]] inline constexpr uint32_t build_s_trap(rj_code_arch_t arch, uint16_t simm16 = 0) {
  return pack_sopp(sopp_op_trap(arch), simm16);
}

/// @brief Encode s_delay_alu for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_delay_alu(uint16_t simm16, rj_code_arch_t arch) {
  return build_sopp_encoding(arch, sopp_op_delay_alu(arch), simm16);
}

/// @brief Encode s_wait_alu for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_wait_alu(uint16_t simm16, rj_code_arch_t) {
  constexpr uint8_t kSoppWaitAlu = 8;
  return pack_sopp(kSoppWaitAlu, simm16);
}

/// @brief Encode s_mov_b32 for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_mov_b32(uint16_t sdst, uint16_t ssrc0,
                                                        rj_code_arch_t arch) {
  return build_sop1_encoding(arch, sop1_op_mov_b32(arch), sdst, ssrc0);
}

/// @brief Encode s_lshl_b32 for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_lshl_b32(uint16_t sdst, uint16_t ssrc0,
                                                         uint16_t ssrc1, rj_code_arch_t arch) {
  return build_sop2_encoding(arch, sop2_op_lshl_b32(arch), sdst, ssrc0, ssrc1);
}

/// @brief Encode s_lshr_b32 for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_lshr_b32(uint16_t sdst, uint16_t ssrc0,
                                                         uint16_t ssrc1, rj_code_arch_t arch) {
  return build_sop2_encoding(arch, sop2_op_lshr_b32(arch), sdst, ssrc0, ssrc1);
}

} // namespace rocjitsu

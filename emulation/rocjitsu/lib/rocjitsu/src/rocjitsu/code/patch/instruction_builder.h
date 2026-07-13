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
inline constexpr uint32_t kSopcEncodingPrefix = 0x17E;
inline constexpr uint16_t kScalarPositiveInlineBase = 128;
inline constexpr uint16_t kDelayAluSaluDep1 = 9;
inline constexpr uint16_t kWaitAluDepctrVaSdst0 = 0xF19F;
inline constexpr uint16_t kWaitAluDepctrVaVdst0 = 0x0F9F;
inline constexpr uint16_t kWaitAluDepctrVaVcc0 = 0xFF9D;
inline constexpr uint16_t kWaitAluDepctrVmVsrc0 = 0xFF83;
inline constexpr uint16_t kWaitAluDepctrVaVdstVmVsrc0 =
    kWaitAluDepctrVaVdst0 & kWaitAluDepctrVmVsrc0;
inline constexpr uint16_t kWaitAluDepctrSaSdst0 = 0xFF9E;

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

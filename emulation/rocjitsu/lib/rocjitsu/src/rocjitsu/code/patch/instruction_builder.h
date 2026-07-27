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
#include "util/except.h"

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
// SOPK has the same representation split: its machine field stores the low
// fixed selector, while generated encoding IDs describe primary decode.
inline constexpr uint32_t kSopkEncodingPrefix = 0xB;
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
    throw util::UnimplementedInst("SOPP builder for target architecture");
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
    throw util::UnimplementedInst("SOP1 builder for target architecture");
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
    throw util::UnimplementedInst("SOP2 builder for target architecture");
  }
}

/// @brief Pack a SOPC instruction word from its constituent fields.
///
/// SOPC compares two scalar sources and writes only SCC; there is no sdst.
[[nodiscard]] inline constexpr uint32_t pack_sopc(uint32_t op, uint32_t ssrc0, uint32_t ssrc1) {
  return (kSopcEncodingPrefix << 23) | ((op & 0x7Fu) << 16) | ((ssrc1 & 0xFFu) << 8) |
         (ssrc0 & 0xFFu);
}

/// @brief Build a SOPC word using the generated layout for @p arch.
[[nodiscard]] inline constexpr uint32_t build_sopc_encoding(rj_code_arch_t arch, uint16_t op,
                                                            uint16_t ssrc0, uint16_t ssrc1) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::build_sopc(
        op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
  default:
    return pack_sopc(op, ssrc0, ssrc1);
  }
}

/// @brief Pack a SOPK instruction word from its constituent fields.
[[nodiscard]] inline constexpr uint32_t pack_sopk(uint32_t op, uint32_t sdst, uint16_t simm16) {
  return (kSopkEncodingPrefix << 28) | ((op & 0x1Fu) << 23) | ((sdst & 0x7Fu) << 16) | simm16;
}

/// @brief Build a SOPK word using the generated layout for @p arch.
[[nodiscard]] inline constexpr uint32_t build_sopk_encoding(rj_code_arch_t arch, uint16_t op,
                                                            uint16_t sdst, uint16_t simm16) {
#define ROCJITSU_BUILD_SOPK(isa)                                                                   \
  return isa::build_sopk(op, {.simm16 = simm16, .sdst = static_cast<uint8_t>(sdst)})[0]
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    ROCJITSU_BUILD_SOPK(cdna1);
  case ROCJITSU_CODE_ARCH_CDNA2:
    ROCJITSU_BUILD_SOPK(cdna2);
  case ROCJITSU_CODE_ARCH_CDNA3:
    ROCJITSU_BUILD_SOPK(cdna3);
  case ROCJITSU_CODE_ARCH_CDNA4:
    ROCJITSU_BUILD_SOPK(cdna4);
  case ROCJITSU_CODE_ARCH_RDNA1:
    ROCJITSU_BUILD_SOPK(rdna1);
  case ROCJITSU_CODE_ARCH_RDNA2:
    ROCJITSU_BUILD_SOPK(rdna2);
  case ROCJITSU_CODE_ARCH_RDNA3:
    ROCJITSU_BUILD_SOPK(rdna3);
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    ROCJITSU_BUILD_SOPK(rdna3_5);
  case ROCJITSU_CODE_ARCH_RDNA4:
    ROCJITSU_BUILD_SOPK(rdna4);
  case ROCJITSU_CODE_ARCH_GFX1250:
    ROCJITSU_BUILD_SOPK(gfx1250);
  default:
    throw util::UnimplementedInst("SOPK builder for target architecture");
  }
#undef ROCJITSU_BUILD_SOPK
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

// Expand the common architecture switch once for instructions whose mnemonic
// exists on every generated target. Each helper still names the generated raw
// opcode, so opcode renumbering is picked up by ISA regeneration.
#define ROCJITSU_COMMON_OPCODE_CASES(opcode)                                                       \
  case ROCJITSU_CODE_ARCH_CDNA1:                                                                   \
    return cdna1::opcode;                                                                          \
  case ROCJITSU_CODE_ARCH_CDNA2:                                                                   \
    return cdna2::opcode;                                                                          \
  case ROCJITSU_CODE_ARCH_CDNA3:                                                                   \
    return cdna3::opcode;                                                                          \
  case ROCJITSU_CODE_ARCH_CDNA4:                                                                   \
    return cdna4::opcode;                                                                          \
  case ROCJITSU_CODE_ARCH_RDNA1:                                                                   \
    return rdna1::opcode;                                                                          \
  case ROCJITSU_CODE_ARCH_RDNA2:                                                                   \
    return rdna2::opcode;                                                                          \
  case ROCJITSU_CODE_ARCH_RDNA3:                                                                   \
    return rdna3::opcode;                                                                          \
  case ROCJITSU_CODE_ARCH_RDNA3_5:                                                                 \
    return rdna3_5::opcode;                                                                        \
  case ROCJITSU_CODE_ARCH_RDNA4:                                                                   \
    return rdna4::opcode;                                                                          \
  case ROCJITSU_CODE_ARCH_GFX1250:                                                                 \
    return gfx1250::opcode

/// @brief Get the s_branch opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_branch(rj_code_arch_t arch) {
  switch (arch) {
    ROCJITSU_COMMON_OPCODE_CASES(kSBranchSopp);
  default:
    throw util::UnimplementedInst("s_branch for target architecture");
  }
}

/// @brief Get the s_endpgm opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_endpgm(rj_code_arch_t arch) {
  switch (arch) {
    ROCJITSU_COMMON_OPCODE_CASES(kSEndpgmSopp);
  default:
    throw util::UnimplementedInst("s_endpgm for target architecture");
  }
}

/// @brief Get the s_trap opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_trap(rj_code_arch_t arch) {
  switch (arch) {
    ROCJITSU_COMMON_OPCODE_CASES(kSTrapSopp);
  default:
    throw util::UnimplementedInst("s_trap for target architecture");
  }
}

/// @brief Get the s_nop opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopp_op_nop(rj_code_arch_t arch) {
  switch (arch) {
    ROCJITSU_COMMON_OPCODE_CASES(kSNopSopp);
  default:
    throw util::UnimplementedInst("s_nop for target architecture");
  }
}

/// @brief Get the s_getpc_b64 SOP1 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop1_op_getpc_b64(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSGetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSGetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSGetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSGetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSGetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSGetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSGetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSGetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSGetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSGetPcI64Sop1;
  default:
    throw util::UnimplementedInst("s_getpc for target architecture");
  }
}

/// @brief Get the s_setpc_b64 SOP1 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop1_op_setpc_b64(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSSetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSSetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSSetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSSetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSSetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSSetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSSetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSSetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSSetPcB64Sop1;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSSetPcI64Sop1;
  default:
    throw util::UnimplementedInst("s_setpc for target architecture");
  }
}

/// @brief Get the s_swappc_b64 SOP1 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop1_op_swappc_b64(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSSwapPcB64Sop1;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSSwapPcB64Sop1;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSSwapPcB64Sop1;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSSwapPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSSwapPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSSwapPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSSwapPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSSwapPcB64Sop1;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSSwapPcB64Sop1;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSSwapPcI64Sop1;
  default:
    throw util::UnimplementedInst("s_swappc for target architecture");
  }
}

/// @brief Get the s_call_b64 SOPK opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sopk_op_call_b64(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::kSCallB64Sopk;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::kSCallB64Sopk;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::kSCallB64Sopk;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::kSCallB64Sopk;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::kSCallB64Sopk;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::kSCallB64Sopk;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::kSCallB64Sopk;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::kSCallB64Sopk;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::kSCallB64Sopk;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::kSCallI64Sopk;
  default:
    throw util::UnimplementedInst("s_call_b64 for target architecture");
  }
}

/// @brief Get the s_lshl_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop2_op_lshl_b32(rj_code_arch_t arch) {
  switch (arch) {
    ROCJITSU_COMMON_OPCODE_CASES(kSLshlB32Sop2);
  default:
    throw util::UnimplementedInst("s_lshl_b32 for target architecture");
  }
}

/// @brief Get the s_lshr_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop2_op_lshr_b32(rj_code_arch_t arch) {
  switch (arch) {
    ROCJITSU_COMMON_OPCODE_CASES(kSLshrB32Sop2);
  default:
    throw util::UnimplementedInst("s_lshr_b32 for target architecture");
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
    throw util::UnimplementedInst("s_delay_alu for target architecture");
  }
}

/// @brief Get the s_mov_b32 opcode for a target ISA.
[[nodiscard]] inline constexpr uint32_t sop1_op_mov_b32(rj_code_arch_t arch) {
  switch (arch) {
    ROCJITSU_COMMON_OPCODE_CASES(kSMovB32Sop1);
  default:
    throw util::UnimplementedInst("s_mov_b32 for target architecture");
  }
}

/// @brief SOP2 opcode for s_cselect_b32 on @p arch.
[[nodiscard]] inline constexpr uint32_t sop2_op_cselect_b32(rj_code_arch_t arch) {
  switch (arch) {
    ROCJITSU_COMMON_OPCODE_CASES(kSCselectB32Sop2);
  default:
    throw util::UnimplementedInst("s_cselect_b32 for target architecture");
  }
}

/// @brief SOPC opcode for s_cmp_lg_u32 on @p arch.
[[nodiscard]] inline constexpr uint32_t sopc_op_cmp_lg_u32(rj_code_arch_t arch) {
  switch (arch) {
    ROCJITSU_COMMON_OPCODE_CASES(kSCmpLgU32Sopc);
  default:
    throw util::UnimplementedInst("s_cmp_lg_u32 for target architecture");
  }
}

#undef ROCJITSU_COMMON_OPCODE_CASES

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
  return build_sop1_encoding(arch, sop1_op_getpc_b64(arch), sdst, 0);
}

/// @brief Encode an s_setpc_b64 instruction for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_setpc_b64(uint16_t ssrc0, rj_code_arch_t arch) {
  return build_sop1_encoding(arch, sop1_op_setpc_b64(arch), 0, ssrc0);
}

/// @brief Encode an s_swappc_b64 instruction for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_swappc_b64(uint16_t sdst, uint16_t ssrc0,
                                                           rj_code_arch_t arch) {
  return build_sop1_encoding(arch, sop1_op_swappc_b64(arch), sdst, ssrc0);
}

/// @brief Encode an s_call_b64 instruction for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_call_b64(uint16_t sdst, int16_t offset_dwords,
                                                         rj_code_arch_t arch) {
  return build_sopk_encoding(arch, sopk_op_call_b64(arch), sdst,
                             static_cast<uint16_t>(offset_dwords));
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
  return build_sopp_encoding(arch, sopp_op_nop(arch), cycles);
}

/// @brief Encode an s_endpgm instruction for the given target ISA.
///
/// @param arch    Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t build_s_endpgm(rj_code_arch_t arch) {
  return build_sopp_encoding(arch, sopp_op_endpgm(arch), 0);
}

/// @brief Encode an s_trap instruction for the given target ISA.
///
/// @details The immediate is a caller-defined trap code, not a printable message.
[[nodiscard]] inline constexpr uint32_t build_s_trap(rj_code_arch_t arch, uint16_t simm16 = 0) {
  return build_sopp_encoding(arch, sopp_op_trap(arch), simm16);
}

/// @brief Encode s_delay_alu for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_delay_alu(uint16_t simm16, rj_code_arch_t arch) {
  return build_sopp_encoding(arch, sopp_op_delay_alu(arch), simm16);
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

/// @brief Encode s_add_u32 for the given target ISA (SOP2 opcode 0, all gens).
[[nodiscard]] inline constexpr uint32_t build_s_add_u32(uint16_t sdst, uint16_t ssrc0,
                                                        uint16_t ssrc1, rj_code_arch_t arch) {
  constexpr uint16_t kSop2AddU32 = 0;
  return build_sop2_encoding(arch, kSop2AddU32, sdst, ssrc0, ssrc1);
}

/// @brief Encode s_addc_u32 for the given target ISA (SOP2 opcode 4, all gens).
[[nodiscard]] inline constexpr uint32_t build_s_addc_u32(uint16_t sdst, uint16_t ssrc0,
                                                         uint16_t ssrc1, rj_code_arch_t arch) {
  constexpr uint16_t kSop2AddcU32 = 4;
  return build_sop2_encoding(arch, kSop2AddcU32, sdst, ssrc0, ssrc1);
}

/// @brief Encode s_cselect_b32 for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_cselect_b32(uint16_t sdst, uint16_t ssrc0,
                                                            uint16_t ssrc1, rj_code_arch_t arch) {
  return build_sop2_encoding(arch, sop2_op_cselect_b32(arch), sdst, ssrc0, ssrc1);
}

/// @brief Encode s_cmp_lg_u32 for the given target ISA.
[[nodiscard]] inline constexpr uint32_t build_s_cmp_lg_u32(uint16_t ssrc0, uint16_t ssrc1,
                                                           rj_code_arch_t arch) {
  return build_sopc_encoding(arch, sopc_op_cmp_lg_u32(arch), ssrc0, ssrc1);
}

} // namespace rocjitsu

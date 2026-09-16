// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file smem_builders.h
/// @brief ISA-dispatched scalar-memory (SMEM) instruction builders.

#pragma once

#include <array>
#include <cstdint>

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"
#include "util/except.h"

namespace rocjitsu {

/// @brief Highest SGPR index the 6-bit SBASE field can name, given it holds the
///        index halved.
inline constexpr uint16_t kMaxSmemSbase = 126;

/// @brief Highest SGPR index the 7-bit SDATA field can name.
inline constexpr uint16_t kMaxSmemSdata = 127;

/// @brief Largest forward byte offset a 21-bit SMEM immediate can name.
///
/// @details CDNA1-4 and RDNA1-3.5 place the offset at bit 32 with width 21
/// (`generated/<gen>/builders.h`, `build_smem`) and read it as signed, so a
/// forward offset gets the low 20 bits. The address calculation sign-extends it:
/// `shared/addr_calc_scalar.h` does `(inst.offset << 11) >> 11`.
inline constexpr uint32_t kSmem21BitMaxByteOffset = 0x0FFFFFu;

/// @brief Largest forward byte offset a 24-bit SMEM immediate can name.
///
/// @details RDNA4 and CDNA5 widen the field to 24 bits and rename it `ioffset`.
/// Still signed: `rdna4/addr_calc.cpp` and `cdna5/addr_calc.h` both sign-extend
/// with `(ioffset << 8) >> 8`.
inline constexpr uint32_t kSmem24BitMaxByteOffset = 0x7FFFFFu;

/// @brief Largest forward byte offset @p arch's SMEM immediate field can name.
///
/// @throws util::UnimplementedInst for a non-AMDGPU architecture.
[[nodiscard]] inline constexpr uint32_t max_smem_byte_offset(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return kSmem21BitMaxByteOffset;
  case ROCJITSU_CODE_ARCH_CDNA5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return kSmem24BitMaxByteOffset;
  default:
    throw util::UnimplementedInst("SMEM offset range for target architecture");
  }
}

/// @brief Encode `s_load_dwordx2 s[sdst:sdst+1], s[sbase:sbase+1], @p byte_offset`
///        (spelled `s_load_b64` from RDNA3 on).
///
/// @param sdst Base of the destination SGPR pair. Must be even.
/// @param sbase Base of the 64-bit address SGPR pair. Must be even: the encoding
///        carries the index halved and has no bit for the low one.
/// @param byte_offset Forward immediate byte offset from the base address. Must
///        not exceed @ref max_smem_byte_offset for @p arch.
///
/// @details Three things the caller should not have to know. SBASE is the
/// register index halved on every generation; the decoder multiplies it back.
/// "No SGPR offset" has two spellings: CDNA1-4 gate the register and the
/// immediate with independent SOFFSET_EN and IMM bits, so an immediate-only load
/// clears the former and sets the latter, while RDNA has neither bit and instead
/// carries an always-present SOFFSET field that must name NULL. The NULL code
/// itself moved (125 on RDNA1/2, 124 from RDNA3 on), so each case takes
/// OPR_SMEM_OFFSET_NULL from its own generation's table for this field rather
/// than a shared constant or the generic scalar-source table.
///
/// @throws util::InvalidInst if either pair base is odd or out of field range, or
///         if @p byte_offset is out of range.
/// @throws util::UnimplementedInst for a non-AMDGPU architecture.
[[nodiscard]] inline std::array<uint32_t, 2>
build_s_load_dwordx2(uint16_t sdst, uint16_t sbase, uint32_t byte_offset, rj_code_arch_t arch) {
  // Both operands name a 64-bit pair, so both bases must be even. The encoded
  // fields are narrower than uint16_t and the generated set_field masks rather
  // than rejects, so an out-of-range index would silently become a different
  // register: SBASE is 6 bits holding the index halved, SDATA is 7 bits.
  if ((sbase % 2) != 0)
    throw util::InvalidInst("SMEM address SGPR pair base must be even");
  if ((sdst % 2) != 0)
    throw util::InvalidInst("SMEM destination SGPR pair base must be even");
  if (sbase > kMaxSmemSbase)
    throw util::InvalidInst("SMEM address SGPR index exceeds the SBASE field");
  if (sdst > kMaxSmemSdata)
    throw util::InvalidInst("SMEM destination SGPR index exceeds the SDATA field");
  if (byte_offset > max_smem_byte_offset(arch))
    throw util::InvalidInst("SMEM immediate byte offset out of range for target architecture");

  const auto base = static_cast<uint8_t>(sbase / 2);
  const auto data = static_cast<uint8_t>(sdst);
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_smem(
        cdna1::kSLoadDwordx2Smem,
        {.sbase = base, .sdata = data, .soffset_en = 0, .imm = 1, .offset = byte_offset});
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_smem(
        cdna2::kSLoadDwordx2Smem,
        {.sbase = base, .sdata = data, .soffset_en = 0, .imm = 1, .offset = byte_offset});
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_smem(
        cdna3::kSLoadDwordx2Smem,
        {.sbase = base, .sdata = data, .soffset_en = 0, .imm = 1, .offset = byte_offset});
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_smem(
        cdna4::kSLoadDwordx2Smem,
        {.sbase = base, .sdata = data, .soffset_en = 0, .imm = 1, .offset = byte_offset});
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_smem(rdna1::kSLoadDwordx2Smem,
                             {.sbase = base,
                              .sdata = data,
                              .offset = byte_offset,
                              .soffset = static_cast<uint8_t>(rdna1::OPR_SMEM_OFFSET_NULL)});
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_smem(rdna2::kSLoadDwordx2Smem,
                             {.sbase = base,
                              .sdata = data,
                              .offset = byte_offset,
                              .soffset = static_cast<uint8_t>(rdna2::OPR_SMEM_OFFSET_NULL)});
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_smem(rdna3::kSLoadB64Smem,
                             {.sbase = base,
                              .sdata = data,
                              .offset = byte_offset,
                              .soffset = static_cast<uint8_t>(rdna3::OPR_SMEM_OFFSET_NULL)});
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_smem(rdna3_5::kSLoadB64Smem,
                               {.sbase = base,
                                .sdata = data,
                                .offset = byte_offset,
                                .soffset = static_cast<uint8_t>(rdna3_5::OPR_SMEM_OFFSET_NULL)});
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_smem(rdna4::kSLoadB64Smem,
                             {.sbase = base,
                              .sdata = data,
                              .ioffset = byte_offset,
                              .soffset = static_cast<uint8_t>(rdna4::OPR_SMEM_OFFSET_NULL)});
  case ROCJITSU_CODE_ARCH_CDNA5:
    // scale_offset is CDNA5-only and left 0: this is a plain byte offset, not an
    // offset scaled by the load's data width.
    return cdna5::build_smem(cdna5::kSLoadB64Smem,
                             {.sbase = base,
                              .sdata = data,
                              .ioffset = byte_offset,
                              .scale_offset = 0,
                              .soffset = static_cast<uint8_t>(cdna5::OPR_SMEM_OFFSET_NULL)});
  default:
    throw util::UnimplementedInst("s_load_dwordx2 for target architecture");
  }
}

/// @brief Encode a wait for outstanding scalar-memory loads to land in their SGPRs.
///
/// @details Through RDNA3.5 the monolithic `s_waitcnt 0` covers lgkmcnt, which is
/// where scalar loads retire. GFX12 (RDNA4, CDNA5) splits the counters and gives
/// scalar loads their own, `s_wait_kmcnt 0`; CDNA5 has no `s_waitcnt` opcode at
/// all. Neither of spill_builders.h's load waits is this counter:
/// build_wait_loads_complete drains LOADCNT, which orders VMEM, not SMEM.
///
/// @note Scalar loads are asynchronous. Emit this between the load and the first
///   read of its destination.
///
/// @throws util::UnimplementedInst for a non-AMDGPU architecture.
[[nodiscard]] inline uint32_t build_wait_scalar_loads_complete(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return build_sopp_encoding(arch, cdna1::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_CDNA2:
    return build_sopp_encoding(arch, cdna2::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sopp_encoding(arch, cdna3::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sopp_encoding(arch, cdna4::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_RDNA1:
    return build_sopp_encoding(arch, rdna1::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_RDNA2:
    return build_sopp_encoding(arch, rdna2::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_RDNA3:
    return build_sopp_encoding(arch, rdna3::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return build_sopp_encoding(arch, rdna3_5::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sopp_encoding(arch, rdna4::kSWaitKmcntSopp, 0);
  case ROCJITSU_CODE_ARCH_CDNA5:
    return build_sopp_encoding(arch, cdna5::kSWaitKmcntSopp, 0);
  default:
    throw util::UnimplementedInst("scalar-load wait for target architecture");
  }
}

} // namespace rocjitsu

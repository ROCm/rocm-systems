// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file scalar_builders.h
/// @brief Scalar-side encoders that the shared builders in code/builders/ do
///        not cover, for hand-built DBI simulator test code on the targets
///        DbiSim runs (CDNA3, CDNA4, RDNA4).

#pragma once

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"

#include "util/except.h"

#include <cstdint>

namespace rocjitsu {
namespace test {

/// @brief The scalar source operand code that selects a trailing 32-bit literal.
inline constexpr uint16_t kScalarSrcLiteral = rdna4::OPR_SSRC_SRC_LITERAL;
static_assert(cdna3::OPR_SSRC_SRC_LITERAL == kScalarSrcLiteral &&
              cdna4::OPR_SSRC_SRC_LITERAL == kScalarSrcLiteral);

/// @brief Encode `v_readfirstlane_b32 s<sdst>, v<vsrc>`.
[[nodiscard]] inline uint32_t build_v_readfirstlane_b32(uint8_t sdst, uint8_t vsrc,
                                                        rj_code_arch_t arch) {
  // VOP1 src0 uses the same 9-bit operand code as VOP3, and vdst names the SGPR.
  const uint16_t src0 = vop3_vgpr_src(vsrc);
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_vop1(cdna3::kVReadfirstlaneB32Vop1, {.src0 = src0, .vdst = sdst}).front();
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_vop1(cdna4::kVReadfirstlaneB32Vop1, {.src0 = src0, .vdst = sdst}).front();
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_vop1(rdna4::kVReadfirstlaneB32Vop1, {.src0 = src0, .vdst = sdst}).front();
  default:
    throw util::UnimplementedInst("v_readfirstlane_b32 for target architecture");
  }
}

/// @brief Encode `s_bcnt1_i32_b64 s<sdst>, s[ssrc0:ssrc0+1]`.
[[nodiscard]] inline uint32_t build_s_bcnt1_i32_b64(uint16_t sdst, uint16_t ssrc0,
                                                    rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sop1_encoding(arch, cdna3::kSBcnt1I32B64Sop1, sdst, ssrc0);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sop1_encoding(arch, cdna4::kSBcnt1I32B64Sop1, sdst, ssrc0);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sop1_encoding(arch, rdna4::kSBcnt1I32B64Sop1, sdst, ssrc0);
  default:
    throw util::UnimplementedInst("s_bcnt1_i32_b64 for target architecture");
  }
}

/// @brief Encode the index of the lowest set bit of s[ssrc0:ssrc0+1] into
///        s<sdst>, or -1 when none is set.
/// @details `s_ff1_i32_b64` on CDNA; RDNA4 renames it `s_ctz_i32_b64`. Both store
/// -1 for a zero input.
[[nodiscard]] inline uint32_t build_s_first_set_bit_b64(uint16_t sdst, uint16_t ssrc0,
                                                        rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sop1_encoding(arch, cdna3::kSFf1I32B64Sop1, sdst, ssrc0);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sop1_encoding(arch, cdna4::kSFf1I32B64Sop1, sdst, ssrc0);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sop1_encoding(arch, rdna4::kSCtzI32B64Sop1, sdst, ssrc0);
  default:
    throw util::UnimplementedInst("s_ff1_i32_b64 for target architecture");
  }
}

/// @brief Encode `s_and_b32 s<sdst>, ssrc0, ssrc1`.
[[nodiscard]] inline uint32_t build_s_and_b32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1,
                                              rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sop2_encoding(arch, cdna3::kSAndB32Sop2, sdst, ssrc0, ssrc1);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sop2_encoding(arch, cdna4::kSAndB32Sop2, sdst, ssrc0, ssrc1);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sop2_encoding(arch, rdna4::kSAndB32Sop2, sdst, ssrc0, ssrc1);
  default:
    throw util::UnimplementedInst("s_and_b32 for target architecture");
  }
}

/// @brief Encode `s_cbranch_scc1` when @p scc1, else `s_cbranch_scc0`, branching
///        @p offset_dwords from the next instruction.
[[nodiscard]] inline uint32_t build_s_cbranch_scc(bool scc1, int16_t offset_dwords,
                                                  rj_code_arch_t arch) {
  const auto offset = static_cast<uint16_t>(offset_dwords);
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sopp_encoding(arch, scc1 ? cdna3::kSCbranchScc1Sopp : cdna3::kSCbranchScc0Sopp,
                               offset);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sopp_encoding(arch, scc1 ? cdna4::kSCbranchScc1Sopp : cdna4::kSCbranchScc0Sopp,
                               offset);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sopp_encoding(arch, scc1 ? rdna4::kSCbranchScc1Sopp : rdna4::kSCbranchScc0Sopp,
                               offset);
  default:
    throw util::UnimplementedInst("s_cbranch_scc for target architecture");
  }
}

} // namespace test
} // namespace rocjitsu

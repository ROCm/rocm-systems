// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vector_builders.h
/// @brief ISA-dispatched DBI vector-ALU instruction builders.
///
/// @details The plain VALU ops the DBI patcher emits outside the spill bracket.
/// instruction_builder.h is scalar (SOP*) by construction and spill_builders.h
/// covers only the lane-bridge, scratch, and waitcnt encoders the spill bracket
/// needs; neither is the home for a VALU op emitted on its own.
///
/// Same two-layer shape as instruction_builder.h: each build_* takes an
/// rj_code_arch_t and delegates to that generation's generated builder, so an
/// opcode or field layout that moves between generations is picked up from the
/// generated tables rather than hard-coded here.

#pragma once

#include <array>
#include <cstdint>

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "util/diagnostic.h"
#include "util/result.h"

namespace rocjitsu {

/// @brief The src0 code selecting a trailing 32-bit literal, stable across every
///        AMDGPU generation modeled here (each generation's
///        OPR_SRC_SRC_LITERAL); pinned against the generated tables by test.
inline constexpr uint16_t kVectorSrcLiteral = cdna1::OPR_SRC_SRC_LITERAL;

/// @brief Encode v_mov_b32 @p vdst, @p imm as a VOP1 word plus its literal word.
///
/// The VOP1 src0 field names the literal constant rather than carrying the
/// value, so the immediate travels in a second word. Emitting the pair as one
/// unit keeps the caller from having to know that.
///
/// @returns Failure when the target ISA has no VOP1 encoding.
/// @param error Optional caller-owned diagnostic emitter, used only on failure.
///
/// @note On gfx1250 an encoded @p vdst names a register within the bank
///   MODE.VGPR_MSB selects, not a fixed physical VGPR. That constrains callers
///   that must name a specific register, not this encoder
inline util::FailureOr<std::array<uint32_t, 2>>
build_v_mov_b32_imm(uint16_t vdst, uint32_t imm, rj_code_arch_t arch,
                    const util::DiagnosticEmitter *error = nullptr) {
  const uint8_t dst = static_cast<uint8_t>(vdst);
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return std::array<uint32_t, 2>{
        cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst}).front(),
        imm};
  case ROCJITSU_CODE_ARCH_CDNA2:
    return std::array<uint32_t, 2>{
        cdna2::build_vop1(cdna2::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst}).front(),
        imm};
  case ROCJITSU_CODE_ARCH_CDNA3:
    return std::array<uint32_t, 2>{
        cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst}).front(),
        imm};
  case ROCJITSU_CODE_ARCH_CDNA4:
    return std::array<uint32_t, 2>{
        cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst}).front(),
        imm};
  case ROCJITSU_CODE_ARCH_CDNA5:
    return std::array<uint32_t, 2>{
        cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst}).front(),
        imm};
  case ROCJITSU_CODE_ARCH_RDNA1:
    return std::array<uint32_t, 2>{
        rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst}).front(),
        imm};
  case ROCJITSU_CODE_ARCH_RDNA2:
    return std::array<uint32_t, 2>{
        rdna2::build_vop1(rdna2::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst}).front(),
        imm};
  case ROCJITSU_CODE_ARCH_RDNA3:
    return std::array<uint32_t, 2>{
        rdna3::build_vop1(rdna3::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst}).front(),
        imm};
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return std::array<uint32_t, 2>{
        rdna3_5::build_vop1(rdna3_5::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst})
            .front(),
        imm};
  case ROCJITSU_CODE_ARCH_RDNA4:
    return std::array<uint32_t, 2>{
        rdna4::build_vop1(rdna4::kVMovB32Vop1, {.src0 = kVectorSrcLiteral, .vdst = dst}).front(),
        imm};
  default:
    // RV32I/RV64I and ROCJITSU_CODE_ARCH_INVALID reach here: not AMDGPU, so
    // there is no VOP1 encoding to pick.
    if (error != nullptr)
      return error->emit() << "unsupported target architecture for v_mov_b32 with a literal";
    return util::Result::failure();
  }
}

/// @brief Encode v_mov_b32 @p vdst, @p src0 as a single VOP1 word.
///
/// The register-sourced counterpart of build_v_mov_b32_imm. @p src0 is a VOP1
/// src operand code, which for an SGPR is the register index itself (the same
/// raw form the lane-bridge builders take).
///
/// Broadcasting a scalar this way is how the trampoline hands a probe a value it
/// holds in an SGPR rather than as a build-time constant.
///
/// @returns Failure when the target ISA has no VOP1 encoding or @p src0 names
///   the literal constant. That form
///   needs a trailing word this encoder has no way to return, so accepting it
///   would emit an instruction that reads whatever follows it.
///
/// @param error Optional caller-owned diagnostic emitter, used only on failure.
///
/// @note On gfx1250 an encoded @p vdst names a register within the bank
///   MODE.VGPR_MSB selects, not a fixed physical VGPR. That constrains callers
///   that must name a specific register, not this encoder
inline util::FailureOr<uint32_t>
build_v_mov_b32_src(uint16_t vdst, uint16_t src0, rj_code_arch_t arch,
                    const util::DiagnosticEmitter *error = nullptr) {
  if (src0 == kVectorSrcLiteral) {
    if (error != nullptr)
      return error->emit() << "v_mov_b32 from a literal needs build_v_mov_b32_imm";
    return util::Result::failure();
  }
  const uint8_t dst = static_cast<uint8_t>(vdst);
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_vop1(cdna1::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_vop1(cdna2::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_vop1(cdna3::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  case ROCJITSU_CODE_ARCH_CDNA5:
    return cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_vop1(rdna2::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_vop1(rdna3::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_vop1(rdna3_5::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_vop1(rdna4::kVMovB32Vop1, {.src0 = src0, .vdst = dst}).front();
  default:
    // RV32I/RV64I and ROCJITSU_CODE_ARCH_INVALID reach here: not AMDGPU, so
    // there is no VOP1 encoding to pick.
    if (error != nullptr)
      return error->emit() << "unsupported target architecture for v_mov_b32 from a register";
    return util::Result::failure();
  }
}

} // namespace rocjitsu

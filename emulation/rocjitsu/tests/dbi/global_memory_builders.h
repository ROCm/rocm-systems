// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file global_memory_builders.h
/// @brief Global-memory load and store encoders for hand-built DBI simulator
///        test code, on the targets DbiSim runs (CDNA3, CDNA4, RDNA4).
///
/// Encoders return failure for unsupported architectures, unencodable immediate
/// offsets, or invalid register alignment and scalar base pairs.
///
/// Every encoder uses the SGPR-base form: the address is the 64-bit SGPR pair at
/// @p saddr plus a 32-bit per-lane offset in VGPR @p voffset plus an immediate
/// byte offset. 2 words on CDNA, 3 on RDNA4.

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"

#include "util/diagnostic.h"
#include "util/result.h"

#include <cstdint>
#include <vector>

namespace rocjitsu {
namespace test {

namespace detail {

/// @brief One global memory operation's opcode on each target.
struct GlobalOpcodes {
  uint16_t cdna3 = 0;
  uint16_t cdna4 = 0;
  uint16_t rdna4 = 0;
};

/// @brief The immediate byte offset each target encodes, in this form.
/// @details CDNA's FLAT offset is a 13-bit signed field whose low 12 bits the
/// generated builder carries; only non-negative offsets below 4096 are used here.
/// RDNA4's ioffset is 24-bit signed.
inline util::FailureOr<uint32_t> checked_global_offset(uint32_t byte_offset, rj_code_arch_t arch,
                                                       util::DiagnosticEmitter emit_error = {}) {
  const uint32_t limit = arch == ROCJITSU_CODE_ARCH_RDNA4 ? 0x7FFFFFu : 0xFFFu;
  if (byte_offset > limit)
    return emit_error.emit() << "global memory offset does not fit the immediate field";
  return byte_offset;
}

/// @brief CDNA3 and CDNA4 (gfx942, gfx950) require a multi-dword VGPR operand to
///        start on an even register; RDNA4 does not.
/// @details The simulator does not enforce this, so a misaligned tuple would
/// pass there and be illegal on hardware.
inline util::Result check_tuple_alignment(uint8_t base, uint32_t dwords, rj_code_arch_t arch,
                                          util::DiagnosticEmitter emit_error = {}) {
  if (dwords > 1 && base % 2 != 0 && arch != ROCJITSU_CODE_ARCH_RDNA4)
    return emit_error.emit() << "multi-dword VGPR operand must start on an even register";
  return util::Result::success();
}

/// @brief The SGPR base is a pair of ordinary SGPRs, which every target
///        requires to start on an even register.
/// @details The decoder accepts an odd base, so the simulator would run it. A
/// base past the ordinary SGPRs selects something other than an SGPR pair:
/// special registers, RDNA4's `off` at 124 (which takes the address from a VGPR
/// pair instead), and from 128 a wrap of the 7-bit field (128 encodes s0).
inline util::Result check_scalar_base(uint8_t saddr, rj_code_arch_t arch,
                                      util::DiagnosticEmitter emit_error = {}) {
  if (saddr % 2 != 0)
    return emit_error.emit() << "SGPR base pair must start on an even register";
  const uint32_t ordinary_sgprs = arch == ROCJITSU_CODE_ARCH_RDNA4
                                      ? amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF
                                      : amdgpu::CdnaIsaBase::MAX_SGPRS_PER_WF;
  if (saddr + 1u >= ordinary_sgprs)
    return emit_error.emit() << "SGPR base pair must lie within the ordinary SGPRs";
  return util::Result::success();
}

inline util::FailureOr<std::vector<uint32_t>>
build_global_load(GlobalOpcodes ops, uint32_t dwords, uint8_t vdst, uint8_t voffset, uint8_t saddr,
                  uint32_t byte_offset, rj_code_arch_t arch,
                  util::DiagnosticEmitter emit_error = {}) {
  const util::FailureOr<uint32_t> maybe_offset =
      checked_global_offset(byte_offset, arch, emit_error);
  if (maybe_offset.failed() || check_tuple_alignment(vdst, dwords, arch, emit_error).failed() ||
      check_scalar_base(saddr, arch, emit_error).failed())
    return util::Result::failure();
  const uint32_t offset = maybe_offset.value();
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const auto encoded = cdna3::build_flat(ops.cdna3, {.offset = static_cast<uint16_t>(offset),
                                                       .seg = 2,
                                                       .addr = voffset,
                                                       .saddr = saddr,
                                                       .vdst = vdst});
    return std::vector<uint32_t>(encoded.begin(), encoded.end());
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const auto encoded = cdna4::build_flat(ops.cdna4, {.offset = static_cast<uint16_t>(offset),
                                                       .seg = 2,
                                                       .addr = voffset,
                                                       .saddr = saddr,
                                                       .vdst = vdst});
    return std::vector<uint32_t>(encoded.begin(), encoded.end());
  }
  case ROCJITSU_CODE_ARCH_RDNA4: {
    const auto encoded = rdna4::build_vglobal(
        ops.rdna4, {.saddr = saddr, .vdst = vdst, .vaddr = voffset, .ioffset = offset});
    return std::vector<uint32_t>(encoded.begin(), encoded.end());
  }
  default:
    return emit_error.emit() << "Unimplemented instruction: global load for target architecture";
  }
}

inline util::FailureOr<std::vector<uint32_t>>
build_global_store(GlobalOpcodes ops, uint32_t dwords, uint8_t vdata, uint8_t voffset,
                   uint8_t saddr, uint32_t byte_offset, rj_code_arch_t arch,
                   util::DiagnosticEmitter emit_error = {}) {
  const util::FailureOr<uint32_t> maybe_offset =
      checked_global_offset(byte_offset, arch, emit_error);
  if (maybe_offset.failed() || check_tuple_alignment(vdata, dwords, arch, emit_error).failed() ||
      check_scalar_base(saddr, arch, emit_error).failed())
    return util::Result::failure();
  const uint32_t offset = maybe_offset.value();
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const auto encoded = cdna3::build_flat(ops.cdna3, {.offset = static_cast<uint16_t>(offset),
                                                       .seg = 2,
                                                       .addr = voffset,
                                                       .data = vdata,
                                                       .saddr = saddr});
    return std::vector<uint32_t>(encoded.begin(), encoded.end());
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const auto encoded = cdna4::build_flat(ops.cdna4, {.offset = static_cast<uint16_t>(offset),
                                                       .seg = 2,
                                                       .addr = voffset,
                                                       .data = vdata,
                                                       .saddr = saddr});
    return std::vector<uint32_t>(encoded.begin(), encoded.end());
  }
  case ROCJITSU_CODE_ARCH_RDNA4: {
    const auto encoded = rdna4::build_vglobal(
        ops.rdna4, {.saddr = saddr, .vsrc = vdata, .vaddr = voffset, .ioffset = offset});
    return std::vector<uint32_t>(encoded.begin(), encoded.end());
  }
  default:
    return emit_error.emit() << "Unimplemented instruction: global store for target architecture";
  }
}

} // namespace detail

/// @brief Encode `global_load_dword vdst, voffset, s[saddr:saddr+1] offset:byte_offset`.
/// @note A build_wait_loads_complete() must precede any use of @p vdst.
inline util::FailureOr<std::vector<uint32_t>>
build_global_load_dword(uint8_t vdst, uint8_t voffset, uint8_t saddr, uint32_t byte_offset,
                        rj_code_arch_t arch, util::DiagnosticEmitter emit_error = {}) {
  return detail::build_global_load(
      {cdna3::kFlatLoadDwordFlat, cdna4::kFlatLoadDwordFlat, rdna4::kGlobalLoadB32Vglobal}, 1, vdst,
      voffset, saddr, byte_offset, arch, emit_error);
}

/// @brief Encode `global_load_dwordx2 v[vdst:vdst+1], voffset, s[saddr:saddr+1]
/// offset:byte_offset`.
/// @note A build_wait_loads_complete() must precede any use of the destination.
inline util::FailureOr<std::vector<uint32_t>>
build_global_load_dwordx2(uint8_t vdst, uint8_t voffset, uint8_t saddr, uint32_t byte_offset,
                          rj_code_arch_t arch, util::DiagnosticEmitter emit_error = {}) {
  return detail::build_global_load(
      {cdna3::kFlatLoadDwordx2Flat, cdna4::kFlatLoadDwordx2Flat, rdna4::kGlobalLoadB64Vglobal}, 2,
      vdst, voffset, saddr, byte_offset, arch, emit_error);
}

/// @brief Encode `global_store_dword voffset, vdata, s[saddr:saddr+1] offset:byte_offset`.
inline util::FailureOr<std::vector<uint32_t>>
build_global_store_dword(uint8_t vdata, uint8_t voffset, uint8_t saddr, uint32_t byte_offset,
                         rj_code_arch_t arch, util::DiagnosticEmitter emit_error = {}) {
  return detail::build_global_store(
      {cdna3::kFlatStoreDwordFlat, cdna4::kFlatStoreDwordFlat, rdna4::kGlobalStoreB32Vglobal}, 1,
      vdata, voffset, saddr, byte_offset, arch, emit_error);
}

/// @brief Encode `global_store_dwordx2` of v[vdata:vdata+1].
inline util::FailureOr<std::vector<uint32_t>>
build_global_store_dwordx2(uint8_t vdata, uint8_t voffset, uint8_t saddr, uint32_t byte_offset,
                           rj_code_arch_t arch, util::DiagnosticEmitter emit_error = {}) {
  return detail::build_global_store(
      {cdna3::kFlatStoreDwordx2Flat, cdna4::kFlatStoreDwordx2Flat, rdna4::kGlobalStoreB64Vglobal},
      2, vdata, voffset, saddr, byte_offset, arch, emit_error);
}

/// @brief Encode `global_store_dwordx3` of v[vdata:vdata+2].
inline util::FailureOr<std::vector<uint32_t>>
build_global_store_dwordx3(uint8_t vdata, uint8_t voffset, uint8_t saddr, uint32_t byte_offset,
                           rj_code_arch_t arch, util::DiagnosticEmitter emit_error = {}) {
  return detail::build_global_store(
      {cdna3::kFlatStoreDwordx3Flat, cdna4::kFlatStoreDwordx3Flat, rdna4::kGlobalStoreB96Vglobal},
      3, vdata, voffset, saddr, byte_offset, arch, emit_error);
}

/// @brief Encode `global_store_dwordx4` of v[vdata:vdata+3].
inline util::FailureOr<std::vector<uint32_t>>
build_global_store_dwordx4(uint8_t vdata, uint8_t voffset, uint8_t saddr, uint32_t byte_offset,
                           rj_code_arch_t arch, util::DiagnosticEmitter emit_error = {}) {
  return detail::build_global_store(
      {cdna3::kFlatStoreDwordx4Flat, cdna4::kFlatStoreDwordx4Flat, rdna4::kGlobalStoreB128Vglobal},
      4, vdata, voffset, saddr, byte_offset, arch, emit_error);
}

} // namespace test
} // namespace rocjitsu

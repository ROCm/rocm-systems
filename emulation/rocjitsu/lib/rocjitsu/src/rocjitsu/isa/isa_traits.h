// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file isa_traits.h
/// @brief Compile-time ISA trait mapping and GPU ISA concept.

#ifndef ROCJITSU_ISA_ISA_TRAITS_H_
#define ROCJITSU_ISA_ISA_TRAITS_H_

#include "rocjitsu/base/api.h"

#include <concepts>
#include <cstdint>

namespace rocjitsu {

/// @brief Compile-time mapping from rj_code_arch_t enum values to Isa trait types.
///
/// @details Specialize this template in each architecture's isa.h to bind its
/// namespace::Isa type. Unspecialized archs will cause a compile error if
/// used, catching unsupported archs early.
/// @tparam Arch Architecture enum value to map.
template <rj_code_arch_t Arch> struct IsaTrait;

/// @brief Concept for ISAs that define GPU wavefront properties.
///
/// @details Requires the static constants and type aliases that all ISA class
/// templates depend on:
///   - `WF_SIZE`               — lanes per wavefront.
///   - `WF_SIZE_MAX`           — largest supported wavefront size.
///   - `MAX_SGPRS_PER_WF`      — maximum scalar GPRs.
///   - `MAX_VGPRS_PER_WF`      — maximum vector GPRs.
///   - `MAX_ACC_VGPRS_PER_WF`  — maximum accumulator VGPRs (0 if absent).
///   - `WAITCNT_LGKMCNT_MASK`  — lgkmcnt field mask in S_WAITCNT (0 if no
///                               monolithic S_WAITCNT — RDNA4 only).
///   - `Context`               — wavefront execution context type.
///   - `OperandType`           — per-ISA operand classification enum.
///   - `StatusReg`             — STATUS register bitfield type.
template <typename Isa>
concept GpuIsa = requires {
  { Isa::WF_SIZE } -> std::convertible_to<uint32_t>;
  { Isa::WF_SIZE_MAX } -> std::convertible_to<uint32_t>;
  { Isa::MAX_SGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  { Isa::MAX_VGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  { Isa::MAX_ACC_VGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  { Isa::WAITCNT_LGKMCNT_MASK } -> std::convertible_to<uint32_t>;
  typename Isa::Context;
  typename Isa::OperandType;
  typename Isa::StatusReg;
};

/// @brief Derived concept: ISA has a dedicated AccVGPR register file.
///
/// @details True for CDNA2/3/4 (`MAX_ACC_VGPRS_PER_WF > 0`).
/// False for CDNA1 and all RDNA ISAs.
template <typename Isa>
concept HasAccVgpr = GpuIsa<Isa> && (Isa::MAX_ACC_VGPRS_PER_WF > 0);

/// @brief Derived concept: ISA has a monolithic S_WAITCNT instruction.
///
/// @details True for CDNA1–4, RDNA1/2, and RDNA3/3.5 (`WAITCNT_LGKMCNT_MASK != 0`).
/// RDNA3/3.5 have BOTH S_WAITCNT (new GFX11 bit layout) AND named per-counter
/// S_WAITCNT_VMCNT etc.  False only for RDNA4 which has no S_WAITCNT at all.
template <typename Isa>
concept HasMonolithicWaitcnt = GpuIsa<Isa> && (Isa::WAITCNT_LGKMCNT_MASK != 0);

/// @brief Compile-time wave-size support query for one ISA.
template <GpuIsa Isa> inline constexpr bool supports_wave_size(uint32_t wf) {
  return (wf == 32 || wf == 64) && wf >= Isa::WF_SIZE && wf <= Isa::WF_SIZE_MAX;
}

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ISA_TRAITS_H_

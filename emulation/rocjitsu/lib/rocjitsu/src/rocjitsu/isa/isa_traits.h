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

/// @brief Width in bits of a single register lane — the finest granularity at
/// which registers are tracked.
///
/// @details A 32-bit lane is the smallest unit RegisterSet addresses. An operand
/// narrower than this writes only part of its lowest lane and read-modify-writes
/// the register rather than fully redefining it.
inline constexpr int REGISTER_GRANULARITY = 32;

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
///   - `MAX_WF_SLOTS`          — maximum simulated wavefront slots per CU.
///   - `MAX_SGPRS_PER_WF`      — maximum scalar GPRs.
///   - `MAX_VGPRS_PER_WF`      — maximum vector GPRs.
///   - `MAX_ADDRESSABLE_VGPRS_PER_WF` — maximum ordinary VGPR address span.
///   - `MAX_ACC_VGPRS_PER_WF`  — maximum accumulator VGPRs (0 if absent).
///   - `WAITCNT_LGKMCNT_MASK`  — lgkmcnt field mask in S_WAITCNT (0 if no
///                               monolithic S_WAITCNT — RDNA4 only).
///   - `MODE_HAS_GPR_IDX_EN`   — true when MODE bit 27 controls VGPR indexing.
///   - `Context`               — wavefront execution context type.
///   - `OperandType`           — per-ISA operand classification enum.
///   - `StatusReg`             — STATUS register bitfield type.
template <typename Isa>
concept GpuIsa = requires {
  { Isa::WF_SIZE } -> std::convertible_to<uint32_t>;
  { Isa::WF_SIZE_MAX } -> std::convertible_to<uint32_t>;
  { Isa::MAX_WF_SLOTS } -> std::convertible_to<uint32_t>;
  { Isa::MAX_SGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  { Isa::MAX_VGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  { Isa::MAX_ADDRESSABLE_VGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  { Isa::MAX_ACC_VGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  { Isa::WAITCNT_LGKMCNT_MASK } -> std::convertible_to<uint32_t>;
  { Isa::MODE_HAS_GPR_IDX_EN } -> std::convertible_to<bool>;
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

/// @brief Return true when @p arch is CDNA1 through CDNA4.
///
/// @details This predicate represents the CDNA1-4 descriptor and wavefront
/// policy boundary, not the entire CDNA family. Rocjitsu models CDNA5, which
/// originated as gfx1250, with RDNA4-derived descriptor and wavefront
/// properties. Adding CDNA5 here would silently change those decisions in
/// callers. Keep this policy near the ISA trait declarations so DBT call sites
/// do not grow their own partial architecture switch statements.
[[nodiscard]] inline constexpr bool arch_is_cdna_4_or_lower(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

/// @brief Return true when @p arch belongs to the RDNA ISA family.
[[nodiscard]] inline constexpr bool arch_is_rdna(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

/// @brief Return true when scalar selectors 102 and 103 name FLAT_SCRATCH.
///
/// @details GFX10+ makes those selectors ordinary SGPRs. Keep both the legacy
/// and modern architecture sets explicit so an unclassified future target does
/// not silently inherit the legacy register alias.
[[nodiscard]] inline constexpr bool arch_uses_legacy_flat_scratch_sgprs(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return true;
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
  default:
    return false;
  }
}

/// @brief Maximum SGPR allocation encodable in an AMDHSA descriptor for @p arch.
///
/// @details CDNA descriptors account for reserved architectural SGPRs such as
/// VCC in the encoded wavefront allocation. That descriptor limit is larger
/// than the ordinary scratch SGPR range exposed through CdnaIsaBase, so DBT
/// must query the descriptor limit separately from semantic scratch limits.
/// RDNA and gfx1250 descriptors use the ordinary ISA SGPR maximum. gfx1250 is
/// kept out of arch_is_rdna() because several of its descriptor and register
/// allocation rules differ from generic RDNA despite sharing the GFX10+ ABI.
/// @brief Whether @p arch encodes wavefront SGPR allocation in the kernel descriptor.
///
/// @details GFX10+ leaves COMPUTE_PGM_RSRC1.GRANULATED_WAVEFRONT_SGPR_COUNT reserved and gives
/// every wave the architectural SGPR file, so a count decoded from that field is an artifact of
/// the granule-0 encoding rather than a budget any caller depends on. Only where the field is
/// live can an SGPR requirement above the decoded count under-provision anyone.
[[nodiscard]] inline constexpr bool arch_descriptor_encodes_sgpr_allocation(rj_code_arch_t arch) {
  return !(arch_is_rdna(arch) || arch == ROCJITSU_CODE_ARCH_CDNA5);
}

[[nodiscard]] inline constexpr uint32_t arch_descriptor_sgpr_allocation_limit(rj_code_arch_t arch) {
  // Architectural descriptor SGPR-allocation ceilings. CDNA descriptors may name
  // up to 112 SGPRs; RDNA up to 106. These are fixed ISA facts (not the smaller
  // scratch range in CdnaIsaBase), so they are named here rather than derived.
  constexpr uint32_t kCdnaDescriptorSgprLimit = 112;
  constexpr uint32_t kRdnaDescriptorSgprLimit = 106;
  if (arch_is_cdna_4_or_lower(arch))
    return kCdnaDescriptorSgprLimit;
  if (arch_is_rdna(arch) || arch == ROCJITSU_CODE_ARCH_CDNA5)
    return kRdnaDescriptorSgprLimit;
  return 0;
}

/// @brief Maximum hardware LDS bytes available to one workgroup on @p arch.
///
/// @details Zero means the limit is not modeled yet. Callers use that
/// conservative value to avoid inventing compatibility policy for architectures
/// whose LDS allocation limit has not been wired into rocjitsu.
[[nodiscard]] inline constexpr uint32_t arch_lds_bytes(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
    return 64u * 1024u;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return 160u * 1024u;
  case ROCJITSU_CODE_ARCH_CDNA5:
    // gfx1250 can allocate up to 320 KiB to one workgroup. This is distinct
    // from the configurable LDS/vector-cache partition sizes reported for a
    // TCP, which must not be used as the descriptor allocation ceiling.
    return 320u * 1024u;
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return 64u * 1024u;
  default:
    return 0;
  }
}

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ISA_TRAITS_H_

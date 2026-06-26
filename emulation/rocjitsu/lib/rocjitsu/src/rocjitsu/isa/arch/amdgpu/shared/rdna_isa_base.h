// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rdna_isa_base.h
/// @brief Shared ISA base definitions for RDNA1–4 (GFX10/11/12 family).
///
/// Provides `RdnaIsaBase` — a struct containing the static constants shared
/// by all RDNA ISAs.  Each per-ISA `isa.h` inherits from `RdnaIsaBase` and
/// overrides only the fields that differ:
///   - RDNA1/2 override `WAITCNT_LGKMCNT_MASK` to 0x3F (6-bit lgkmcnt at [13:8]).
///   - RDNA3/3.5 override `WAITCNT_LGKMCNT_MASK` to 0x3F (6-bit lgkmcnt at [9:4]
///     in GFX11's new S_WAITCNT layout); they also have named per-counter variants.
///   - RDNA4 leaves it at 0 (no monolithic S_WAITCNT; only split S_WAIT_*).
///
/// STATUS register notes:
///   RDNA1/2 and RDNA3/4 have slightly different STATUS register layouts
///   (RDNA1/2 expose TTRACE_EN, EXPORT_RDY, SKIP_EXPORT; RDNA3/4 do not).
///   Each RDNA `isa.h` therefore keeps its own per-ISA `StatusReg` class.
///
/// SGPR count: 106 — verified per RDNA ISA specifications.  RDNA allows up
/// to 106 SGPRs per wavefront (two fewer than the encodable maximum of 108
/// to avoid aliasing with the trap temporaries TTMP0–TTMP15).
///
/// WF_SIZE / WF_SIZE_MAX: RDNA default wave size is Wave32 (`WF_SIZE = 32`).
/// Most RDNA targets supported here also allow Wave64 via
/// ENABLE_WAVEFRONT_SIZE32=0 in the kernel descriptor (`WF_SIZE_MAX = 64`);
/// target-specific `isa.h` files can shadow `WF_SIZE_MAX` when needed.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_RDNA_ISA_BASE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_RDNA_ISA_BASE_H_

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief Shared ISA base struct for all RDNA ISAs (GFX10/11/12 family, Wave32 default).
///
/// @details Inheriting structs may shadow any constant with an ISA-specific
/// override.  In particular:
///   - RDNA1/2 override `WAITCNT_LGKMCNT_MASK = 0x3F` (6-bit lgkmcnt at [13:8]).
///   - RDNA3/3.5 override `WAITCNT_LGKMCNT_MASK = 0x3F` (6-bit lgkmcnt at [9:4]
///     in GFX11's new S_WAITCNT layout).
///   - RDNA4 inherits `WAITCNT_LGKMCNT_MASK = 0` (no monolithic S_WAITCNT).
///
/// Each per-ISA `Isa` struct adds `Decoder`, `MachineInst`, `OperandType`, and
/// `StatusReg` type aliases.
struct RdnaIsaBase {
  static constexpr uint32_t WF_SIZE = 32;             ///< Default wave size (Wave32).
  static constexpr uint32_t WF_SIZE_MAX = 64;         ///< Maximum wave size (Wave64 capable).
  static constexpr uint32_t MAX_SGPRS_PER_WF = 106;   ///< SGPRs per wavefront.
  static constexpr uint32_t MAX_VGPRS_PER_WF = 256;   ///< VGPRs per wavefront.
  static constexpr uint32_t MAX_ACC_VGPRS_PER_WF = 0; ///< No AccVGPR file on any RDNA ISA.
  static constexpr bool SRAM_ECC = false;             ///< RDNA does not alter D16 for ECC.
  static constexpr uint8_t WAITCNT_LGKMCNT_MASK = 0;  ///< 0 = no monolithic S_WAITCNT.

  using Context = amdgpu::Wavefront;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_RDNA_ISA_BASE_H_

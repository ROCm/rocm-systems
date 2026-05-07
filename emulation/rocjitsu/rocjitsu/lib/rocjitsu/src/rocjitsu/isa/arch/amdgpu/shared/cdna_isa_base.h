// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna_isa_base.h
/// @brief Shared ISA base definitions for CDNA1–4 (GFX9 family).
///
/// Provides `CdnaStatusReg` — the common wavefront status register layout —
/// and `CdnaIsaBase` — a struct containing the static constants shared by all
/// CDNA ISAs. Each per-ISA `isa.h` inherits from `CdnaIsaBase` and overrides
/// only the fields that differ (e.g. `MAX_ACC_VGPRS_PER_WF` for CDNA2/3/4).
///
/// SGPR count: 102 (not 104 — verified per CDNA ISA specifications; the two
/// highest-numbered SGPRs are reserved for VCC on CDNA Wave64).
///
/// WAITCNT_LGKMCNT_MASK: 0x0F — GFX9 S_WAITCNT encodes lgkmcnt in bits [11:8]
/// (4 bits).  CDNA4, while GFX11-generation hardware, retains the single
/// monolithic S_WAITCNT instruction from the GFX9 encoding family.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_CDNA_ISA_BASE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_CDNA_ISA_BASE_H_

#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/bitfield.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief Shared STATUS register layout for all CDNA ISAs (GFX9 family).
///
/// @details Bit positions follow the CDNA3/4 ISA specification, which is a
/// superset of CDNA1/2.  CDNA1/2 do not expose COND_DBG_USER or COND_DBG_SYS
/// in hardware, but those bits are inert in simulation.
///
/// Each accessor returns a lightweight `Bitfield<32>` proxy that supports
/// implicit read (integral conversion) and write (`operator=`).
class CdnaStatusReg : public ::util::Bitfield<32> {
public:
  using Bitfield::Bitfield;
  using Bitfield::operator=;
  CdnaStatusReg(const CdnaStatusReg &) = default;
  CdnaStatusReg &operator=(const CdnaStatusReg &) = default;

  auto SCC() { return member<0, 0>(); }
  auto SCC() const { return member<0, 0>(); }
  auto SPI_PRIO() { return member<1, 2>(); }
  auto SPI_PRIO() const { return member<1, 2>(); }
  auto WAVE_PRIO() { return member<3, 4>(); }
  auto WAVE_PRIO() const { return member<3, 4>(); }
  auto PRIV() { return member<5, 5>(); }
  auto PRIV() const { return member<5, 5>(); }
  auto TRAP_EN() { return member<6, 6>(); }
  auto TRAP_EN() const { return member<6, 6>(); }
  auto EXECZ() { return member<9, 9>(); }
  auto EXECZ() const { return member<9, 9>(); }
  auto VCCZ() { return member<10, 10>(); }
  auto VCCZ() const { return member<10, 10>(); }
  auto IN_TG() { return member<11, 11>(); }
  auto IN_TG() const { return member<11, 11>(); }
  auto IN_BARRIER() { return member<12, 12>(); }
  auto IN_BARRIER() const { return member<12, 12>(); }
  auto HALT() { return member<13, 13>(); }
  auto HALT() const { return member<13, 13>(); }
  auto TRAP() { return member<14, 14>(); }
  auto TRAP() const { return member<14, 14>(); }
  auto VALID() { return member<16, 16>(); }
  auto VALID() const { return member<16, 16>(); }
  auto ECC_ERR() { return member<17, 17>(); }
  auto ECC_ERR() const { return member<17, 17>(); }
  auto PERF_EN() { return member<19, 19>(); }
  auto PERF_EN() const { return member<19, 19>(); }
  /// @brief Conditional debug — user mode (CDNA3/4 only; inert on CDNA1/2).
  auto COND_DBG_USER() { return member<20, 20>(); }
  auto COND_DBG_USER() const { return member<20, 20>(); }
  /// @brief Conditional debug — system mode (CDNA3/4 only; inert on CDNA1/2).
  auto COND_DBG_SYS() { return member<21, 21>(); }
  auto COND_DBG_SYS() const { return member<21, 21>(); }
  auto ALLOW_REPLAY() { return member<22, 22>(); }
  auto ALLOW_REPLAY() const { return member<22, 22>(); }
};

/// @brief Shared ISA base struct for all CDNA ISAs (GFX9 family, Wave64).
///
/// @details Inheriting structs may shadow any constant with an ISA-specific
/// override, e.g. `static constexpr uint32_t MAX_ACC_VGPRS_PER_WF = 256;`
/// for CDNA2/3/4.  Constants not overridden are inherited as-is.
///
/// `Context` is fixed to `amdgpu::Wavefront` for all CDNA ISAs; each per-ISA
/// `Isa` struct adds `Decoder`, `MachineInst`, `OperandType`, and `StatusReg`
/// type aliases.
struct CdnaIsaBase {
  static constexpr uint32_t WF_SIZE = 64;               ///< Lanes per wavefront (Wave64).
  static constexpr uint32_t MAX_SGPRS_PER_WF = 102;     ///< SGPRs per wavefront.
  static constexpr uint32_t MAX_VGPRS_PER_WF = 256;     ///< VGPRs per wavefront.
  static constexpr uint32_t MAX_ACC_VGPRS_PER_WF = 0;   ///< AccVGPRs (0 = absent; CDNA1 default).
  static constexpr uint8_t WAITCNT_LGKMCNT_MASK = 0x0F; ///< lgkmcnt mask in S_WAITCNT [11:8].
  static constexpr bool SRAM_ECC = false;               ///< CDNA1 default: no SRAM ECC.

  using Context = amdgpu::Wavefront;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_CDNA_ISA_BASE_H_

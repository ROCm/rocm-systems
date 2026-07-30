// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/cdna4/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h"
#include "rocjitsu/isa/isa_traits.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
}
namespace cdna4 {

/// @brief CDNA4 ISA traits (GFX950, Wave64, dedicated AccVGPR file, GFX9 S_WAITCNT).
///
/// @details Overrides `MAX_ACC_VGPRS_PER_WF = 256` (CDNA4 retains the
/// dedicated AccVGPR register file from CDNA2/3).
/// All other constants inherit from `amdgpu::CdnaIsaBase`.
///
/// CDNA4 is GFX11-generation hardware but retains the single monolithic
/// S_WAITCNT instruction from the GFX9 encoding family; hence
/// `WAITCNT_LGKMCNT_MASK = 0x0F` is inherited unchanged from `CdnaIsaBase`.
///
/// StatusReg uses the shared `amdgpu::CdnaStatusReg` layout including
/// COND_DBG_USER and COND_DBG_SYS (both active on CDNA3/4 hardware).
struct Isa : amdgpu::CdnaIsaBase {
  static constexpr uint32_t MAX_ACC_VGPRS_PER_WF =
      256;                               ///< Unified AccVGPR file (src encoding alias at 768).
  static constexpr bool SRAM_ECC = true; ///< gfx950 has SRAM ECC.

  using Decoder = cdna4::Decoder;
  using MachineInst = cdna4::MachineInst;
  using OperandType = cdna4::OperandType;
  using StatusReg = amdgpu::CdnaStatusReg;

  // SIMD fast-path traits — consumed by AmdgpuIsaOperand<Isa> in
  // rocjitsu/isa/isa_operand_simd_inl.h. Definitions live in this arch's
  // operand.cpp; bodies forward to the anonymous-namespace helpers.
  static std::optional<uint32_t> resolved_vgpr_offset(OperandType opr_type, int ev);
  static bool simd_capable_value(OperandType opr_type, int ev);
  static uint32_t simd_broadcast_value(const amdgpu::Wavefront &wf, OperandType opr_type, int ev);
};

} // namespace cdna4

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_CDNA4> {
  using type = cdna4::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_ISA_H_

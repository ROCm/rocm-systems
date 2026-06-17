// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/cdna2/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h"
#include "rocjitsu/isa/isa_traits.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
}
namespace cdna2 {

/// @brief CDNA2 ISA traits (GFX90a, Wave64, separate AccVGPR file, GFX9 S_WAITCNT).
///
/// @details Overrides `MAX_ACC_VGPRS_PER_WF = 256` (CDNA2 introduced a
/// dedicated AccVGPR register file; MFMA outputs go to AccVGPRs, not VGPRs).
/// All other constants inherit from `amdgpu::CdnaIsaBase`.
///
/// CDNA2 hardware does not expose COND_DBG_USER / COND_DBG_SYS but those
/// bits are inert in simulation via `amdgpu::CdnaStatusReg`.
struct Isa : amdgpu::CdnaIsaBase {
  static constexpr uint32_t MAX_ACC_VGPRS_PER_WF =
      256;                               ///< Separate AccVGPR file (encoding base 512).
  static constexpr bool SRAM_ECC = true; ///< gfx90a has SRAM ECC.

  using Decoder = cdna2::Decoder;
  using MachineInst = cdna2::MachineInst;
  using OperandType = cdna2::OperandType;
  using StatusReg = amdgpu::CdnaStatusReg;

  // SIMD fast-path traits — consumed by AmdgpuIsaOperand<Isa> in
  // rocjitsu/isa/isa_operand_simd_inl.h. Definitions live in this arch's
  // operand.cpp; bodies forward to the anonymous-namespace helpers.
  static std::optional<uint32_t> resolved_vgpr_offset(OperandType opr_type, int ev);
  static bool simd_capable_value(OperandType opr_type, int ev);
  static uint32_t simd_broadcast_value(const amdgpu::Wavefront &wf, OperandType opr_type, int ev);
};

} // namespace cdna2

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_CDNA2> {
  using type = cdna2::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_ISA_H_

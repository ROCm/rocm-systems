// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h"
#include "rocjitsu/isa/isa_traits.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
}
namespace cdna1 {

/// @brief CDNA1 ISA traits (GFX908, Wave64, separate AccVGPR file, GFX9 S_WAITCNT).
///
/// @details Inherits scalar and vector defaults from `amdgpu::CdnaIsaBase`:
///   - WF_SIZE = 64, MAX_SGPRS = 102, MAX_VGPRS = 256
///   - WAITCNT_LGKMCNT_MASK = 0x0F  (GFX9 4-bit lgkmcnt at [11:8])
/// Overrides MAX_ACC_VGPRS_PER_WF to 256 (CDNA1 ISA chapter 7: MFMA writes ACC).
///
/// StatusReg uses the shared `amdgpu::CdnaStatusReg` layout.  CDNA1 hardware
/// does not expose COND_DBG_USER / COND_DBG_SYS but those bits are inert in
/// simulation.
struct Isa : amdgpu::CdnaIsaBase {
  static constexpr uint32_t MAX_ACC_VGPRS_PER_WF =
      256; ///< Separate accumulator bank, encoded from 768.
  using Decoder = cdna1::Decoder;
  using MachineInst = cdna1::MachineInst;
  using OperandType = cdna1::OperandType;
  using StatusReg = amdgpu::CdnaStatusReg;

  // SIMD fast-path traits — consumed by AmdgpuIsaOperand<Isa> in
  // rocjitsu/isa/isa_operand_simd_inl.h. Definitions live in this arch's
  // operand.cpp; bodies forward to the anonymous-namespace helpers.
  static std::optional<uint32_t> resolved_vgpr_offset(OperandType opr_type, int ev);
  static bool simd_capable_value(OperandType opr_type, int ev);
  static uint32_t simd_broadcast_value(const amdgpu::Wavefront &wf, OperandType opr_type, int ev);
};

} // namespace cdna1

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_CDNA1> {
  using type = cdna1::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_ISA_H_

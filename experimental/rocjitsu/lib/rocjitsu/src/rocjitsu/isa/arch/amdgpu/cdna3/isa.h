// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/cdna3/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h"
#include "rocjitsu/isa/isa_traits.h"

#include <cstdint>

namespace rocjitsu {
namespace cdna3 {

/// @brief CDNA3 ISA traits (GFX940, Wave64, dedicated AccVGPR file, GFX9 S_WAITCNT).
///
/// @details Overrides `MAX_ACC_VGPRS_PER_WF = 256` (CDNA3 retains the
/// dedicated AccVGPR register file introduced in CDNA2).
/// All other constants inherit from `amdgpu::CdnaIsaBase`.
///
/// StatusReg uses the shared `amdgpu::CdnaStatusReg` layout (CDNA3 is the
/// source of truth for the superset layout including COND_DBG_USER/SYS).
struct Isa : amdgpu::CdnaIsaBase {
  static constexpr uint32_t MAX_ACC_VGPRS_PER_WF =
      256; ///< Unified AccVGPR file (src encoding alias at 768).

  using Decoder = cdna3::Decoder;
  using MachineInst = cdna3::MachineInst;
  using OperandType = cdna3::OperandType;
  using StatusReg = amdgpu::CdnaStatusReg;
};

} // namespace cdna3

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_CDNA3> {
  using type = cdna3::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_ISA_H_

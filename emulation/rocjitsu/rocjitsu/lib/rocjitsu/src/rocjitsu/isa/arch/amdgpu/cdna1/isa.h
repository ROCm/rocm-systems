// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/cdna1/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h"
#include "rocjitsu/isa/isa_traits.h"

#include <cstdint>

namespace rocjitsu {
namespace cdna1 {

/// @brief CDNA1 ISA traits (GFX908, Wave64, no AccVGPR, GFX9 S_WAITCNT).
///
/// @details Inherits all defaults from `amdgpu::CdnaIsaBase`:
///   - WF_SIZE = 64, MAX_SGPRS = 102, MAX_VGPRS = 256
///   - MAX_ACC_VGPRS = 0   (CDNA1 has no AccVGPR file; MFMA writes to VGPRs)
///   - WAITCNT_LGKMCNT_MASK = 0x0F  (GFX9 4-bit lgkmcnt at [11:8])
///
/// StatusReg uses the shared `amdgpu::CdnaStatusReg` layout.  CDNA1 hardware
/// does not expose COND_DBG_USER / COND_DBG_SYS but those bits are inert in
/// simulation.
struct Isa : amdgpu::CdnaIsaBase {
  using Decoder = cdna1::Decoder;
  using MachineInst = cdna1::MachineInst;
  using OperandType = cdna1::OperandType;
  using StatusReg = amdgpu::CdnaStatusReg;
};

} // namespace cdna1

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_CDNA1> {
  using type = cdna1::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_ISA_H_

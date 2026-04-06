// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA2_EXP_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA2_EXP_H_

#include "rocjitsu/isa/arch/amdgpu/rdna2/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/operand.h"

namespace rocjitsu {
namespace rdna2 {

class ExpExp : public Exp {
public:
  ExpExp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;
  Operand tgt;
  Operand vsrc0;
  Operand vsrc1;
  Operand vsrc2;
  Operand vsrc3;
};

} // namespace rdna2
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA2_EXP_H_

// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_EXP_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_EXP_H_

#include "rocjitsu/isa/arch/amdgpu/cdna1/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/operand.h"

namespace rocjitsu {
namespace cdna1 {

class ExpExp : public Exp {
public:
  ExpExp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand tgt;
  Operand vsrc0;
  Operand vsrc1;
  Operand vsrc2;
  Operand vsrc3;
};

} // namespace cdna1
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_EXP_H_

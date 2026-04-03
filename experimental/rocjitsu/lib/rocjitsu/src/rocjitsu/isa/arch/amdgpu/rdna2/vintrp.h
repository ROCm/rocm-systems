// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA2_VINTRP_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA2_VINTRP_H_

#include "rocjitsu/isa/arch/amdgpu/rdna2/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/operand.h"

namespace rocjitsu {
namespace rdna2 {

class VInterpP1F32Vintrp : public Vintrp {
public:
  VInterpP1F32Vintrp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand vsrc;
  Operand attr;
};

class VInterpP2F32Vintrp : public Vintrp {
public:
  VInterpP2F32Vintrp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand vsrc;
  Operand attr;
};

class VInterpMovF32Vintrp : public Vintrp {
public:
  VInterpMovF32Vintrp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand vsrc;
  Operand attr;
};

} // namespace rdna2
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA2_VINTRP_H_

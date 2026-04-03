// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_VINTERP_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_VINTERP_H_

#include "rocjitsu/isa/arch/amdgpu/rdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/operand.h"

namespace rocjitsu {
namespace rdna3 {

class VInterpP10F32Vinterp : public Vinterp {
public:
  VInterpP10F32Vinterp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VInterpP2F32Vinterp : public Vinterp {
public:
  VInterpP2F32Vinterp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VInterpP10F16F32Vinterp : public Vinterp {
public:
  VInterpP10F16F32Vinterp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VInterpP2F16F32Vinterp : public Vinterp {
public:
  VInterpP2F16F32Vinterp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VInterpP10RtzF16F32Vinterp : public Vinterp {
public:
  VInterpP10RtzF16F32Vinterp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VInterpP2RtzF16F32Vinterp : public Vinterp {
public:
  VInterpP2RtzF16F32Vinterp(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

} // namespace rdna3
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_VINTERP_H_

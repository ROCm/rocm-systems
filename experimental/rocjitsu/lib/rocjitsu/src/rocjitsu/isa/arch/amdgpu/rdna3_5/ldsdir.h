// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_LDSDIR_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_LDSDIR_H_

#include "rocjitsu/isa/arch/amdgpu/rdna3_5/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/operand.h"

namespace rocjitsu {
namespace rdna3_5 {

class LdsParamLoadLdsdir : public Ldsdir {
public:
  LdsParamLoadLdsdir(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand attr;
};

class LdsDirectLoadLdsdir : public Ldsdir {
public:
  LdsDirectLoadLdsdir(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
};

} // namespace rdna3_5
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_LDSDIR_H_

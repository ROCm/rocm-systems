// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_VOP3_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_VOP3_H_

#include "rocjitsu/isa/arch/amdgpu/cdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/operand.h"

namespace rocjitsu {
namespace cdna3 {

class VNopVop3 : public Vop3 {
public:
  VNopVop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
};

class VMovB32Vop3 : public Vop3 {
public:
  VMovB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VReadfirstlaneB32Vop3 : public Vop3 {
public:
  VReadfirstlaneB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtI32F64Vop3 : public Vop3 {
public:
  VCvtI32F64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF64I32Vop3 : public Vop3 {
public:
  VCvtF64I32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32I32Vop3 : public Vop3 {
public:
  VCvtF32I32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32U32Vop3 : public Vop3 {
public:
  VCvtF32U32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtU32F32Vop3 : public Vop3 {
public:
  VCvtU32F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtI32F32Vop3 : public Vop3 {
public:
  VCvtI32F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF16F32Vop3 : public Vop3 {
public:
  VCvtF16F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32F16Vop3 : public Vop3 {
public:
  VCvtF32F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtRpiI32F32Vop3 : public Vop3 {
public:
  VCvtRpiI32F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtFlrI32F32Vop3 : public Vop3 {
public:
  VCvtFlrI32F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtOffF32I4Vop3 : public Vop3 {
public:
  VCvtOffF32I4Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32F64Vop3 : public Vop3 {
public:
  VCvtF32F64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF64F32Vop3 : public Vop3 {
public:
  VCvtF64F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32Ubyte0Vop3 : public Vop3 {
public:
  VCvtF32Ubyte0Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32Ubyte1Vop3 : public Vop3 {
public:
  VCvtF32Ubyte1Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32Ubyte2Vop3 : public Vop3 {
public:
  VCvtF32Ubyte2Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32Ubyte3Vop3 : public Vop3 {
public:
  VCvtF32Ubyte3Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtU32F64Vop3 : public Vop3 {
public:
  VCvtU32F64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF64U32Vop3 : public Vop3 {
public:
  VCvtF64U32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VTruncF64Vop3 : public Vop3 {
public:
  VTruncF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCeilF64Vop3 : public Vop3 {
public:
  VCeilF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRndneF64Vop3 : public Vop3 {
public:
  VRndneF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFloorF64Vop3 : public Vop3 {
public:
  VFloorF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFractF32Vop3 : public Vop3 {
public:
  VFractF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VTruncF32Vop3 : public Vop3 {
public:
  VTruncF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCeilF32Vop3 : public Vop3 {
public:
  VCeilF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRndneF32Vop3 : public Vop3 {
public:
  VRndneF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFloorF32Vop3 : public Vop3 {
public:
  VFloorF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VExpF32Vop3 : public Vop3 {
public:
  VExpF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VLogF32Vop3 : public Vop3 {
public:
  VLogF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRcpF32Vop3 : public Vop3 {
public:
  VRcpF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRcpIflagF32Vop3 : public Vop3 {
public:
  VRcpIflagF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRsqF32Vop3 : public Vop3 {
public:
  VRsqF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRcpF64Vop3 : public Vop3 {
public:
  VRcpF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRsqF64Vop3 : public Vop3 {
public:
  VRsqF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VSqrtF32Vop3 : public Vop3 {
public:
  VSqrtF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VSqrtF64Vop3 : public Vop3 {
public:
  VSqrtF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VSinF32Vop3 : public Vop3 {
public:
  VSinF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCosF32Vop3 : public Vop3 {
public:
  VCosF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VNotB32Vop3 : public Vop3 {
public:
  VNotB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VBfrevB32Vop3 : public Vop3 {
public:
  VBfrevB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFfbhU32Vop3 : public Vop3 {
public:
  VFfbhU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFfblB32Vop3 : public Vop3 {
public:
  VFfblB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFfbhI32Vop3 : public Vop3 {
public:
  VFfbhI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFrexpExpI32F64Vop3 : public Vop3 {
public:
  VFrexpExpI32F64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFrexpMantF64Vop3 : public Vop3 {
public:
  VFrexpMantF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFractF64Vop3 : public Vop3 {
public:
  VFractF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFrexpExpI32F32Vop3 : public Vop3 {
public:
  VFrexpExpI32F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFrexpMantF32Vop3 : public Vop3 {
public:
  VFrexpMantF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VClrexcpVop3 : public Vop3 {
public:
  VClrexcpVop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
};

class VScreenPartition4seB32Vop3 : public Vop3 {
public:
  VScreenPartition4seB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VMovB64Vop3 : public Vop3 {
public:
  VMovB64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF16U16Vop3 : public Vop3 {
public:
  VCvtF16U16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF16I16Vop3 : public Vop3 {
public:
  VCvtF16I16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtU16F16Vop3 : public Vop3 {
public:
  VCvtU16F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtI16F16Vop3 : public Vop3 {
public:
  VCvtI16F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRcpF16Vop3 : public Vop3 {
public:
  VRcpF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VSqrtF16Vop3 : public Vop3 {
public:
  VSqrtF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRsqF16Vop3 : public Vop3 {
public:
  VRsqF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VLogF16Vop3 : public Vop3 {
public:
  VLogF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VExpF16Vop3 : public Vop3 {
public:
  VExpF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFrexpMantF16Vop3 : public Vop3 {
public:
  VFrexpMantF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFrexpExpI16F16Vop3 : public Vop3 {
public:
  VFrexpExpI16F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFloorF16Vop3 : public Vop3 {
public:
  VFloorF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCeilF16Vop3 : public Vop3 {
public:
  VCeilF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VTruncF16Vop3 : public Vop3 {
public:
  VTruncF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VRndneF16Vop3 : public Vop3 {
public:
  VRndneF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VFractF16Vop3 : public Vop3 {
public:
  VFractF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VSinF16Vop3 : public Vop3 {
public:
  VSinF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCosF16Vop3 : public Vop3 {
public:
  VCosF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtNormI16F16Vop3 : public Vop3 {
public:
  VCvtNormI16F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtNormU16F16Vop3 : public Vop3 {
public:
  VCvtNormU16F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VSatPkU8I16Vop3 : public Vop3 {
public:
  VSatPkU8I16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VSwapB32Vop3 : public Vop3 {
public:
  VSwapB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VAccvgprMovB32Vop3 : public Vop3 {
public:
  VAccvgprMovB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32Fp8Vop3 : public Vop3 {
public:
  VCvtF32Fp8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtF32Bf8Vop3 : public Vop3 {
public:
  VCvtF32Bf8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtPkF32Fp8Vop3 : public Vop3 {
public:
  VCvtPkF32Fp8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCvtPkF32Bf8Vop3 : public Vop3 {
public:
  VCvtPkF32Bf8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
};

class VCndmaskB32Vop3 : public Vop3 {
public:
  VCndmaskB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VAddF32Vop3 : public Vop3 {
public:
  VAddF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubF32Vop3 : public Vop3 {
public:
  VSubF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubrevF32Vop3 : public Vop3 {
public:
  VSubrevF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VFmacF64Vop3 : public Vop3 {
public:
  VFmacF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulF32Vop3 : public Vop3 {
public:
  VMulF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulI32I24Vop3 : public Vop3 {
public:
  VMulI32I24Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulHiI32I24Vop3 : public Vop3 {
public:
  VMulHiI32I24Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulU32U24Vop3 : public Vop3 {
public:
  VMulU32U24Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulHiU32U24Vop3 : public Vop3 {
public:
  VMulHiU32U24Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMinF32Vop3 : public Vop3 {
public:
  VMinF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMaxF32Vop3 : public Vop3 {
public:
  VMaxF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMinI32Vop3 : public Vop3 {
public:
  VMinI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMaxI32Vop3 : public Vop3 {
public:
  VMaxI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMinU32Vop3 : public Vop3 {
public:
  VMinU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMaxU32Vop3 : public Vop3 {
public:
  VMaxU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VLshrrevB32Vop3 : public Vop3 {
public:
  VLshrrevB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VAshrrevI32Vop3 : public Vop3 {
public:
  VAshrrevI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VLshlrevB32Vop3 : public Vop3 {
public:
  VLshlrevB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VAndB32Vop3 : public Vop3 {
public:
  VAndB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VOrB32Vop3 : public Vop3 {
public:
  VOrB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VXorB32Vop3 : public Vop3 {
public:
  VXorB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VAddCoU32Vop3SdstEnc : public Vop3SdstEnc {
public:
  VAddCoU32Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
};

class VSubCoU32Vop3SdstEnc : public Vop3SdstEnc {
public:
  VSubCoU32Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
};

class VSubrevCoU32Vop3SdstEnc : public Vop3SdstEnc {
public:
  VSubrevCoU32Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
};

class VAddcCoU32Vop3SdstEnc : public Vop3SdstEnc {
public:
  VAddcCoU32Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VSubbCoU32Vop3SdstEnc : public Vop3SdstEnc {
public:
  VSubbCoU32Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VSubbrevCoU32Vop3SdstEnc : public Vop3SdstEnc {
public:
  VSubbrevCoU32Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VAddF16Vop3 : public Vop3 {
public:
  VAddF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubF16Vop3 : public Vop3 {
public:
  VSubF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubrevF16Vop3 : public Vop3 {
public:
  VSubrevF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulF16Vop3 : public Vop3 {
public:
  VMulF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMacF16Vop3 : public Vop3 {
public:
  VMacF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VAddU16Vop3 : public Vop3 {
public:
  VAddU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubU16Vop3 : public Vop3 {
public:
  VSubU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubrevU16Vop3 : public Vop3 {
public:
  VSubrevU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulLoU16Vop3 : public Vop3 {
public:
  VMulLoU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VLshlrevB16Vop3 : public Vop3 {
public:
  VLshlrevB16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VLshrrevB16Vop3 : public Vop3 {
public:
  VLshrrevB16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VAshrrevI16Vop3 : public Vop3 {
public:
  VAshrrevI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMaxF16Vop3 : public Vop3 {
public:
  VMaxF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMinF16Vop3 : public Vop3 {
public:
  VMinF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMaxU16Vop3 : public Vop3 {
public:
  VMaxU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMaxI16Vop3 : public Vop3 {
public:
  VMaxI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMinU16Vop3 : public Vop3 {
public:
  VMinU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMinI16Vop3 : public Vop3 {
public:
  VMinI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VLdexpF16Vop3 : public Vop3 {
public:
  VLdexpF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VAddU32Vop3 : public Vop3 {
public:
  VAddU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubU32Vop3 : public Vop3 {
public:
  VSubU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubrevU32Vop3 : public Vop3 {
public:
  VSubrevU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VDot2cF32F16Vop3 : public Vop3 {
public:
  VDot2cF32F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VDot2cI32I16Vop3 : public Vop3 {
public:
  VDot2cI32I16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VDot4cI32I8Vop3 : public Vop3 {
public:
  VDot4cI32I8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VDot8cI32I4Vop3 : public Vop3 {
public:
  VDot8cI32I4Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VFmacF32Vop3 : public Vop3 {
public:
  VFmacF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VPkFmacF16Vop3 : public Vop3 {
public:
  VPkFmacF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VXnorB32Vop3 : public Vop3 {
public:
  VXnorB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMadI32I24Vop3 : public Vop3 {
public:
  VMadI32I24Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadU32U24Vop3 : public Vop3 {
public:
  VMadU32U24Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VCubeidF32Vop3 : public Vop3 {
public:
  VCubeidF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VCubescF32Vop3 : public Vop3 {
public:
  VCubescF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VCubetcF32Vop3 : public Vop3 {
public:
  VCubetcF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VCubemaF32Vop3 : public Vop3 {
public:
  VCubemaF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VBfeU32Vop3 : public Vop3 {
public:
  VBfeU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VBfeI32Vop3 : public Vop3 {
public:
  VBfeI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VBfiB32Vop3 : public Vop3 {
public:
  VBfiB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VFmaF32Vop3 : public Vop3 {
public:
  VFmaF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VFmaF64Vop3 : public Vop3 {
public:
  VFmaF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VLerpU8Vop3 : public Vop3 {
public:
  VLerpU8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VAlignbitB32Vop3 : public Vop3 {
public:
  VAlignbitB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VAlignbyteB32Vop3 : public Vop3 {
public:
  VAlignbyteB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMin3F32Vop3 : public Vop3 {
public:
  VMin3F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMin3I32Vop3 : public Vop3 {
public:
  VMin3I32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMin3U32Vop3 : public Vop3 {
public:
  VMin3U32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMax3F32Vop3 : public Vop3 {
public:
  VMax3F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMax3I32Vop3 : public Vop3 {
public:
  VMax3I32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMax3U32Vop3 : public Vop3 {
public:
  VMax3U32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMed3F32Vop3 : public Vop3 {
public:
  VMed3F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMed3I32Vop3 : public Vop3 {
public:
  VMed3I32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMed3U32Vop3 : public Vop3 {
public:
  VMed3U32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VSadU8Vop3 : public Vop3 {
public:
  VSadU8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VSadHiU8Vop3 : public Vop3 {
public:
  VSadHiU8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VSadU16Vop3 : public Vop3 {
public:
  VSadU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VSadU32Vop3 : public Vop3 {
public:
  VSadU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VCvtPkU8F32Vop3 : public Vop3 {
public:
  VCvtPkU8F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VDivFixupF32Vop3 : public Vop3 {
public:
  VDivFixupF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VDivFixupF64Vop3 : public Vop3 {
public:
  VDivFixupF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VDivScaleF32Vop3SdstEnc : public Vop3SdstEnc {
public:
  VDivScaleF32Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VDivScaleF64Vop3SdstEnc : public Vop3SdstEnc {
public:
  VDivScaleF64Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VDivFmasF32Vop3 : public Vop3 {
public:
  VDivFmasF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VDivFmasF64Vop3 : public Vop3 {
public:
  VDivFmasF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMsadU8Vop3 : public Vop3 {
public:
  VMsadU8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VQsadPkU16U8Vop3 : public Vop3 {
public:
  VQsadPkU16U8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMqsadPkU16U8Vop3 : public Vop3 {
public:
  VMqsadPkU16U8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMqsadU32U8Vop3 : public Vop3 {
public:
  VMqsadU32U8Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadU64U32Vop3SdstEnc : public Vop3SdstEnc {
public:
  VMadU64U32Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadI64I32Vop3SdstEnc : public Vop3SdstEnc {
public:
  VMadI64I32Vop3SdstEnc(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand sdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadLegacyF16Vop3 : public Vop3 {
public:
  VMadLegacyF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadLegacyU16Vop3 : public Vop3 {
public:
  VMadLegacyU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadLegacyI16Vop3 : public Vop3 {
public:
  VMadLegacyI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VPermB32Vop3 : public Vop3 {
public:
  VPermB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VFmaLegacyF16Vop3 : public Vop3 {
public:
  VFmaLegacyF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VDivFixupLegacyF16Vop3 : public Vop3 {
public:
  VDivFixupLegacyF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VCvtPkaccumU8F32Vop3 : public Vop3 {
public:
  VCvtPkaccumU8F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMadU32U16Vop3 : public Vop3 {
public:
  VMadU32U16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadI32I16Vop3 : public Vop3 {
public:
  VMadI32I16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VXadU32Vop3 : public Vop3 {
public:
  VXadU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMin3F16Vop3 : public Vop3 {
public:
  VMin3F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMin3I16Vop3 : public Vop3 {
public:
  VMin3I16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMin3U16Vop3 : public Vop3 {
public:
  VMin3U16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMax3F16Vop3 : public Vop3 {
public:
  VMax3F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMax3I16Vop3 : public Vop3 {
public:
  VMax3I16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMax3U16Vop3 : public Vop3 {
public:
  VMax3U16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMed3F16Vop3 : public Vop3 {
public:
  VMed3F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMed3I16Vop3 : public Vop3 {
public:
  VMed3I16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMed3U16Vop3 : public Vop3 {
public:
  VMed3U16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VLshlAddU32Vop3 : public Vop3 {
public:
  VLshlAddU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VAddLshlU32Vop3 : public Vop3 {
public:
  VAddLshlU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VAdd3U32Vop3 : public Vop3 {
public:
  VAdd3U32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VLshlOrB32Vop3 : public Vop3 {
public:
  VLshlOrB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VAndOrB32Vop3 : public Vop3 {
public:
  VAndOrB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VOr3B32Vop3 : public Vop3 {
public:
  VOr3B32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadF16Vop3 : public Vop3 {
public:
  VMadF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadU16Vop3 : public Vop3 {
public:
  VMadU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VMadI16Vop3 : public Vop3 {
public:
  VMadI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VFmaF16Vop3 : public Vop3 {
public:
  VFmaF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VDivFixupF16Vop3 : public Vop3 {
public:
  VDivFixupF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VLshlAddU64Vop3 : public Vop3 {
public:
  VLshlAddU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
  Operand src2;
};

class VAddF64Vop3 : public Vop3 {
public:
  VAddF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulF64Vop3 : public Vop3 {
public:
  VMulF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMinF64Vop3 : public Vop3 {
public:
  VMinF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMaxF64Vop3 : public Vop3 {
public:
  VMaxF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VLdexpF64Vop3 : public Vop3 {
public:
  VLdexpF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulLoU32Vop3 : public Vop3 {
public:
  VMulLoU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulHiU32Vop3 : public Vop3 {
public:
  VMulHiU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulHiI32Vop3 : public Vop3 {
public:
  VMulHiI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VLdexpF32Vop3 : public Vop3 {
public:
  VLdexpF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VReadlaneB32Vop3 : public Vop3 {
public:
  VReadlaneB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VWritelaneB32Vop3 : public Vop3 {
public:
  VWritelaneB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VBcntU32B32Vop3 : public Vop3 {
public:
  VBcntU32B32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMbcntLoU32B32Vop3 : public Vop3 {
public:
  VMbcntLoU32B32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMbcntHiU32B32Vop3 : public Vop3 {
public:
  VMbcntHiU32B32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VLshlrevB64Vop3 : public Vop3 {
public:
  VLshlrevB64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VLshrrevB64Vop3 : public Vop3 {
public:
  VLshrrevB64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VAshrrevI64Vop3 : public Vop3 {
public:
  VAshrrevI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VTrigPreopF64Vop3 : public Vop3 {
public:
  VTrigPreopF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VBfmB32Vop3 : public Vop3 {
public:
  VBfmB32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtPknormI16F32Vop3 : public Vop3 {
public:
  VCvtPknormI16F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtPknormU16F32Vop3 : public Vop3 {
public:
  VCvtPknormU16F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtPkrtzF16F32Vop3 : public Vop3 {
public:
  VCvtPkrtzF16F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtPkU16U32Vop3 : public Vop3 {
public:
  VCvtPkU16U32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtPkI16I32Vop3 : public Vop3 {
public:
  VCvtPkI16I32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtPknormI16F16Vop3 : public Vop3 {
public:
  VCvtPknormI16F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtPknormU16F16Vop3 : public Vop3 {
public:
  VCvtPknormU16F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VAddI32Vop3 : public Vop3 {
public:
  VAddI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubI32Vop3 : public Vop3 {
public:
  VSubI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VAddI16Vop3 : public Vop3 {
public:
  VAddI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VSubI16Vop3 : public Vop3 {
public:
  VSubI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VPackB32F16Vop3 : public Vop3 {
public:
  VPackB32F16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VMulLegacyF32Vop3 : public Vop3 {
public:
  VMulLegacyF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtPkFp8F32Vop3 : public Vop3 {
public:
  VCvtPkFp8F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtPkBf8F32Vop3 : public Vop3 {
public:
  VCvtPkBf8F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtSrFp8F32Vop3 : public Vop3 {
public:
  VCvtSrFp8F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCvtSrBf8F32Vop3 : public Vop3 {
public:
  VCvtSrBf8F32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpClassF32Vop3 : public Vop3 {
public:
  VCmpClassF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxClassF32Vop3 : public Vop3 {
public:
  VCmpxClassF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpClassF64Vop3 : public Vop3 {
public:
  VCmpClassF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxClassF64Vop3 : public Vop3 {
public:
  VCmpxClassF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpClassF16Vop3 : public Vop3 {
public:
  VCmpClassF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxClassF16Vop3 : public Vop3 {
public:
  VCmpxClassF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpFF16Vop3 : public Vop3 {
public:
  VCmpFF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLtF16Vop3 : public Vop3 {
public:
  VCmpLtF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpEqF16Vop3 : public Vop3 {
public:
  VCmpEqF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLeF16Vop3 : public Vop3 {
public:
  VCmpLeF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGtF16Vop3 : public Vop3 {
public:
  VCmpGtF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLgF16Vop3 : public Vop3 {
public:
  VCmpLgF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGeF16Vop3 : public Vop3 {
public:
  VCmpGeF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpOF16Vop3 : public Vop3 {
public:
  VCmpOF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpUF16Vop3 : public Vop3 {
public:
  VCmpUF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNgeF16Vop3 : public Vop3 {
public:
  VCmpNgeF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNlgF16Vop3 : public Vop3 {
public:
  VCmpNlgF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNgtF16Vop3 : public Vop3 {
public:
  VCmpNgtF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNleF16Vop3 : public Vop3 {
public:
  VCmpNleF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNeqF16Vop3 : public Vop3 {
public:
  VCmpNeqF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNltF16Vop3 : public Vop3 {
public:
  VCmpNltF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpTruF16Vop3 : public Vop3 {
public:
  VCmpTruF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxFF16Vop3 : public Vop3 {
public:
  VCmpxFF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLtF16Vop3 : public Vop3 {
public:
  VCmpxLtF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxEqF16Vop3 : public Vop3 {
public:
  VCmpxEqF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLeF16Vop3 : public Vop3 {
public:
  VCmpxLeF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGtF16Vop3 : public Vop3 {
public:
  VCmpxGtF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLgF16Vop3 : public Vop3 {
public:
  VCmpxLgF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGeF16Vop3 : public Vop3 {
public:
  VCmpxGeF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxOF16Vop3 : public Vop3 {
public:
  VCmpxOF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxUF16Vop3 : public Vop3 {
public:
  VCmpxUF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNgeF16Vop3 : public Vop3 {
public:
  VCmpxNgeF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNlgF16Vop3 : public Vop3 {
public:
  VCmpxNlgF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNgtF16Vop3 : public Vop3 {
public:
  VCmpxNgtF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNleF16Vop3 : public Vop3 {
public:
  VCmpxNleF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNeqF16Vop3 : public Vop3 {
public:
  VCmpxNeqF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNltF16Vop3 : public Vop3 {
public:
  VCmpxNltF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxTruF16Vop3 : public Vop3 {
public:
  VCmpxTruF16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpFF32Vop3 : public Vop3 {
public:
  VCmpFF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLtF32Vop3 : public Vop3 {
public:
  VCmpLtF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpEqF32Vop3 : public Vop3 {
public:
  VCmpEqF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLeF32Vop3 : public Vop3 {
public:
  VCmpLeF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGtF32Vop3 : public Vop3 {
public:
  VCmpGtF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLgF32Vop3 : public Vop3 {
public:
  VCmpLgF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGeF32Vop3 : public Vop3 {
public:
  VCmpGeF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpOF32Vop3 : public Vop3 {
public:
  VCmpOF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpUF32Vop3 : public Vop3 {
public:
  VCmpUF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNgeF32Vop3 : public Vop3 {
public:
  VCmpNgeF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNlgF32Vop3 : public Vop3 {
public:
  VCmpNlgF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNgtF32Vop3 : public Vop3 {
public:
  VCmpNgtF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNleF32Vop3 : public Vop3 {
public:
  VCmpNleF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNeqF32Vop3 : public Vop3 {
public:
  VCmpNeqF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNltF32Vop3 : public Vop3 {
public:
  VCmpNltF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpTruF32Vop3 : public Vop3 {
public:
  VCmpTruF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxFF32Vop3 : public Vop3 {
public:
  VCmpxFF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLtF32Vop3 : public Vop3 {
public:
  VCmpxLtF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxEqF32Vop3 : public Vop3 {
public:
  VCmpxEqF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLeF32Vop3 : public Vop3 {
public:
  VCmpxLeF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGtF32Vop3 : public Vop3 {
public:
  VCmpxGtF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLgF32Vop3 : public Vop3 {
public:
  VCmpxLgF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGeF32Vop3 : public Vop3 {
public:
  VCmpxGeF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxOF32Vop3 : public Vop3 {
public:
  VCmpxOF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxUF32Vop3 : public Vop3 {
public:
  VCmpxUF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNgeF32Vop3 : public Vop3 {
public:
  VCmpxNgeF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNlgF32Vop3 : public Vop3 {
public:
  VCmpxNlgF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNgtF32Vop3 : public Vop3 {
public:
  VCmpxNgtF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNleF32Vop3 : public Vop3 {
public:
  VCmpxNleF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNeqF32Vop3 : public Vop3 {
public:
  VCmpxNeqF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNltF32Vop3 : public Vop3 {
public:
  VCmpxNltF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxTruF32Vop3 : public Vop3 {
public:
  VCmpxTruF32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpFF64Vop3 : public Vop3 {
public:
  VCmpFF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLtF64Vop3 : public Vop3 {
public:
  VCmpLtF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpEqF64Vop3 : public Vop3 {
public:
  VCmpEqF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLeF64Vop3 : public Vop3 {
public:
  VCmpLeF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGtF64Vop3 : public Vop3 {
public:
  VCmpGtF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLgF64Vop3 : public Vop3 {
public:
  VCmpLgF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGeF64Vop3 : public Vop3 {
public:
  VCmpGeF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpOF64Vop3 : public Vop3 {
public:
  VCmpOF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpUF64Vop3 : public Vop3 {
public:
  VCmpUF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNgeF64Vop3 : public Vop3 {
public:
  VCmpNgeF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNlgF64Vop3 : public Vop3 {
public:
  VCmpNlgF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNgtF64Vop3 : public Vop3 {
public:
  VCmpNgtF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNleF64Vop3 : public Vop3 {
public:
  VCmpNleF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNeqF64Vop3 : public Vop3 {
public:
  VCmpNeqF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNltF64Vop3 : public Vop3 {
public:
  VCmpNltF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpTruF64Vop3 : public Vop3 {
public:
  VCmpTruF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxFF64Vop3 : public Vop3 {
public:
  VCmpxFF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLtF64Vop3 : public Vop3 {
public:
  VCmpxLtF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxEqF64Vop3 : public Vop3 {
public:
  VCmpxEqF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLeF64Vop3 : public Vop3 {
public:
  VCmpxLeF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGtF64Vop3 : public Vop3 {
public:
  VCmpxGtF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLgF64Vop3 : public Vop3 {
public:
  VCmpxLgF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGeF64Vop3 : public Vop3 {
public:
  VCmpxGeF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxOF64Vop3 : public Vop3 {
public:
  VCmpxOF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxUF64Vop3 : public Vop3 {
public:
  VCmpxUF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNgeF64Vop3 : public Vop3 {
public:
  VCmpxNgeF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNlgF64Vop3 : public Vop3 {
public:
  VCmpxNlgF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNgtF64Vop3 : public Vop3 {
public:
  VCmpxNgtF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNleF64Vop3 : public Vop3 {
public:
  VCmpxNleF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNeqF64Vop3 : public Vop3 {
public:
  VCmpxNeqF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNltF64Vop3 : public Vop3 {
public:
  VCmpxNltF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxTruF64Vop3 : public Vop3 {
public:
  VCmpxTruF64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpFI16Vop3 : public Vop3 {
public:
  VCmpFI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLtI16Vop3 : public Vop3 {
public:
  VCmpLtI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpEqI16Vop3 : public Vop3 {
public:
  VCmpEqI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLeI16Vop3 : public Vop3 {
public:
  VCmpLeI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGtI16Vop3 : public Vop3 {
public:
  VCmpGtI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNeI16Vop3 : public Vop3 {
public:
  VCmpNeI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGeI16Vop3 : public Vop3 {
public:
  VCmpGeI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpTI16Vop3 : public Vop3 {
public:
  VCmpTI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpFU16Vop3 : public Vop3 {
public:
  VCmpFU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLtU16Vop3 : public Vop3 {
public:
  VCmpLtU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpEqU16Vop3 : public Vop3 {
public:
  VCmpEqU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLeU16Vop3 : public Vop3 {
public:
  VCmpLeU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGtU16Vop3 : public Vop3 {
public:
  VCmpGtU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNeU16Vop3 : public Vop3 {
public:
  VCmpNeU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGeU16Vop3 : public Vop3 {
public:
  VCmpGeU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpTU16Vop3 : public Vop3 {
public:
  VCmpTU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxFI16Vop3 : public Vop3 {
public:
  VCmpxFI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLtI16Vop3 : public Vop3 {
public:
  VCmpxLtI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxEqI16Vop3 : public Vop3 {
public:
  VCmpxEqI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLeI16Vop3 : public Vop3 {
public:
  VCmpxLeI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGtI16Vop3 : public Vop3 {
public:
  VCmpxGtI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNeI16Vop3 : public Vop3 {
public:
  VCmpxNeI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGeI16Vop3 : public Vop3 {
public:
  VCmpxGeI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxTI16Vop3 : public Vop3 {
public:
  VCmpxTI16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxFU16Vop3 : public Vop3 {
public:
  VCmpxFU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLtU16Vop3 : public Vop3 {
public:
  VCmpxLtU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxEqU16Vop3 : public Vop3 {
public:
  VCmpxEqU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLeU16Vop3 : public Vop3 {
public:
  VCmpxLeU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGtU16Vop3 : public Vop3 {
public:
  VCmpxGtU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNeU16Vop3 : public Vop3 {
public:
  VCmpxNeU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGeU16Vop3 : public Vop3 {
public:
  VCmpxGeU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxTU16Vop3 : public Vop3 {
public:
  VCmpxTU16Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpFI32Vop3 : public Vop3 {
public:
  VCmpFI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLtI32Vop3 : public Vop3 {
public:
  VCmpLtI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpEqI32Vop3 : public Vop3 {
public:
  VCmpEqI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLeI32Vop3 : public Vop3 {
public:
  VCmpLeI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGtI32Vop3 : public Vop3 {
public:
  VCmpGtI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNeI32Vop3 : public Vop3 {
public:
  VCmpNeI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGeI32Vop3 : public Vop3 {
public:
  VCmpGeI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpTI32Vop3 : public Vop3 {
public:
  VCmpTI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpFU32Vop3 : public Vop3 {
public:
  VCmpFU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLtU32Vop3 : public Vop3 {
public:
  VCmpLtU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpEqU32Vop3 : public Vop3 {
public:
  VCmpEqU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLeU32Vop3 : public Vop3 {
public:
  VCmpLeU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGtU32Vop3 : public Vop3 {
public:
  VCmpGtU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNeU32Vop3 : public Vop3 {
public:
  VCmpNeU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGeU32Vop3 : public Vop3 {
public:
  VCmpGeU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpTU32Vop3 : public Vop3 {
public:
  VCmpTU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxFI32Vop3 : public Vop3 {
public:
  VCmpxFI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLtI32Vop3 : public Vop3 {
public:
  VCmpxLtI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxEqI32Vop3 : public Vop3 {
public:
  VCmpxEqI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLeI32Vop3 : public Vop3 {
public:
  VCmpxLeI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGtI32Vop3 : public Vop3 {
public:
  VCmpxGtI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNeI32Vop3 : public Vop3 {
public:
  VCmpxNeI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGeI32Vop3 : public Vop3 {
public:
  VCmpxGeI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxTI32Vop3 : public Vop3 {
public:
  VCmpxTI32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxFU32Vop3 : public Vop3 {
public:
  VCmpxFU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLtU32Vop3 : public Vop3 {
public:
  VCmpxLtU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxEqU32Vop3 : public Vop3 {
public:
  VCmpxEqU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLeU32Vop3 : public Vop3 {
public:
  VCmpxLeU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGtU32Vop3 : public Vop3 {
public:
  VCmpxGtU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNeU32Vop3 : public Vop3 {
public:
  VCmpxNeU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGeU32Vop3 : public Vop3 {
public:
  VCmpxGeU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxTU32Vop3 : public Vop3 {
public:
  VCmpxTU32Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpFI64Vop3 : public Vop3 {
public:
  VCmpFI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLtI64Vop3 : public Vop3 {
public:
  VCmpLtI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpEqI64Vop3 : public Vop3 {
public:
  VCmpEqI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLeI64Vop3 : public Vop3 {
public:
  VCmpLeI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGtI64Vop3 : public Vop3 {
public:
  VCmpGtI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNeI64Vop3 : public Vop3 {
public:
  VCmpNeI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGeI64Vop3 : public Vop3 {
public:
  VCmpGeI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpTI64Vop3 : public Vop3 {
public:
  VCmpTI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpFU64Vop3 : public Vop3 {
public:
  VCmpFU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLtU64Vop3 : public Vop3 {
public:
  VCmpLtU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpEqU64Vop3 : public Vop3 {
public:
  VCmpEqU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpLeU64Vop3 : public Vop3 {
public:
  VCmpLeU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGtU64Vop3 : public Vop3 {
public:
  VCmpGtU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpNeU64Vop3 : public Vop3 {
public:
  VCmpNeU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpGeU64Vop3 : public Vop3 {
public:
  VCmpGeU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpTU64Vop3 : public Vop3 {
public:
  VCmpTU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxFI64Vop3 : public Vop3 {
public:
  VCmpxFI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLtI64Vop3 : public Vop3 {
public:
  VCmpxLtI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxEqI64Vop3 : public Vop3 {
public:
  VCmpxEqI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLeI64Vop3 : public Vop3 {
public:
  VCmpxLeI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGtI64Vop3 : public Vop3 {
public:
  VCmpxGtI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNeI64Vop3 : public Vop3 {
public:
  VCmpxNeI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGeI64Vop3 : public Vop3 {
public:
  VCmpxGeI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxTI64Vop3 : public Vop3 {
public:
  VCmpxTI64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxFU64Vop3 : public Vop3 {
public:
  VCmpxFU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLtU64Vop3 : public Vop3 {
public:
  VCmpxLtU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxEqU64Vop3 : public Vop3 {
public:
  VCmpxEqU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxLeU64Vop3 : public Vop3 {
public:
  VCmpxLeU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGtU64Vop3 : public Vop3 {
public:
  VCmpxGtU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxNeU64Vop3 : public Vop3 {
public:
  VCmpxNeU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxGeU64Vop3 : public Vop3 {
public:
  VCmpxGeU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

class VCmpxTU64Vop3 : public Vop3 {
public:
  VCmpxTU64Vop3(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand src0;
  Operand src1;
};

} // namespace cdna3
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_VOP3_H_

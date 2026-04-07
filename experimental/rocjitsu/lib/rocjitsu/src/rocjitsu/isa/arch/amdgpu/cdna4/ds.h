// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_DS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_DS_H_

#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/operand.h"

namespace rocjitsu {
namespace cdna4 {

class DsAddU32Ds : public Ds {
public:
  DsAddU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsSubU32Ds : public Ds {
public:
  DsSubU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsRsubU32Ds : public Ds {
public:
  DsRsubU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsIncU32Ds : public Ds {
public:
  DsIncU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsDecU32Ds : public Ds {
public:
  DsDecU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMinI32Ds : public Ds {
public:
  DsMinI32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMaxI32Ds : public Ds {
public:
  DsMaxI32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMinU32Ds : public Ds {
public:
  DsMinU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMaxU32Ds : public Ds {
public:
  DsMaxU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsAndB32Ds : public Ds {
public:
  DsAndB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsOrB32Ds : public Ds {
public:
  DsOrB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsXorB32Ds : public Ds {
public:
  DsXorB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMskorB32Ds : public Ds {
public:
  DsMskorB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsWriteB32Ds : public Ds {
public:
  DsWriteB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsWrite2B32Ds : public Ds {
public:
  DsWrite2B32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsWrite2st64B32Ds : public Ds {
public:
  DsWrite2st64B32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsCmpstB32Ds : public Ds {
public:
  DsCmpstB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsCmpstF32Ds : public Ds {
public:
  DsCmpstF32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsMinF32Ds : public Ds {
public:
  DsMinF32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMaxF32Ds : public Ds {
public:
  DsMaxF32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsNopDs : public Ds {
public:
  DsNopDs(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
};

class DsAddF32Ds : public Ds {
public:
  DsAddF32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsPkAddF16Ds : public Ds {
public:
  DsPkAddF16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsPkAddBf16Ds : public Ds {
public:
  DsPkAddBf16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsWriteAddtidB32Ds : public Ds {
public:
  DsWriteAddtidB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand data0;
};

class DsWriteB8Ds : public Ds {
public:
  DsWriteB8Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsWriteB16Ds : public Ds {
public:
  DsWriteB16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsAddRtnU32Ds : public Ds {
public:
  DsAddRtnU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsSubRtnU32Ds : public Ds {
public:
  DsSubRtnU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsRsubRtnU32Ds : public Ds {
public:
  DsRsubRtnU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsIncRtnU32Ds : public Ds {
public:
  DsIncRtnU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsDecRtnU32Ds : public Ds {
public:
  DsDecRtnU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMinRtnI32Ds : public Ds {
public:
  DsMinRtnI32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMaxRtnI32Ds : public Ds {
public:
  DsMaxRtnI32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMinRtnU32Ds : public Ds {
public:
  DsMinRtnU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMaxRtnU32Ds : public Ds {
public:
  DsMaxRtnU32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsAndRtnB32Ds : public Ds {
public:
  DsAndRtnB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsOrRtnB32Ds : public Ds {
public:
  DsOrRtnB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsXorRtnB32Ds : public Ds {
public:
  DsXorRtnB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMskorRtnB32Ds : public Ds {
public:
  DsMskorRtnB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsWrxchgRtnB32Ds : public Ds {
public:
  DsWrxchgRtnB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsWrxchg2RtnB32Ds : public Ds {
public:
  DsWrxchg2RtnB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsWrxchg2st64RtnB32Ds : public Ds {
public:
  DsWrxchg2st64RtnB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsCmpstRtnB32Ds : public Ds {
public:
  DsCmpstRtnB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsCmpstRtnF32Ds : public Ds {
public:
  DsCmpstRtnF32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsMinRtnF32Ds : public Ds {
public:
  DsMinRtnF32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMaxRtnF32Ds : public Ds {
public:
  DsMaxRtnF32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsWrapRtnB32Ds : public Ds {
public:
  DsWrapRtnB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsAddRtnF32Ds : public Ds {
public:
  DsAddRtnF32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsReadB32Ds : public Ds {
public:
  DsReadB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsRead2B32Ds : public Ds {
public:
  DsRead2B32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsRead2st64B32Ds : public Ds {
public:
  DsRead2st64B32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadI8Ds : public Ds {
public:
  DsReadI8Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadU8Ds : public Ds {
public:
  DsReadU8Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadI16Ds : public Ds {
public:
  DsReadI16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadU16Ds : public Ds {
public:
  DsReadU16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsSwizzleB32Ds : public Ds {
public:
  DsSwizzleB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsPermuteB32Ds : public Ds {
public:
  DsPermuteB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsBpermuteB32Ds : public Ds {
public:
  DsBpermuteB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsAddU64Ds : public Ds {
public:
  DsAddU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsSubU64Ds : public Ds {
public:
  DsSubU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsRsubU64Ds : public Ds {
public:
  DsRsubU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsIncU64Ds : public Ds {
public:
  DsIncU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsDecU64Ds : public Ds {
public:
  DsDecU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMinI64Ds : public Ds {
public:
  DsMinI64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMaxI64Ds : public Ds {
public:
  DsMaxI64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMinU64Ds : public Ds {
public:
  DsMinU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMaxU64Ds : public Ds {
public:
  DsMaxU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsAndB64Ds : public Ds {
public:
  DsAndB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsOrB64Ds : public Ds {
public:
  DsOrB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsXorB64Ds : public Ds {
public:
  DsXorB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMskorB64Ds : public Ds {
public:
  DsMskorB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsWriteB64Ds : public Ds {
public:
  DsWriteB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsWrite2B64Ds : public Ds {
public:
  DsWrite2B64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsWrite2st64B64Ds : public Ds {
public:
  DsWrite2st64B64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsCmpstB64Ds : public Ds {
public:
  DsCmpstB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsCmpstF64Ds : public Ds {
public:
  DsCmpstF64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsMinF64Ds : public Ds {
public:
  DsMinF64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsMaxF64Ds : public Ds {
public:
  DsMaxF64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsWriteB8D16HiDs : public Ds {
public:
  DsWriteB8D16HiDs(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsWriteB16D16HiDs : public Ds {
public:
  DsWriteB16D16HiDs(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsReadU8D16Ds : public Ds {
public:
  DsReadU8D16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadU8D16HiDs : public Ds {
public:
  DsReadU8D16HiDs(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadI8D16Ds : public Ds {
public:
  DsReadI8D16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadI8D16HiDs : public Ds {
public:
  DsReadI8D16HiDs(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadU16D16Ds : public Ds {
public:
  DsReadU16D16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadU16D16HiDs : public Ds {
public:
  DsReadU16D16HiDs(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsAddF64Ds : public Ds {
public:
  DsAddF64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsAddRtnU64Ds : public Ds {
public:
  DsAddRtnU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsSubRtnU64Ds : public Ds {
public:
  DsSubRtnU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsRsubRtnU64Ds : public Ds {
public:
  DsRsubRtnU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsIncRtnU64Ds : public Ds {
public:
  DsIncRtnU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsDecRtnU64Ds : public Ds {
public:
  DsDecRtnU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMinRtnI64Ds : public Ds {
public:
  DsMinRtnI64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMaxRtnI64Ds : public Ds {
public:
  DsMaxRtnI64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMinRtnU64Ds : public Ds {
public:
  DsMinRtnU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMaxRtnU64Ds : public Ds {
public:
  DsMaxRtnU64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsAndRtnB64Ds : public Ds {
public:
  DsAndRtnB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsOrRtnB64Ds : public Ds {
public:
  DsOrRtnB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsXorRtnB64Ds : public Ds {
public:
  DsXorRtnB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMskorRtnB64Ds : public Ds {
public:
  DsMskorRtnB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsWrxchgRtnB64Ds : public Ds {
public:
  DsWrxchgRtnB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsWrxchg2RtnB64Ds : public Ds {
public:
  DsWrxchg2RtnB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsWrxchg2st64RtnB64Ds : public Ds {
public:
  DsWrxchg2st64RtnB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsCmpstRtnB64Ds : public Ds {
public:
  DsCmpstRtnB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsCmpstRtnF64Ds : public Ds {
public:
  DsCmpstRtnF64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
  Operand data1;
};

class DsMinRtnF64Ds : public Ds {
public:
  DsMinRtnF64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsMaxRtnF64Ds : public Ds {
public:
  DsMaxRtnF64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsReadB64Ds : public Ds {
public:
  DsReadB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsRead2B64Ds : public Ds {
public:
  DsRead2B64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsRead2st64B64Ds : public Ds {
public:
  DsRead2st64B64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsAddRtnF64Ds : public Ds {
public:
  DsAddRtnF64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsCondxchg32RtnB64Ds : public Ds {
public:
  DsCondxchg32RtnB64Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsReadAddtidB32Ds : public Ds {
public:
  DsReadAddtidB32Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
};

class DsPkAddRtnF16Ds : public Ds {
public:
  DsPkAddRtnF16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsPkAddRtnBf16Ds : public Ds {
public:
  DsPkAddRtnBf16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
  Operand data0;
};

class DsConsumeDs : public Ds {
public:
  DsConsumeDs(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
};

class DsAppendDs : public Ds {
public:
  DsAppendDs(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
};

class DsWriteB96Ds : public Ds {
public:
  DsWriteB96Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsWriteB128Ds : public Ds {
public:
  DsWriteB128Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand addr;
  Operand data0;
};

class DsReadB64TrB4Ds : public Ds {
public:
  DsReadB64TrB4Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadB96TrB6Ds : public Ds {
public:
  DsReadB96TrB6Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadB64TrB8Ds : public Ds {
public:
  DsReadB64TrB8Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadB64TrB16Ds : public Ds {
public:
  DsReadB64TrB16Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadB96Ds : public Ds {
public:
  DsReadB96Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

class DsReadB128Ds : public Ds {
public:
  DsReadB128Ds(const MachineInst *inst);
  void execute(amdgpu::Wavefront &wf) override;

private:
  Operand vdst;
  Operand addr;
};

} // namespace cdna4
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_DS_H_

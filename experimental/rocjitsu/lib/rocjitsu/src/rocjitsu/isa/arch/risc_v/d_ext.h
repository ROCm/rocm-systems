// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_D_EXT_H_
#define ROCJITSU_ISA_ARCH_RISC_V_D_EXT_H_

#include "rocjitsu/isa/arch/risc_v/encodings.h"
#include "rocjitsu/isa/arch/risc_v/operand.h"

#include <cstdint>

namespace rocjitsu {
namespace risc_v {
namespace detail {

// I-type FP load instruction

class FldInst : public IType {
public:
  explicit FldInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand offset;
};

// S-type FP store instruction

class FsdInst : public SType {
public:
  explicit FsdInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rs2_op;
  Operand rs1_op;
  Operand offset;
};

// R-type FP compute instructions (FPR -> FPR)

class FaddDInst : public RType {
public:
  explicit FaddDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FsubDInst : public RType {
public:
  explicit FsubDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FmulDInst : public RType {
public:
  explicit FmulDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FdivDInst : public RType {
public:
  explicit FdivDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FsgnjDInst : public RType {
public:
  explicit FsgnjDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FsgnjnDInst : public RType {
public:
  explicit FsgnjnDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FsgnjxDInst : public RType {
public:
  explicit FsgnjxDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FminDInst : public RType {
public:
  explicit FminDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FmaxDInst : public RType {
public:
  explicit FmaxDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

// R-type FP unary instruction (single source)

class FsqrtDInst : public RType {
public:
  explicit FsqrtDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type FP-to-int conversions (FPR -> GPR)

class FcvtWDInst : public RType {
public:
  explicit FcvtWDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtWuDInst : public RType {
public:
  explicit FcvtWuDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtLDInst : public RType {
public:
  explicit FcvtLDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtLuDInst : public RType {
public:
  explicit FcvtLuDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type int-to-FP conversions (GPR -> FPR)

class FcvtDWInst : public RType {
public:
  explicit FcvtDWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtDWuInst : public RType {
public:
  explicit FcvtDWuInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtDLInst : public RType {
public:
  explicit FcvtDLInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtDLuInst : public RType {
public:
  explicit FcvtDLuInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type FP-FP conversion instructions

class FcvtSDInst : public RType {
public:
  explicit FcvtSDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtDSInst : public RType {
public:
  explicit FcvtDSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type move/classify (FPR -> GPR)

class FmvXDInst : public RType {
public:
  explicit FmvXDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FclassDInst : public RType {
public:
  explicit FclassDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type move (GPR -> FPR)

class FmvDXInst : public RType {
public:
  explicit FmvDXInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type FP compare instructions (FPR -> GPR)

class FeqDInst : public RType {
public:
  explicit FeqDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FltDInst : public RType {
public:
  explicit FltDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FleDInst : public RType {
public:
  explicit FleDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

// R4-type fused multiply-add instructions

class FmaddDInst : public R4Type {
public:
  explicit FmaddDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
  Operand rs3;
};

class FmsubDInst : public R4Type {
public:
  explicit FmsubDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
  Operand rs3;
};

class FnmsubDInst : public R4Type {
public:
  explicit FnmsubDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
  Operand rs3;
};

class FnmaddDInst : public R4Type {
public:
  explicit FnmaddDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
  Operand rs3;
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_RISC_V_D_EXT_H_

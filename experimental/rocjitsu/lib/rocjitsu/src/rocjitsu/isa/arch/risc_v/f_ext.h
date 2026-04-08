// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_F_EXT_H_
#define ROCJITSU_ISA_ARCH_RISC_V_F_EXT_H_

#include "rocjitsu/isa/arch/risc_v/encodings.h"
#include "rocjitsu/isa/arch/risc_v/operand.h"

#include <cstdint>

namespace rocjitsu {
namespace risc_v {
namespace detail {

// I-type FP load instruction

class FlwInst : public IType {
public:
  explicit FlwInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand offset;
};

// S-type FP store instruction

class FswInst : public SType {
public:
  explicit FswInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rs2_op;
  Operand rs1_op;
  Operand offset;
};

// R-type FP compute instructions (FPR -> FPR)

class FaddSInst : public RType {
public:
  explicit FaddSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FsubSInst : public RType {
public:
  explicit FsubSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FmulSInst : public RType {
public:
  explicit FmulSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FdivSInst : public RType {
public:
  explicit FdivSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FsgnjSInst : public RType {
public:
  explicit FsgnjSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FsgnjnSInst : public RType {
public:
  explicit FsgnjnSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FsgnjxSInst : public RType {
public:
  explicit FsgnjxSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FminSInst : public RType {
public:
  explicit FminSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FmaxSInst : public RType {
public:
  explicit FmaxSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

// R-type FP unary instruction (single source)

class FsqrtSInst : public RType {
public:
  explicit FsqrtSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type FP-to-int conversions (FPR -> GPR)

class FcvtWSInst : public RType {
public:
  explicit FcvtWSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtWuSInst : public RType {
public:
  explicit FcvtWuSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtLSInst : public RType {
public:
  explicit FcvtLSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtLuSInst : public RType {
public:
  explicit FcvtLuSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type int-to-FP conversions (GPR -> FPR)

class FcvtSWInst : public RType {
public:
  explicit FcvtSWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtSWuInst : public RType {
public:
  explicit FcvtSWuInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtSLInst : public RType {
public:
  explicit FcvtSLInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FcvtSLuInst : public RType {
public:
  explicit FcvtSLuInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type move/classify (FPR -> GPR)

class FmvXWInst : public RType {
public:
  explicit FmvXWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

class FclassSInst : public RType {
public:
  explicit FclassSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type move (GPR -> FPR)

class FmvWXInst : public RType {
public:
  explicit FmvWXInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
};

// R-type FP compare instructions (FPR -> GPR)

class FeqSInst : public RType {
public:
  explicit FeqSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FltSInst : public RType {
public:
  explicit FltSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

class FleSInst : public RType {
public:
  explicit FleSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
};

// R4-type fused multiply-add instructions

class FmaddSInst : public R4Type {
public:
  explicit FmaddSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
  Operand rs3;
};

class FmsubSInst : public R4Type {
public:
  explicit FmsubSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
  Operand rs3;
};

class FnmsubSInst : public R4Type {
public:
  explicit FnmsubSInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd;
  Operand rs1;
  Operand rs2;
  Operand rs3;
};

class FnmaddSInst : public R4Type {
public:
  explicit FnmaddSInst(uint32_t raw);
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

#endif // ROCJITSU_ISA_ARCH_RISC_V_F_EXT_H_

// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_A_EXT_H_
#define ROCJITSU_ISA_ARCH_RISC_V_A_EXT_H_

#include "rocjitsu/isa/arch/risc_v/encodings.h"
#include "rocjitsu/isa/arch/risc_v/operand.h"

namespace rocjitsu {
namespace risc_v {
namespace detail {

class LrWInst : public RType {
public:
  LrWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1;
};

class ScWInst : public RType {
public:
  ScWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoswapWInst : public RType {
public:
  AmoswapWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoaddWInst : public RType {
public:
  AmoaddWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoxorWInst : public RType {
public:
  AmoxorWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoandWInst : public RType {
public:
  AmoandWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoorWInst : public RType {
public:
  AmoorWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmominWInst : public RType {
public:
  AmominWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmomaxWInst : public RType {
public:
  AmomaxWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmominuWInst : public RType {
public:
  AmominuWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmomaxuWInst : public RType {
public:
  AmomaxuWInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class LrDInst : public RType {
public:
  LrDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1;
};

class ScDInst : public RType {
public:
  ScDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoswapDInst : public RType {
public:
  AmoswapDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoaddDInst : public RType {
public:
  AmoaddDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoxorDInst : public RType {
public:
  AmoxorDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoandDInst : public RType {
public:
  AmoandDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmoorDInst : public RType {
public:
  AmoorDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmominDInst : public RType {
public:
  AmominDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmomaxDInst : public RType {
public:
  AmomaxDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmominuDInst : public RType {
public:
  AmominuDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class AmomaxuDInst : public RType {
public:
  AmomaxuDInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_RISC_V_A_EXT_H_

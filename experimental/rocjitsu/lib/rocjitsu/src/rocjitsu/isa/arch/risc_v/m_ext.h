// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_M_EXT_H_
#define ROCJITSU_ISA_ARCH_RISC_V_M_EXT_H_

#include "rocjitsu/isa/arch/risc_v/encodings.h"
#include "rocjitsu/isa/arch/risc_v/operand.h"

namespace rocjitsu {
namespace risc_v {
namespace detail {

class MulInst : public RType {
public:
  MulInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class MulhInst : public RType {
public:
  MulhInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class MulhsuInst : public RType {
public:
  MulhsuInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class MulhuInst : public RType {
public:
  MulhuInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class DivInst : public RType {
public:
  DivInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class DivuInst : public RType {
public:
  DivuInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class RemInst : public RType {
public:
  RemInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class RemuInst : public RType {
public:
  RemuInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class MulwInst : public RType {
public:
  MulwInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class DivwInst : public RType {
public:
  DivwInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class DivuwInst : public RType {
public:
  DivuwInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class RemwInst : public RType {
public:
  RemwInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

class RemuwInst : public RType {
public:
  RemuwInst(uint32_t raw);
  void execute(HartState &ctx) override;

private:
  Operand rd, rs1, rs2;
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_RISC_V_M_EXT_H_

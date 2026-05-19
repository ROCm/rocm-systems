// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_ZICSR_H_
#define ROCJITSU_ISA_ARCH_RISC_V_ZICSR_H_

#include "rocjitsu/isa/arch/risc_v/encodings.h"
#include "rocjitsu/isa/arch/risc_v/operand.h"

#include <cstdint>

namespace rocjitsu {
namespace risc_v {
namespace detail {

class CsrrwInst : public IType {
public:
  explicit CsrrwInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand csr;
};

class CsrrsInst : public IType {
public:
  explicit CsrrsInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand csr;
};

class CsrrcInst : public IType {
public:
  explicit CsrrcInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand rs1;
  Operand csr;
};

class CsrrwiInst : public IType {
public:
  explicit CsrrwiInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand uimm;
  Operand csr;
};

class CsrrsiInst : public IType {
public:
  explicit CsrrsiInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand uimm;
  Operand csr;
};

class CsrrciInst : public IType {
public:
  explicit CsrrciInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rd;
  Operand uimm;
  Operand csr;
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_RISC_V_ZICSR_H_

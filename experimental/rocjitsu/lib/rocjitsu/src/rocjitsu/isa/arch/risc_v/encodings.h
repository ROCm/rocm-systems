// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_ENCODINGS_H_
#define ROCJITSU_ISA_ARCH_RISC_V_ENCODINGS_H_

#include "rocjitsu/isa/arch/risc_v/isa.h"
#include "rocjitsu/isa/arch/risc_v/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include <cstdint>
#include <string>

namespace rocjitsu {
namespace risc_v {
namespace detail {

class RType : public IsaInstruction<Isa> {
public:
  RType(const std::string &mnemonic, uint32_t raw);

protected:
  const RTypeMachineInst inst_;
};

class IType : public IsaInstruction<Isa> {
public:
  IType(const std::string &mnemonic, uint32_t raw);
  int32_t imm() const;

protected:
  const ITypeMachineInst inst_;
};

class SType : public IsaInstruction<Isa> {
public:
  SType(const std::string &mnemonic, uint32_t raw);
  int32_t imm() const;

protected:
  const STypeMachineInst inst_;
};

class BType : public IsaInstruction<Isa> {
public:
  BType(const std::string &mnemonic, uint32_t raw);
  int32_t imm() const;

protected:
  const BTypeMachineInst inst_;
};

class UType : public IsaInstruction<Isa> {
public:
  UType(const std::string &mnemonic, uint32_t raw);
  int32_t imm() const;

protected:
  const UTypeMachineInst inst_;
};

class JType : public IsaInstruction<Isa> {
public:
  JType(const std::string &mnemonic, uint32_t raw);
  int32_t imm() const;

protected:
  const JTypeMachineInst inst_;
};

class R4Type : public IsaInstruction<Isa> {
public:
  R4Type(const std::string &mnemonic, uint32_t raw);

protected:
  const R4TypeMachineInst inst_;
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_RISC_V_ENCODINGS_H_

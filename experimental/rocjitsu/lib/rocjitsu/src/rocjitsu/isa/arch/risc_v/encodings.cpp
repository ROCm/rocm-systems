// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/encodings.h"

#include <bit>

namespace rocjitsu {
namespace risc_v {
namespace detail {

RType::RType(std::string_view mnemonic, uint32_t raw, ExecuteFn exec_fn)
    : IsaInstruction<Isa>(mnemonic, exec_fn), inst_(std::bit_cast<RTypeMachineInst>(raw)) {
  size_ = 4;
}

IType::IType(std::string_view mnemonic, uint32_t raw, ExecuteFn exec_fn)
    : IsaInstruction<Isa>(mnemonic, exec_fn), inst_(std::bit_cast<ITypeMachineInst>(raw)) {
  size_ = 4;
}

int32_t IType::imm() const {
  // Sign-extend 12-bit immediate
  return static_cast<int32_t>(inst_.imm11_0 << 20) >> 20;
}

SType::SType(std::string_view mnemonic, uint32_t raw, ExecuteFn exec_fn)
    : IsaInstruction<Isa>(mnemonic, exec_fn), inst_(std::bit_cast<STypeMachineInst>(raw)) {
  size_ = 4;
}

int32_t SType::imm() const {
  // Reassemble imm[11:5|4:0] and sign-extend from 12 bits
  uint32_t raw = (inst_.imm11_5 << 5) | inst_.imm4_0;
  return static_cast<int32_t>(raw << 20) >> 20;
}

BType::BType(std::string_view mnemonic, uint32_t raw, ExecuteFn exec_fn)
    : IsaInstruction<Isa>(mnemonic, exec_fn), inst_(std::bit_cast<BTypeMachineInst>(raw)) {
  size_ = 4;
}

int32_t BType::imm() const {
  // Reassemble imm[12|10:5|4:1|11] and sign-extend from 13 bits
  uint32_t raw =
      (inst_.imm12 << 12) | (inst_.imm11 << 11) | (inst_.imm10_5 << 5) | (inst_.imm4_1 << 1);
  return static_cast<int32_t>(raw << 19) >> 19;
}

UType::UType(std::string_view mnemonic, uint32_t raw, ExecuteFn exec_fn)
    : IsaInstruction<Isa>(mnemonic, exec_fn), inst_(std::bit_cast<UTypeMachineInst>(raw)) {
  size_ = 4;
}

int32_t UType::imm() const {
  // Upper immediate: imm[31:12] placed in bits [31:12]
  return static_cast<int32_t>(inst_.imm31_12 << 12);
}

JType::JType(std::string_view mnemonic, uint32_t raw, ExecuteFn exec_fn)
    : IsaInstruction<Isa>(mnemonic, exec_fn), inst_(std::bit_cast<JTypeMachineInst>(raw)) {
  size_ = 4;
}

int32_t JType::imm() const {
  // Reassemble imm[20|10:1|11|19:12] and sign-extend from 21 bits
  uint32_t raw =
      (inst_.imm20 << 20) | (inst_.imm19_12 << 12) | (inst_.imm11 << 11) | (inst_.imm10_1 << 1);
  return static_cast<int32_t>(raw << 11) >> 11;
}

R4Type::R4Type(std::string_view mnemonic, uint32_t raw, ExecuteFn exec_fn)
    : IsaInstruction<Isa>(mnemonic, exec_fn), inst_(std::bit_cast<R4TypeMachineInst>(raw)) {
  size_ = 4;
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

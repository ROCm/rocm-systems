// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/operand.h"

#include <format>

namespace rocjitsu {
namespace risc_v {
namespace detail {

Operand::Operand(int size_bits, OperandType opr_type, int encoding_value)
    : IsaOperand<Isa>(size_bits, opr_type, encoding_value) {}

std::string Operand::name() const {
  switch (opr_type_) {
  case OperandType::OPR_GPR:
    return "x" + std::to_string(encoding_value_);
  case OperandType::OPR_FPR:
    return "f" + std::to_string(encoding_value_);
  case OperandType::OPR_IMM:
    return std::to_string(encoding_value_);
  case OperandType::OPR_CSR:
    return std::format("0x{:x}", encoding_value_);
  }
  return std::to_string(encoding_value_);
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/risc_v/isa.h"
#include "rocjitsu/isa/operand.h"

#include <string>

namespace rocjitsu {
namespace risc_v {
namespace detail {

class Operand : public IsaOperand<Isa> {
public:
  Operand(int size_bits, OperandType opr_type, int encoding_value);
  std::string name() const override;
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

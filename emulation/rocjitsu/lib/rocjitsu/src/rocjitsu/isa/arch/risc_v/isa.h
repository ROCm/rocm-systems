// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/risc_v/operand_types.h"
#include "rocjitsu/vm/risc_v/hart_state.h"

namespace rocjitsu {
namespace risc_v {

class Decoder;

struct Isa {
  using Context = HartState;
  using Decoder = risc_v::Decoder;
  using OperandType = detail::OperandType;
};

} // namespace risc_v
} // namespace rocjitsu

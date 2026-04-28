// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file def_use_chain.h
/// @brief Instruction-level register def/use extraction for DBT dataflow.
///
/// @details This is the bridge between decoded instructions and the
/// CFG-aware liveness analysis described in docs/dbt_dbi_plan.md. Operand
/// membership determines direction: dst operands define registers, source
/// operands use registers. Operand::to_register_ref() determines which
/// register class and index are involved. That split keeps the generic
/// analysis free of ISA selector constants and avoids parsing disassembly text.

#pragma once

#include "rocjitsu/analysis/register_set.h"

#include <cstdint>

namespace rocjitsu {

class Instruction;

/// @brief Registers read and written by one decoded instruction.
class InstDefUse {
public:
  /// @brief Extract explicit operand refs and implicit architectural effects.
  /// @param inst Decoded instruction whose operands have stable lifetimes.
  /// @param wf_size Wavefront width in lanes; controls EXEC/VCC pair width.
  InstDefUse(const Instruction &inst, uint8_t wf_size);

  RegisterSet defs; ///< Registers overwritten by the instruction.
  RegisterSet uses; ///< Registers read before the instruction writes defs.
};

} // namespace rocjitsu

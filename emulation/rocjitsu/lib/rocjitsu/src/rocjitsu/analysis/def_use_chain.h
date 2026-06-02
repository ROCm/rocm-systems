// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file def_use_chain.h
/// @brief Instruction-level register def/use extraction for DBT dataflow.
///
/// @details This is the bridge between decoded instructions and CFG-aware
/// liveness. Operand membership determines direction: dst operands define
/// registers, source operands use registers. Operand::to_register_ref()
/// determines which register class and index are involved. Instruction
/// subclasses may also report hidden register effects through implicit hooks.

#pragma once

#include "rocjitsu/isa/register_set.h"

namespace rocjitsu {

class Instruction;

/// @brief Registers read and written by one decoded instruction.
class InstDefUse {
public:
  /// @brief Extract explicit operand register refs.
  /// @param inst Decoded instruction whose operands have stable lifetimes.
  InstDefUse(const Instruction &inst);

  RegisterSet defs;                        ///< Registers overwritten by the instruction.
  RegisterSet uses;                        ///< Registers read before the instruction writes defs.
  bool has_exec_masked_vector_def = false; ///< True if any vector def is predicated by EXEC.
  bool has_predicated_def = false;         ///< True if defs preserve old values on some paths.
};

} // namespace rocjitsu

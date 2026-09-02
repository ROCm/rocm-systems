// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_gfx11_target_ops.cpp
/// @brief Independent gfx11 proofs for emitted synchronization semantics.

#include "rocjitsu/code/patch/consan/consan_validation_target_ops.h"

#include "rocjitsu/isa/instruction.h"

namespace rocjitsu::consan_validation_target_detail {

bool validate_gfx11_dependency(ConSanDependencyKind, const Instruction &instruction) {
  const Operand *operand =
      instruction.num_src_operands() == 1 ? instruction.src_operand(0) : nullptr;
  return instruction.mnemonic() == "s_delay_alu" && operand && operand->encoding_value() == 9;
}

} // namespace rocjitsu::consan_validation_target_detail

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file control_flow.h
/// @brief Shared instruction-level control-flow predicates.

#pragma once

#include "rocjitsu/isa/instruction.h"

#include <cstdint>

namespace rocjitsu {

/// @brief Whether an instruction ends the current program path.
///
/// @details ROCr reserves trap ID 2 for assertion/abort handling. Unlike other
/// S_TRAP IDs, that path does not resume at the following instruction, so every
/// CFG construction pass must treat it like an ordinary PROGRAM_TERMINATOR.
[[nodiscard]] inline bool is_program_path_terminator(const Instruction &inst) {
  if ((inst.flags() & PROGRAM_TERMINATOR) != 0)
    return true;
  if (inst.mnemonic() != "s_trap" || inst.size() != sizeof(uint32_t))
    return false;
  const uint32_t *raw = inst.raw_encoding();
  return raw != nullptr && static_cast<uint16_t>(raw[0]) == 2;
}

} // namespace rocjitsu

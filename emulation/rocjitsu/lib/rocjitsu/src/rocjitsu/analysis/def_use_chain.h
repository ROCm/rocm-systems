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

#include <cstdint>
#include <optional>

namespace rocjitsu {

class Instruction;
class Gfx1250VgprMsbAnalysis;
class Operand;

/// @brief Return the wave-mode-adjusted physical destination for a wave mask.
///
/// @details Generated operands describe LLVM register classes or the maximum
/// encoded width, while the physical width of an explicit wave mask is one
/// SGPR in Wave32 and two SGPRs in Wave64. Returns nullopt when the destination
/// is not one of the instruction forms with a wave-sized scalar result.
[[nodiscard]] std::optional<RegisterRef> wave_mode_destination_ref(const Instruction &inst,
                                                                   const Operand &operand,
                                                                   int operand_index,
                                                                   uint32_t wavefront_size);

/// @brief How to represent a gfx1250 VGPR definition whose physical bank is unknown.
enum class UnknownVgprDefPolicy : uint8_t {
  Omit,      ///< Do not claim a must-write for liveness kill computation.
  ExpandAll, ///< Mark every possible physical tuple for whole-kernel usage scans.
};

/// @brief Registers read and written by one decoded instruction.
class InstDefUse {
public:
  /// @brief Extract explicit operand register refs.
  /// @param inst Decoded instruction whose operands have stable lifetimes.
  InstDefUse(const Instruction &inst, const Gfx1250VgprMsbAnalysis *vgpr_msb = nullptr,
             UnknownVgprDefPolicy unknown_vgpr_defs = UnknownVgprDefPolicy::Omit);

  RegisterSet defs;                        ///< Registers overwritten by the instruction.
  RegisterSet uses;                        ///< Registers read before the instruction writes defs.
  bool has_exec_masked_vector_def = false; ///< True if any vector def is predicated by EXEC.
  bool has_predicated_def = false;         ///< True if defs preserve old values on some paths.
};

} // namespace rocjitsu

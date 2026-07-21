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
///
/// Ordinary register-file effects (SGPR/VGPR/AccVGPR) live in `defs`/`uses`
/// and drive scratch-allocation liveness. Architectural special-register
/// effects (EXEC/VCC/SCC/M0/PC/...) and memory effects are exposed on separate
/// members (`special_defs`/`special_uses`, `memory`) so consumers can query
/// them without those effects ever entering ordinary liveness. Those separate
/// members are currently always empty and will be populated once generated
/// operands report special/memory effects; the ordinary `defs`/`uses` behavior
/// is unchanged.

#pragma once

#include "rocjitsu/isa/register_set.h"

namespace rocjitsu {

class Instruction;
class Gfx1250VgprMsbAnalysis;

/// @brief Memory-effect summary for one decoded instruction.
///
/// @details A coarse description of what memory an instruction touches. Not
/// currently used. Present now so `InstDefUse` has a stable shape for DBT/DBI
/// consumers. All flags default to false (no memory effect).
///
/// TODO: Update when OPR_GPUMEM, OPR_DSMEM, and OPR_FLAT_SCRATCH become
/// avaialable.
struct MemoryEffects {
  bool reads = false;   ///< Reads memory.
  bool writes = false;  ///< Writes memory.
  bool atomic = false;  ///< Performs an atomic read-modify-write (also sets reads+writes).
  bool global = false;  ///< Touches global/generic memory (OPR_GPUMEM).
  bool lds = false;     ///< Touches LDS/GDS (OPR_DSMEM).
  bool scratch = false; ///< Touches scratch/flat-scratch (OPR_FLAT_SCRATCH).

  /// @brief True if the instruction has any memory effect.
  [[nodiscard]] bool any() const { return reads || writes || atomic || global || lds || scratch; }

  friend bool operator==(const MemoryEffects &, const MemoryEffects &) = default;
};

/// @brief Registers read and written by one decoded instruction.
class InstDefUse {
public:
  /// @brief Extract explicit operand register refs.
  /// @param inst Decoded instruction whose operands have stable lifetimes.
  InstDefUse(const Instruction &inst, const Gfx1250VgprMsbAnalysis *vgpr_msb = nullptr);

  RegisterSet defs;                ///< Ordinary registers overwritten by the instruction.
  RegisterSet uses;                ///< Ordinary registers read before the instruction writes defs.
  SpecialRegisterSet special_defs; ///< Special registers (EXEC/VCC/SCC/M0/PC/...) written.
  SpecialRegisterSet special_uses; ///< Special registers read.
  MemoryEffects memory;            ///< Memory-effect summary.
  bool has_exec_masked_vector_def = false; ///< True if any vector def is predicated by EXEC.
  bool has_predicated_def = false;         ///< True if defs preserve old values on some paths.
};

} // namespace rocjitsu

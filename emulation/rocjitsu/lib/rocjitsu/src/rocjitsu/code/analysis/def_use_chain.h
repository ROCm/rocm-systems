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
/// `defs`/`uses` hold both ordinary register-file effects (SGPR/VGPR/AccVGPR)
/// and architectural special-register effects (EXEC/VCC/SCC/M0/PC/...) in one
/// RegisterSet each. Special registers are singleton members of that set;
/// consumers that drive scratch-allocation liveness project them out with
/// `RegisterSet::ordinary_only()` so special state never looks allocatable.
///
/// Special-register effects are surfaced only from *field-less* special operands
/// — dedicated operand types (OPR_PC, OPR_SDST_EXEC, OPR_SDST_M0,
/// OPR_SSRC_SPECIAL_SCC, OPR_VCC) whose `to_special_reg_class()` names the class
/// directly. A special register named through a *generic selector* field instead
/// (e.g. `s_mov_b32 exec_lo, 0`, where EXEC_LO is encoding value 126 in the
/// OPR_SDST selector) *does* decode to a special `RegisterRef` via
/// `to_register_ref()`, but that ref collapses the LO/HI halves into one
/// class-wide singleton — an imprecise representation — so InstDefUse drops it:
/// only ordinary SGPR/VGPR/AccVGPR refs
/// from `to_register_ref()` are recorded, and such selector-encoded specials
/// appear in neither `defs`/`uses` nor the ordinary projection. Surfacing them
/// precisely is follow-up work (a generator change mapping special selector
/// values to their RegClass, plus width-aware special storage); until then,
/// consumers must not assume selector-encoded EXEC/VCC/M0 reads or writes are
/// visible here.

#pragma once

#include "rocjitsu/isa/register_set.h"

namespace rocjitsu {

class Instruction;
class Gfx1250VgprMsbAnalysis;

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

  RegisterSet defs; ///< Registers written (ordinary lanes + special singletons).
  RegisterSet uses; ///< Registers read before defs (ordinary lanes + special singletons).
  bool has_exec_masked_vector_def =
      false; ///< True if the instruction doesn't ignore EXEC and has a vector def.
  bool has_predicated_def = false; ///< True if defs preserve old values on some paths.
};

} // namespace rocjitsu

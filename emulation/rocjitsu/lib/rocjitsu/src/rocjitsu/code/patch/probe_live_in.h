// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file probe_live_in.h
/// @brief Live-in analysis for a probe body.
///
/// This unit answers the complement of `probe_clobber.h`: what does the probe
/// body read before defining it? That set is the probe's input register
/// footprint.
///
/// Which of those inputs the framework supplies depends on the ABI in force, so
/// the analysis subtracts `supplied_registers()` and never enumerates the
/// conventions itself. Today that subtraction is the return-link pair, which
/// every convention supplies by construction.
///
/// Scope: ordinary registers — SGPR, VGPR, AccVGPR — read before being defined.
/// That is the shape of a value the kernel prologue supplies and an arbitrary
/// instrumentation site does not, which is the dependence this analysis exists
/// to find.
///
/// Special state (EXEC, VCC, SCC, M0, FLAT_SCRATCH) is the trampoline's to
/// handle, not this analysis's: `RegisterSet` cannot represent it
/// (`isa/register_set.h`) and `LivenessAnalysis` does not model it. A probe
/// reading special state it never wrote is therefore accepted here — including
/// SCC, which the call envelope's address arithmetic clobbers outright, and M0,
/// which CDNA `ds_*` reads implicitly. Neither is a gap in this gate.
///
/// The one thing this scope does oblige is that the encoded operands name every
/// ordinary register the body reads. Relative and GPR-indexed forms break that
/// by displacing an index at runtime, so they are rejected rather than analyzed:
/// the VGPR forms via `LivenessAnalysis::global_vgpr_usage_is_complete()`, the
/// scalar `s_movrel*` family via a local scan, since liveness models no scalar
/// equivalent.

#pragma once

#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <optional>
#include <string>

namespace rocjitsu {

/// @brief Registers @p sym reads before defining that @p abi does not supply.
///
/// @details Builds the probe object's CFG, forms the block scope reachable from
/// the probe entry, runs `LivenessAnalysis` over that scope, and subtracts
/// `supplied_registers(abi)` from the entry block's live-in set.
///
/// A non-empty result is a probe that cannot be called from an arbitrary site:
/// it wants values that exist only at kernel entry, which no trampoline can
/// reproduce. What an ABI supplies is `supplied_registers()`'s to say, so this
/// analysis never enumerates the conventions itself.
///
/// CFG liveness rather than a linear scan of the body words: a def under a
/// forward branch does not reach a use after the join, so a linear "used before
/// defined" walk would report a smaller footprint than the body really has.
///
/// A vector write counts here as writing the whole register, even though EXEC
/// masks it to the active lanes. The probe is responsible for its own EXEC; see
/// `LivenessAnalysisOptions::exec_masked_defs_kill` for what that obliges.
///
/// Returns std::nullopt (with a reason written to @p error_out, if non-null)
/// when the analysis could not run: an @p abi that fails is_valid_probe_abi(),
/// a probe object whose `.text` layout the scope walk cannot address, no decoder
/// for @p arch, a decode failure, a probe entry that does not start a decoded
/// block, or a body whose relative / GPR-indexed VGPR access or relative SGPR
/// access means its encoded operands do not name every register it reads.
[[nodiscard]] std::optional<RegisterSet>
analyze_probe_live_ins(const AmdGpuCodeObject &probe_obj, const ResolvedProbeSymbol &sym,
                       rj_code_arch_t arch, const ProbeAbi &abi, std::string *error_out = nullptr);

/// @brief Render @p regs as a comma-separated operand list, e.g. "s4, v0, v1".
///
/// @details Lives here because the rejection message naming the offending
/// registers is this analysis's only consumer today.
[[nodiscard]] std::string format_register_set(const RegisterSet &regs);

} // namespace rocjitsu

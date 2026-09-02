// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file probe_live_in.h
/// @brief Live-in analysis for a probe body.
///
/// This unit answers the complement of `probe_clobber.h`: what does the probe
/// body read before defining it? That set is the probe's input register
/// footprint.
///
/// Deciding which of those inputs the framework is willing to supply is the
/// caller's, because it depends on the calling convention in force. The
/// analysis reports the footprint and subtracts only the return-link pair,
/// which every convention supplies by construction.
///
/// Ordinary registers only — SGPR, VGPR, AccVGPR. Special state (EXEC, VCC,
/// SCC, M0, FLAT_SCRATCH) is outside the `LivenessAnalysis` dataflow model
/// (`analysis/liveness.h`) and `RegisterSet` has no bits for it, so an implicit
/// dependence on special state is neither detected nor reported here. That is a
/// deliberate scope limit, not an oversight: covering it needs a read-side
/// special-state analysis that does not exist. Note in particular that GFX9 /
/// CDNA `ds_*` instructions implicitly read M0, which a kernel prologue
/// initializes, so an LDS-using probe carries an input this analysis is blind
/// to.

#pragma once

#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <optional>
#include <string>

namespace rocjitsu {

/// @brief Registers @p sym reads before defining that @p cc does not supply.
///
/// @details Builds the probe object's CFG, forms the block scope reachable from
/// the probe entry, runs `LivenessAnalysis` over that scope, and subtracts the
/// registers @p cc supplies to a callee from the entry block's live-in set.
/// Today that is the s[30:31] return link alone, which the body's own
/// `s_setpc_b64` reads and the trampoline always materializes.
///
/// A non-empty result is a probe that cannot be called from an arbitrary site:
/// it wants values that exist only at kernel entry, which no trampoline can
/// reproduce. Keeping the subtraction here rather than in the caller keeps all
/// per-convention register knowledge in one place.
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
/// when the analysis could not run: an unrecognized convention, a probe object
/// whose `.text` layout the scope walk cannot address, no decoder for @p arch, a
/// decode failure, a probe entry that does not start a decoded block, or a body
/// whose relative / GPR-indexed VGPR access means its encoded operands do not
/// name every register it reads.
[[nodiscard]] std::optional<RegisterSet> analyze_probe_live_ins(const AmdGpuCodeObject &probe_obj,
                                                                const ResolvedProbeSymbol &sym,
                                                                rj_code_arch_t arch,
                                                                ProbeCallingConvention cc,
                                                                std::string *error_out = nullptr);

/// @brief Render @p regs as a comma-separated operand list, e.g. "s4, v0, v1".
///
/// @details Lives here because the rejection message naming the offending
/// registers is this analysis's only consumer today.
[[nodiscard]] std::string format_register_set(const RegisterSet &regs);

} // namespace rocjitsu

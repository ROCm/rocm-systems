// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file probe_clobber.h
/// @brief Callee-body clobber analysis for a copied probe.
///
/// This unit answers exactly one question: what does the probe *body* itself
/// overwrite? It is deliberately scoped to callee facts.

#pragma once

#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/isa/register_set.h"

#include <optional>
#include <string>

namespace rocjitsu {

/// @brief Registers and special state a probe body overwrites.
///
/// @details `ordinary_clobbers` is derived precisely by decoding the copied
/// probe body and unioning each instruction's ordinary defs (SGPR/VGPR/AccVGPR).
/// The special-state booleans are best-effort: they are set when an explicit
/// operand names that register class, but implicit effects (e.g. SCC written
/// as a side effect of scalar ALU) are not modeled here.
///
/// This struct is intended to stay callee-only.
struct ProbeClobberSummary {
  RegisterSet ordinary_clobbers;
  bool touches_exec = false;
  bool touches_vcc = false;
  bool touches_scc = false;
  bool touches_m0 = false;
  bool touches_flat_scratch = false;
  bool uses_private_segment = false;
};

/// @brief Decode @p callable's copied body and summarize what it clobbers.
///
/// Returns nullopt (and sets @p error_out) only on a structural decode failure
/// e.g. a missing decoder for the arch, an instruction that fails to decode,
/// or body that runs off its end. A successful decode always yields a summary
[[nodiscard]] std::optional<ProbeClobberSummary>
build_probe_clobber_summary(const ProbeCallable &callable, std::string *error_out = nullptr);

} // namespace rocjitsu

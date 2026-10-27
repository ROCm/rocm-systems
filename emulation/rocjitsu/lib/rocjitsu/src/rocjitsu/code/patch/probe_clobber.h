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

/// @brief Per-register proof for the entire copied probe body.
/// @details Only NoWrite permits omitting preservation. MayWrite records a
/// modeled possible write; Unknown includes missing effect coverage or an
/// unproved control transfer. Neither touches_* == false nor absent flags prove NoWrite.
enum class ProbeWriteProof { Unknown, MayWrite, NoWrite };

/// @brief Ordinary clobbers, special-state accesses and whole-body mask-write proofs.
///
/// @details `ordinary_clobbers` is derived precisely by decoding the copied
/// probe body and unioning each instruction's ordinary defs (SGPR/VGPR/AccVGPR).
/// `touches_*` are best-effort accesses exposed as operands, including reads;
/// they do not establish the absence of implicit effects. Independently,
/// `exec_write` and `vcc_write` combine generated implicit-write coverage,
/// actual destinations and body-control validation. Only NoWrite proves absence
/// of all writes; MayWrite and Unknown do not authorize omitting preservation.
///
/// This struct is intended to stay callee-only.
struct ProbeClobberSummary {
  RegisterSet ordinary_clobbers;
  // Cached once per distinct probe, before resource planning or layout.
  // Each proof uses generated coverage plus actual destinations independently.
  ProbeWriteProof exec_write = ProbeWriteProof::Unknown;
  ProbeWriteProof vcc_write = ProbeWriteProof::Unknown;
  // Best-effort access information, including reads. These flags are not
  // no-write proofs and do not control EXEC/VCC preservation.
  bool touches_exec = false;
  bool touches_vcc = false;
  bool touches_scc = false;
  bool touches_m0 = false;
  bool touches_flat_scratch = false;
  bool uses_private_segment = false;
};

/// @brief Decode @p callable's copied body and summarize what it clobbers.
///
/// Returns nullopt (and sets @p error_out) for a target/architecture mismatch,
/// missing decoder or rejected/truncated encoding. Successful decoding yields
/// a summary even when effects, the ABI or control flow cannot prove NoWrite.
[[nodiscard]] std::optional<ProbeClobberSummary>
build_probe_clobber_summary(const ProbeCallable &callable, std::string *error_out = nullptr);

} // namespace rocjitsu

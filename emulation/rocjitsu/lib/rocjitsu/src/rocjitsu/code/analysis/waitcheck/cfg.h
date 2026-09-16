// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file cfg.h
/// @brief Ordinary memory dependencies over reachable direct control flow.

#include "rocjitsu/code/analysis/waitcheck/stream.h"
#include "rocjitsu/code/basic_block.h"

namespace rocjitsu {
struct WaitcheckCfgOptions {
  WaitcheckStreamOptions entry_modes = {};
  /// @brief Hard bound on block transfers, including revisits. Zero is invalid.
  size_t max_block_visits = 100000;
  /// @brief Optional permitted text ranges, with BasicBlock's range contract.
  std::span<const BasicBlock::CodeRange> permitted_ranges = {};
};

/// @brief Stable-state memory diagnostics and every encountered coverage boundary.
/// @details Both outcomes of direct conditional branches are possible. No path
/// feasibility or call contexts are modeled. Calls, indirect branches and deferred
/// instruction families stop propagation on that path and record incomplete;
/// other paths continue. Decoded callee bodies are not analyzed unless supplied
/// as separate external entries; such entries assume no incoming pending operations.
/// Missing CFG edges also record incomplete. No diagnostics
/// does not certify scheduling, async/tensor ordering or program-end waits.
struct WaitcheckCfgReport {
  std::vector<WaitcheckMemoryDiagnostic> diagnostics;
  std::vector<WaitcheckStreamStop> incomplete;
  size_t blocks_analyzed = 0;
  size_t instructions_analyzed = 0;
};

/// @brief Build reachable CFGs and solve ordinary memory dependencies to a fixed point.
/// @details Entries are section-relative external roots in a single text section.
/// Each contributes a fresh state even when it has backedges or other predecessors;
/// live-ins have no pending operations and VGPR banking starts at zero. Empty entries
/// analyze nothing. Reuses shared CFG decoding and discovery; finding indirect
/// targets does not certify coverage. Diagnostics are emitted only after convergence,
/// once per static consumer/event, in address order. Malformed input, invalid modes,
/// and exhausted visit budgets return FailureOr errors, never a successful report.
[[nodiscard]] util::FailureOr<WaitcheckCfgReport>
analyze_waitcheck_cfg(const CodeObject &object, rj_code_arch_t arch,
                      std::span<const uint64_t> entries, WaitcheckCfgOptions options = {},
                      const util::DiagnosticEmitter &emit_error = {});
} // namespace rocjitsu

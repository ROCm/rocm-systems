// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file transfer.h
/// @brief Internal instruction transfer shared by stream and CFG memory analysis.

#include "rocjitsu/code/analysis/waitcheck/state.h"
#include "rocjitsu/code/analysis/waitcheck/stream.h"

namespace rocjitsu::waitcheck_detail {
struct MemoryTransferState {
  PendingState pending;
  // Uses of otherwise available live-ins establish older values locally.
  // Availability must hold on every incoming path, so intersect it at joins.
  RegisterSet local_ready;
  bool operator==(const MemoryTransferState &) const = default;
};

util::Result validate_memory_options(rj_code_arch_t arch, WaitcheckStreamOptions options,
                                     const util::DiagnosticEmitter &emit_error);

/// @brief Transfer an instruction through ordinary memory dependency state.
/// @details A null report suppresses diagnostics during fixed-point iteration. A stop
/// excludes this instruction from transfer; control-flow operands are still checked.
util::FailureOr<std::optional<WaitcheckStreamStop>>
transfer_memory_instruction(MemoryTransferState &state, const Instruction &inst,
                            rj_code_arch_t arch, WaitcheckStreamOptions options,
                            bool allow_direct_branches, WaitcheckStreamReport *report);
} // namespace rocjitsu::waitcheck_detail

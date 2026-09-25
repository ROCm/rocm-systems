// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file stream.h
/// @brief Ordinary memory dependency analysis for instruction streams.

#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "util/diagnostic.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

class IsaTargetRegistry;

/// @brief Consumer access requiring completion of a pending memory event.
enum class WaitcheckAccess { Read, Write };

/// @brief Missing wait with byte offsets and owned producer/consumer text.
struct WaitcheckMemoryDiagnostic {
  uint64_t producer_offset = 0;
  uint64_t consumer_offset = 0;
  WaitCounterKind counter = WaitCounterKind::Load;
  RegisterRef reg;
  WaitcheckAccess access = WaitcheckAccess::Read;
  uint32_t required_count = 0;
  std::string producer;
  std::string consumer;
  std::string required_wait;
};

/// @brief Entry modes supplied by the caller, normally from kernel metadata.
struct WaitcheckStreamOptions {
  /// @brief Wave width in lanes: 32 or 64; CDNA3/4 require 64.
  uint32_t wave_size = 64;
  /// @brief Enable memory source-lifetime checks for expert scheduling mode.
  bool expert_scheduling = false;
};

/// @brief First instruction excluded from analysis and the reason for stopping.
struct WaitcheckStreamStop {
  uint64_t offset = 0;
  std::string instruction;
  std::string reason;
};

/// @brief Ordinary memory/register dependencies in a straight-line stream.
/// @details Does not certify VALU/SGPR scheduling, async/tensor ordering, counter
/// parity, program-end waits or control flow. Trap temporaries and unrepresented
/// scalar-memory destinations stop analysis as incomplete. An empty diagnostics vector alone
/// is not a clean bill of health: inspect incomplete and this limited scope.
struct WaitcheckStreamReport {
  std::vector<WaitcheckMemoryDiagnostic> diagnostics;
  std::optional<WaitcheckStreamStop> incomplete;
  size_t instructions_analyzed = 0;
};

/// @brief Analyze from a fresh entry state to the first program end or input boundary.
/// @details Live-in registers have no outstanding memory operations. VGPR banking
/// starts at zero. Control-flow operands are checked before stopping; other
/// deferred instructions are excluded entirely. Both record incomplete before
/// transferring state, preserving earlier diagnostics. Bytes after a program
/// end are outside the scope. Invalid options/targets or malformed encodings
/// return failure and emit an error, distinct from a hazard report or incomplete
/// analysis. Offsets are bytes from
/// the start of words; the returned report owns all diagnostic text.
/// The concrete target selects instruction features; its registry descriptor
/// supplies the architecture used for dependency analysis.
[[nodiscard]] util::FailureOr<WaitcheckStreamReport>
analyze_waitcheck_stream(std::span<const uint32_t> words, const IsaTargetRegistry &registry,
                         rj_code_target_id_t target, WaitcheckStreamOptions options,
                         const util::DiagnosticEmitter &emit_error = {});

/// @brief Analyze using the architecture's default target.
/// @details Has the same analysis contract as the concrete-target overload, but
/// cannot decode instructions specific to another target in the same architecture.
/// Callers with concrete target metadata should use the registry/target overload.
[[nodiscard]] util::FailureOr<WaitcheckStreamReport>
analyze_waitcheck_stream(std::span<const uint32_t> words, rj_code_arch_t arch,
                         WaitcheckStreamOptions options,
                         const util::DiagnosticEmitter &emit_error = {});

} // namespace rocjitsu

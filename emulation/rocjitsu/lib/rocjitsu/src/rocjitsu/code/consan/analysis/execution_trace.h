// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/capture.h"
#include "util/diagnostic.h"

namespace rocjitsu::consan {

struct ExecutionTraceEntry {
  EntryKey key = 0;
  std::string name;
  EntryDisposition disposition = EntryDisposition::Native;
};

struct ExecutionTraceExecution {
  uint64_t serial = 0;
  EntryKey entry_key = 0;
  std::string entry;
  LaunchDimensions dimensions;
  uint32_t expected_rows = 0;
  uint32_t completed_rows = 0;
  uint64_t entry_events = 0;
  uint64_t exit_events = 0;
  uint32_t integrity_events = 0;
  // Bounded examples; integrity_events counts every malformed row/header.
  std::vector<std::string> issues;
};

struct ExecutionTraceResult {
  std::array<uint8_t, 16> run_identity{};
  ImageIdentity original;
  Digest selection_profile{};
  std::vector<ExecutionTraceEntry> entries;
  std::vector<ExecutionTraceExecution> executions;
  uint64_t expected_rows = 0;
  uint64_t completed_rows = 0;
  uint64_t entry_events = 0;
  uint64_t exit_events = 0;
  bool transport_complete = false;
};

/// @brief Validate saved entry/exit transport without performing LDS analysis.
/// @details Invalid static facts/envelopes reject the input. Malformed GPU rows
/// remain explicit transport gaps. No saved address is dereferenced.
[[nodiscard]] FailureOr<ExecutionTraceResult>
analyze_execution_trace_capture(const Capture &capture, util::DiagnosticEmitter diagnostic = {});

[[nodiscard]] std::string render_execution_trace(const ExecutionTraceResult &result);

} // namespace rocjitsu::consan

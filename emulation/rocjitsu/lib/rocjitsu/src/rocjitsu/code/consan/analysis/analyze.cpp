// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/analyze.h"

#include "util/utf8.h"
#include <algorithm>

namespace rocjitsu::consan {
FailureOr<LdsRaceResult> analyze_capture(const Capture &capture, const AnalysisOptions &options,
                                         ContractError *error) {
  if (error != nullptr)
    *error = {};
  if (validate_manifest(capture.header.manifest, error).failed())
    return Result::failure();
  // Capture names are opaque bytes; the JSON-facing analyzer requires text.
  if (!util::is_valid_utf8(capture.header.manifest.entry.name)) {
    if (error != nullptr)
      *error = {ContractErrorCode::InvalidFacts, "entry name is not valid UTF-8 for JSON output"};
    return Result::failure();
  }
  if (capture.header.execution_conditions != kSupportedExecutionConditions) {
    if (error != nullptr)
      *error = {ContractErrorCode::InvalidFacts, "unsupported capture execution conditions"};
    return Result::failure();
  }
  if (std::ranges::all_of(capture.header.run_identity, [](uint8_t value) { return value == 0; })) {
    if (error != nullptr)
      *error = {ContractErrorCode::InvalidFacts, "missing capture run identity"};
    return Result::failure();
  }
  if (capture.executions.size() > CaptureLimits{}.executions) {
    if (error != nullptr)
      *error = {ContractErrorCode::InvalidFacts, "capture execution budget exceeded"};
    return Result::failure();
  }
  if (options.examples > kMaxFindingExamples) {
    if (error != nullptr)
      *error = {ContractErrorCode::InvalidFacts, "finding example budget exceeded"};
    return Result::failure();
  }
  LdsRaceResult result;
  result.run_identity = capture.header.run_identity;
  result.original = capture.header.manifest.original;
  result.entry = capture.header.manifest.entry.name;
  result.execution_conditions = capture.header.execution_conditions;
  result.target = capture.header.manifest.target;
  result.wave_size = capture.header.manifest.wave_size;
  result.scope = capture.header.manifest.scope;
  result.coverage_complete = !capture.executions.empty();
  for (const CapturedExecution &execution : capture.executions) {
    LdsRaceExecutionResult analyzed =
        analyze_lds_race_execution(capture.header.manifest, execution, options.examples);
    result.cross_wave_pairs += analyzed.cross_wave_pairs;
    result.ordering_unresolved_pairs += analyzed.ordering_unresolved_pairs;
    result.collision_groups += analyzed.collision_groups;
    result.coverage_complete &= analyzed.coverage_complete;
    result.executions.push_back(std::move(analyzed));
  }
  return result;
}

bool has_findings(const LdsRaceResult &result) {
  return result.cross_wave_pairs != 0 || result.collision_groups != 0;
}

} // namespace rocjitsu::consan

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
  if (validate_image_facts(capture.header.image, error).failed())
    return Result::failure();
  // Native names are also rendered as image dispositions, without detector facts.
  for (const ImageEntryFacts &entry : capture.header.image.entries) {
    if (!util::is_valid_utf8(entry.binding.name)) {
      if (error != nullptr)
        *error = {ContractErrorCode::InvalidFacts, "entry name is not valid UTF-8 for JSON output"};
      return Result::failure();
    }
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
  result.original = capture.header.image.original;
  result.selection_profile = capture.header.image.selection_profile;
  for (const ImageEntryFacts &entry : capture.header.image.entries)
    result.entries.push_back(
        {.key = entry.key, .name = entry.binding.name, .disposition = entry.disposition});
  if (result.entries.size() == 1)
    result.entry = result.entries.front().name;
  result.execution_conditions = capture.header.execution_conditions;
  result.target = capture.header.image.target;
  result.wave_size = capture.header.image.wave_size;
  result.scope = *selected_scope(capture.header.image);
  result.coverage_complete = !capture.executions.empty();
  std::map<uint32_t, uint32_t> generations;
  for (const CapturedExecution &execution : capture.executions) {
    const Manifest *manifest = selected_manifest(capture.header.image, execution.entry_key);
    if (manifest == nullptr || execution.serial != result.executions.size() + 1 ||
        execution.storage_generation <= generations[execution.storage_slot] ||
        (manifest != nullptr &&
         execution.arguments.size() != manifest->entry.original_kernarg_bytes)) {
      if (error != nullptr)
        *error = {ContractErrorCode::IdentityMismatch,
                  "invalid capture entry or execution identity"};
      return Result::failure();
    }
    generations[execution.storage_slot] = execution.storage_generation;
    LdsRaceExecutionResult analyzed =
        analyze_lds_race_execution(*manifest, execution, options.examples);
    analyzed.entry_key = execution.entry_key;
    analyzed.entry = manifest->entry.name;
    result.ordered_pairs += analyzed.ordered_pairs;
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

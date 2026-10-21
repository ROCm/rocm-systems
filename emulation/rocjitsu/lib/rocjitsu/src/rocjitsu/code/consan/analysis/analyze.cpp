// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/analyze.h"

#include <algorithm>
#include <string_view>

namespace rocjitsu::consan {
namespace {

bool is_valid_utf8(std::string_view text) {
  for (size_t offset = 0; offset < text.size();) {
    const uint8_t lead = static_cast<uint8_t>(text[offset++]);
    if (lead < 0x80)
      continue;
    const uint32_t continuation_bytes = lead >= 0xc2 && lead <= 0xdf   ? 1
                                        : lead >= 0xe0 && lead <= 0xef ? 2
                                        : lead >= 0xf0 && lead <= 0xf4 ? 3
                                                                       : 0;
    if (continuation_bytes == 0 || continuation_bytes > text.size() - offset)
      return false;
    uint32_t code_point = lead & (0x3f >> continuation_bytes);
    for (uint32_t index = 0; index < continuation_bytes; ++index) {
      const uint8_t byte = static_cast<uint8_t>(text[offset++]);
      if ((byte & 0xc0) != 0x80)
        return false;
      code_point = (code_point << 6) | (byte & 0x3f);
    }
    const uint32_t minimum = continuation_bytes == 1   ? 0x80
                             : continuation_bytes == 2 ? 0x800
                                                       : 0x10000;
    if (code_point < minimum || code_point > 0x10ffff ||
        (code_point >= 0xd800 && code_point <= 0xdfff))
      return false;
  }
  return true;
}

} // namespace

FailureOr<LdsRaceResult> analyze_capture(const Capture &capture, const AnalysisOptions &options,
                                         ContractError *error) {
  if (error != nullptr)
    *error = {};
  if (validate_manifest(capture.header.manifest, error).failed())
    return Result::failure();
  // Capture names are opaque bytes; the JSON-facing analyzer requires text.
  if (!is_valid_utf8(capture.header.manifest.entry.name)) {
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
  result.coverage_complete = !capture.executions.empty();
  for (const auto &execution : capture.executions) {
    LdsRaceExecutionResult analyzed =
        analyze_lds_race_execution(capture.header.manifest, execution, options.examples);
    result.cross_wave_pairs += analyzed.cross_wave_pairs;
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

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/capture.h"

namespace rocjitsu::consan {

inline constexpr uint32_t kMaxIntegrityIssueExamples = 128;

enum class LdsRaceOrdering { Unordered, Ordered, Incomplete };

enum class LdsRaceFindingKind { CrossWave, InstructionCollision };

struct LdsRaceParticipant {
  uint32_t wave = 0;
  uint32_t site = 0;
  uint32_t sequence = 0;
  uint64_t mask = 0;
  uint32_t offset = 0;
  uint32_t width = 0;
  uint64_t original_offset = 0;
  AccessKind access = AccessKind::Read;
  uint32_t phase = 0;
  bool operator==(const LdsRaceParticipant &) const = default;
};

struct LdsRaceFinding {
  LdsRaceFindingKind kind = LdsRaceFindingKind::CrossWave;
  LdsRaceParticipant first;
  // Collision participants are the lanes in first.mask; second is unused.
  LdsRaceParticipant second;
  uint64_t lane_pairs = 0;
  LdsRaceOrdering ordering = LdsRaceOrdering::Unordered;
  bool operator==(const LdsRaceFinding &) const = default;
};

struct LdsRaceIntegrityIssue {
  // UINT32_MAX denotes an execution-wide issue or no particular site.
  uint32_t wave = UINT32_MAX;
  uint32_t site = UINT32_MAX;
  std::string reason;
  bool operator==(const LdsRaceIntegrityIssue &) const = default;
};

struct LdsRaceExecutionResult {
  uint64_t serial = 0;
  uint64_t valid_observations = 0;
  uint64_t rejected_records = 0;
  // Loss counts exhausted instruction occurrences, not missing lane records.
  uint64_t lost_occurrences = 0;
  // Qualified candidate pairs excluded by a complete workgroup certificate.
  uint64_t ordered_pairs = 0;
  uint64_t ordering_unresolved_pairs = 0;
  uint32_t required_barrier_waves = 0;
  uint32_t completed_barrier_waves = 0;
  // Confirmed unordered address-group pairs, then lane pairs.
  // Read/read pairs contribute to neither count.
  uint64_t cross_wave_pairs = 0;
  uint64_t cross_wave_lane_pairs = 0;
  // Each write observation with multiple lanes sharing its address contributes
  // one group; lane pairs count the unordered lane combinations within it.
  uint64_t collision_groups = 0;
  uint64_t collision_lane_pairs = 0;
  bool coverage_complete = true;
  // Unordered applies only to independently valid observations in the admitted
  // scope. Gaps never establish ordering for missing data.
  bool ordering_uncertain = false;
  uint64_t integrity_events = 0;
  // Retain at most kMaxIntegrityIssueExamples; integrity_events remains complete.
  std::vector<LdsRaceIntegrityIssue> issues;
  std::vector<LdsRaceFinding> cross_wave_examples;
  std::vector<LdsRaceFinding> collision_examples;
  std::vector<LdsRaceFinding> ordered_examples;
  std::vector<LdsRaceFinding> unresolved_examples;
  bool operator==(const LdsRaceExecutionResult &) const = default;
};

struct LdsRaceResult {
  std::array<uint8_t, 16> run_identity{};
  ImageIdentity original;
  std::string entry;
  uint32_t execution_conditions = 0;
  uint32_t target = 0;
  uint32_t wave_size = 0;
  ScopeKind scope = ScopeKind::NativeLdsNoSynchronization;
  // Totals over executions, using the same record-pair/group units as above.
  uint64_t cross_wave_pairs = 0;
  uint64_t ordering_unresolved_pairs = 0;
  uint64_t collision_groups = 0;
  bool coverage_complete = false;
  std::vector<LdsRaceExecutionResult> executions;
  bool operator==(const LdsRaceResult &) const = default;
};

/// @brief Qualify native LDS rows and run both checks on shared access evidence.
/// @details The coordinator validates static facts, conditions and example budget.
/// Malformed local evidence produces gaps rather than discarding other findings.
/// See docs/consan.md#analysis-and-admission.
[[nodiscard]] LdsRaceExecutionResult analyze_lds_race_execution(const Manifest &facts,
                                                                const CapturedExecution &execution,
                                                                uint32_t examples);

/// @brief Render the concrete LDS result as deterministic JSON.
[[nodiscard]] std::string render_lds_race_analysis(const LdsRaceResult &result);

} // namespace rocjitsu::consan

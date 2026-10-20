// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/analysis/lds_race.h"

namespace rocjitsu::consan {

inline constexpr uint32_t kMaxFindingExamples = 1024;

struct AnalysisOptions {
  // Retained finding details per check, per execution; zero hides these examples.
  // Pair counts can grow as a product of observation counts. This software budget
  // bounds finding objects/output, not observations, probes, coverage or counts.
  uint32_t examples = 8;
};

/// @brief Coordinate LDS diagnosis of an owning capture.
/// @details File callers first use decode_capture for framing, End and envelope
/// validation. Typed callers must supply equivalent execution envelopes; this
/// entry still validates manifest, execution conditions, identity and budgets.
/// Invalid static facts are input errors. Local raw GPU corruption and
/// unsupported effective ranges become explicit gaps, preserving independent
/// findings. Naturally aligned native b32 ranges have mode-independent effective
/// addresses; unaligned operands require unavailable SH_MEM_CONFIG evidence.
/// Counts refer to valid observed address-group pairs/groups; lane-pair totals
/// are reported separately. Zero findings is never a race-free verdict.
[[nodiscard]] FailureOr<LdsRaceResult> analyze_capture(const Capture &capture,
                                                       const AnalysisOptions &options = {},
                                                       ContractError *error = nullptr);

/// @brief Query observed findings without exposing check-specific counters to callers.
[[nodiscard]] bool has_findings(const LdsRaceResult &result);

/// @brief Render the selected analysis result through its concrete renderer.
[[nodiscard]] std::string render_analysis(const LdsRaceResult &result);

} // namespace rocjitsu::consan

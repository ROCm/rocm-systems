// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file case_io.h
/// @brief End-to-end analyze/test pipeline and the case directory format.
///
/// Case directory layout:
///   case.json              analysis: artifacts, changes, unresolved items, regions+contracts
///   results.json           written by `test`: aggregate + per-region results
///   counterexamples/*.json self-contained replay bundles
#pragma once

#include "differential.h"
#include "region_planner.h"

#include <optional>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

struct CaseData {
  Json identity;
  Json command;
  Json artifacts; ///< old/new: label, kind, sha256, kernel, ranges.
  Json changes;   ///< Hunks with classification.
  std::vector<UnresolvedItem> unresolved;
  std::vector<ArtifactNote> notes;
  std::vector<RegionPair> regions;
  Json analysis; ///< Kernel sizes, timings, plan options.

  [[nodiscard]] Json to_json() const;
  static std::optional<CaseData> from_json(const Json &j, std::string &error);
};

/// @brief Decode, align, classify and plan. Library form of `rj_isa_diff analyze`.
[[nodiscard]] std::optional<CaseData> analyze_artifacts(const Artifact &old_artifact,
                                                        const Artifact &new_artifact,
                                                        const PlanOptions &options,
                                                        std::string &error);

struct Aggregate {
  TestStatus test_status = TestStatus::Inconclusive;
  std::string contract_status; ///< Worst region contract, or NOT_APPLICABLE.
  WitnessScope witness_scope = WitnessScope::None;
  CoverageScope coverage_scope = CoverageScope::NotEstablished;
};

[[nodiscard]] Aggregate aggregate_results(const CaseData &data,
                                          const std::vector<RegionResult> &results);

/// @brief Library form of `rj_isa_diff test`: returns the results.json document.
[[nodiscard]] Json test_case(const CaseData &data, const TestOptions &options,
                             const std::string &region_filter,
                             std::vector<RegionResult> *results_out = nullptr);

/// @brief Read case.json from a case directory.
[[nodiscard]] std::optional<CaseData> read_case_dir(const std::string &dir, std::string &error);
/// @brief Write case.json into a (created) case directory.
[[nodiscard]] bool write_case_dir(const std::string &dir, const CaseData &data, std::string &error);

} // namespace rocjitsu::isa_diff

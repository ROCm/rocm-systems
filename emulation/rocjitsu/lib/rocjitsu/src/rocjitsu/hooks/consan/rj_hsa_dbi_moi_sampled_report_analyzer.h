// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_sampled_report_decoder.h"

#include <cstdint>
#include <optional>
#include <span>
#include <utility>

namespace rocjitsu::consan_hook {

/// The complete host-side semantic analysis product for a Sampled report.
struct AutoMoiSampledConflictAnalysis {
  uint32_t conflict_count = 0;
  std::optional<std::pair<AutoMoiSampledEvidence, AutoMoiSampledEvidence>> first_conflict;
};

[[nodiscard]] AutoMoiSampledConflictAnalysis
analyze_auto_moi_sampled_conflicts(std::span<const AutoMoiSampledEvidence> evidence,
                                   bool synchronization_evidence_complete);

void accumulate_auto_moi_sampled_analysis(AutoMoiReportSummary &summary,
                                          const AutoMoiSampledConflictAnalysis &analysis);

} // namespace rocjitsu::consan_hook

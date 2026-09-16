// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/modes/sampled/rj_hsa_dbi_moi_sampled_report_decoder.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::consan_hook {

/// The complete host-side semantic analysis product for a Sampled report.
struct AutoMoiSampledConflictAnalysis {
  uint32_t conflict_count = 0;
  // Examples are deduplicated and bounded independently from the pair count.
  std::vector<std::pair<AutoMoiSampledEvidence, AutoMoiSampledEvidence>> conflicts;
};

[[nodiscard]] AutoMoiSampledConflictAnalysis
analyze_auto_moi_sampled_conflicts(std::span<const AutoMoiSampledEvidence> evidence,
                                   bool synchronization_evidence_complete,
                                   uint32_t example_limit = 8);

void accumulate_auto_moi_sampled_analysis(AutoMoiReportSummary &summary,
                                          const AutoMoiSampledConflictAnalysis &analysis);

} // namespace rocjitsu::consan_hook

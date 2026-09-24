// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_evidence_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_publication.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::consan::hook {

/// The complete host-side semantic analysis product for a ConSan report.
struct ConflictAnalysis {
  uint32_t conflict_count = 0;
  uint32_t ordered_publication_pairs = 0;
  uint32_t incomplete_publication_pairs = 0;
  uint32_t suppressed_uniform_write_conflict_count = 0;
  // Examples are deduplicated and bounded independently from the pair count.
  std::vector<std::pair<Evidence, Evidence>> examples;
};

[[nodiscard]] ConflictAnalysis analyze_conflicts(std::span<const Evidence> evidence,
                                                 bool synchronization_evidence_complete,
                                                 uint32_t example_limit = 8,
                                                 bool allow_uniform_lds_stores = false,
                                                 const DecodedPublications *publications = nullptr);

void accumulate_analysis(ReportSummary &summary, const ConflictAnalysis &analysis);

} // namespace rocjitsu::consan::hook

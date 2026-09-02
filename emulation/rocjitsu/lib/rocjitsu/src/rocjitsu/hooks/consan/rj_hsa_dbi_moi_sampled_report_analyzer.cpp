// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_sampled_report_analyzer.h"

#include <limits>
#include <ranges>

namespace rocjitsu::consan_hook {

AutoMoiSampledConflictAnalysis
analyze_auto_moi_sampled_conflicts(std::span<const AutoMoiSampledEvidence> evidence,
                                   bool synchronization_evidence_complete) {
  AutoMoiSampledConflictAnalysis result;
  for (size_t index = 0; index < evidence.size(); ++index) {
    const AutoMoiSampledEvidence &current = evidence[index];
    for (size_t prior_index = 0; prior_index < index; ++prior_index) {
      const AutoMoiSampledEvidence &prior = evidence[prior_index];
      if (current.dispatch_id != prior.dispatch_id || current.workgroup_x != prior.workgroup_x ||
          current.workgroup_y != prior.workgroup_y || current.workgroup_z != prior.workgroup_z ||
          current.cluster_workgroup_id != prior.cluster_workgroup_id ||
          current.epoch != prior.epoch)
        continue;
      if (current.static_mapping != nullptr && prior.static_mapping != nullptr &&
          current.static_mapping->owner_provenance_complete &&
          prior.static_mapping->owner_provenance_complete &&
          std::ranges::none_of(
              current.static_mapping->owner_descriptor_file_offsets, [&](uint64_t owner) {
                return std::ranges::find(prior.static_mapping->owner_descriptor_file_offsets,
                                         owner) !=
                       prior.static_mapping->owner_descriptor_file_offsets.end();
              }))
        continue;
      if (!consan_moi_sampled_watchpoints_conflict(current.entry, prior.entry))
        continue;
      if (synchronization_evidence_complete &&
          consan_moi_sampled_atomic_pair_orders_same_workgroup(prior.sync, current.sync))
        continue;
      if (result.conflict_count != std::numeric_limits<uint32_t>::max())
        ++result.conflict_count;
      if (!result.first_conflict)
        result.first_conflict = std::make_pair(prior, current);
    }
  }
  return result;
}

void accumulate_auto_moi_sampled_analysis(AutoMoiReportSummary &summary,
                                          const AutoMoiSampledConflictAnalysis &analysis) {
  summary.sampled_conflict_count = analysis.conflict_count;
}

} // namespace rocjitsu::consan_hook

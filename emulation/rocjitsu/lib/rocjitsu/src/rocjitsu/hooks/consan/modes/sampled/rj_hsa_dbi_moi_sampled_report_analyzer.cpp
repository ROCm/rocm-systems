// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_sampled_report_analyzer.h"

#include <bit>
#include <limits>
#include <ranges>

namespace rocjitsu::consan_hook {
namespace {

bool same_sampled_access(const AutoMoiSampledEvidence &a, const AutoMoiSampledEvidence &b) {
  const bool same_site =
      a.static_mapping && b.static_mapping
          ? a.static_mapping->instruction_offset == b.static_mapping->instruction_offset
          : !a.static_mapping && !b.static_mapping && a.index == b.index;
  return same_site && a.generation == b.generation && a.dispatch_id == b.dispatch_id &&
         a.workgroup_x == b.workgroup_x && a.workgroup_y == b.workgroup_y &&
         a.workgroup_z == b.workgroup_z && a.cluster_workgroup_id == b.cluster_workgroup_id &&
         a.epoch == b.epoch && a.entry.owner_id == b.entry.owner_id &&
         a.entry.kind == b.entry.kind && a.entry.start_byte == b.entry.start_byte &&
         a.entry.byte_count == b.entry.byte_count && a.exact_lane_mask == b.exact_lane_mask;
}

} // namespace

AutoMoiSampledConflictAnalysis
analyze_auto_moi_sampled_conflicts(std::span<const AutoMoiSampledEvidence> evidence,
                                   bool synchronization_evidence_complete, uint32_t example_limit,
                                   bool allow_uniform_lds_stores) {
  AutoMoiSampledConflictAnalysis result;
  // Keep the linear exact-group pass outside the quadratic cross-wave scan.
  for (const auto &current : evidence) {
    if (current.entry.valid && !current.entry.consumed &&
        consan_moi_shadow_kind_conflicts(current.entry.kind, current.entry.kind) &&
        std::popcount(current.exact_lane_mask) > 1 &&
        !(allow_uniform_lds_stores && current.static_mapping &&
          current.static_mapping->uniform_lds_store)) {
      if (result.conflict_count != std::numeric_limits<uint32_t>::max())
        ++result.conflict_count;
      if (result.conflicts.size() < example_limit) {
        auto first = current;
        auto second = current;
        first.exact_lane_mask = current.exact_lane_mask & -current.exact_lane_mask;
        second.exact_lane_mask ^= first.exact_lane_mask;
        if (std::ranges::none_of(result.conflicts, [&](const auto &example) {
              return same_sampled_access(example.first, first) &&
                     same_sampled_access(example.second, second);
            }))
          result.conflicts.emplace_back(first, second);
      }
    }
  }
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
          std::ranges::none_of(current.static_mapping->owner_kernel_ids, [&](uint32_t owner) {
            return std::ranges::find(prior.static_mapping->owner_kernel_ids, owner) !=
                   prior.static_mapping->owner_kernel_ids.end();
          }))
        continue;
      if (!consan_moi_sampled_watchpoints_conflict(current.entry, prior.entry))
        continue;
      if (synchronization_evidence_complete &&
          consan_moi_sampled_atomic_pair_orders_same_workgroup(prior.sync, current.sync))
        continue;
      if (result.conflict_count != std::numeric_limits<uint32_t>::max())
        ++result.conflict_count;
      // Once full, counting continues without allocating, formatting, or
      // searching the retained examples. Traversal order defines stable output.
      if (result.conflicts.size() < example_limit &&
          std::ranges::none_of(result.conflicts, [&](const auto &example) {
            return (same_sampled_access(example.first, prior) &&
                    same_sampled_access(example.second, current)) ||
                   (same_sampled_access(example.first, current) &&
                    same_sampled_access(example.second, prior));
          }))
        result.conflicts.emplace_back(prior, current);
    }
  }
  return result;
}

void accumulate_auto_moi_sampled_analysis(AutoMoiReportSummary &summary,
                                          const AutoMoiSampledConflictAnalysis &analysis) {
  summary.sampled_conflict_count = analysis.conflict_count;
}

} // namespace rocjitsu::consan_hook

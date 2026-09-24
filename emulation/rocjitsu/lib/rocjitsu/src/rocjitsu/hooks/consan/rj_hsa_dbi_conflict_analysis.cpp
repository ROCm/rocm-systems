// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_conflict_analysis.h"

#include <bit>
#include <limits>
#include <ranges>

namespace rocjitsu::consan::hook {
namespace {

bool same_access(const Evidence &a, const Evidence &b) {
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

ConflictAnalysis analyze_conflicts(std::span<const Evidence> evidence,
                                   bool synchronization_evidence_complete, uint32_t example_limit,
                                   bool allow_uniform_lds_stores,
                                   const DecodedPublications *publications) {
  ConflictAnalysis result;
  // Keep the linear exact-group pass outside the quadratic cross-wave scan.
  for (const auto &current : evidence) {
    if (current.entry.valid && !current.entry.consumed &&
        shadow_kind_conflicts(current.entry.kind, current.entry.kind) &&
        std::popcount(current.exact_lane_mask) > 1) {
      if (allow_uniform_lds_stores && current.static_mapping &&
          current.static_mapping->uniform_lds_store) {
        if (result.suppressed_uniform_write_conflict_count != std::numeric_limits<uint32_t>::max())
          ++result.suppressed_uniform_write_conflict_count;
        continue;
      }
      if (result.conflict_count != std::numeric_limits<uint32_t>::max())
        ++result.conflict_count;
      if (result.examples.size() < example_limit) {
        auto first = current;
        auto second = current;
        first.exact_lane_mask = current.exact_lane_mask & -current.exact_lane_mask;
        second.exact_lane_mask ^= first.exact_lane_mask;
        if (std::ranges::none_of(result.examples, [&](const auto &example) {
              return same_access(example.first, first) && same_access(example.second, second);
            }))
          result.examples.emplace_back(first, second);
      }
    }
  }
  for (size_t index = 0; index < evidence.size(); ++index) {
    const Evidence &current = evidence[index];
    for (size_t prior_index = 0; prior_index < index; ++prior_index) {
      const Evidence &prior = evidence[prior_index];
      if (current.generation != prior.generation || current.dispatch_id != prior.dispatch_id ||
          current.workgroup_x != prior.workgroup_x || current.workgroup_y != prior.workgroup_y ||
          current.workgroup_z != prior.workgroup_z ||
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
      if (!watchpoints_conflict(current.entry, prior.entry))
        continue;
      if (publications != nullptr) {
        const auto point = [](const Evidence &entry) {
          return PublicationPoint{.domain = {.generation = entry.generation,
                                             .dispatch = entry.dispatch_id,
                                             .workgroup_x = entry.workgroup_x,
                                             .workgroup_y = entry.workgroup_y,
                                             .workgroup_z = entry.workgroup_z,
                                             .cluster_workgroup = entry.cluster_workgroup_id},
                                  .owner = entry.entry.owner_id,
                                  .sequence = entry.publication_sequence};
        };
        const bool complete = synchronization_evidence_complete &&
                              publications->status == PublicationDecodeStatus::Complete;
        const auto forward =
            publication_orders(point(prior), point(current), publications->events, complete);
        const auto backward =
            publication_orders(point(current), point(prior), publications->events, complete);
        if (forward == PublicationOrdering::Ordered || backward == PublicationOrdering::Ordered) {
          ++result.ordered_publication_pairs;
          continue;
        }
        if (forward == PublicationOrdering::Incomplete ||
            backward == PublicationOrdering::Incomplete)
          ++result.incomplete_publication_pairs;
      } else if (synchronization_evidence_complete &&
                 atomic_pair_orders_same_workgroup(prior.sync, current.sync)) {
        continue;
      }
      if (result.conflict_count != std::numeric_limits<uint32_t>::max())
        ++result.conflict_count;
      // Once full, counting continues without allocating, formatting, or
      // searching the retained examples. Traversal order defines stable output.
      if (result.examples.size() < example_limit &&
          std::ranges::none_of(result.examples, [&](const auto &example) {
            return (same_access(example.first, prior) && same_access(example.second, current)) ||
                   (same_access(example.first, current) && same_access(example.second, prior));
          }))
        result.examples.emplace_back(prior, current);
    }
  }
  return result;
}

void accumulate_analysis(ReportSummary &summary, const ConflictAnalysis &analysis) {
  summary.conflict_count = analysis.conflict_count;
  summary.unsupported_sync_count += analysis.incomplete_publication_pairs;
  summary.suppressed_uniform_write_conflict_count =
      analysis.suppressed_uniform_write_conflict_count;
}

} // namespace rocjitsu::consan::hook

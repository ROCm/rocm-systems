// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/lds_race.h"

#include <locale>
#include <sstream>

namespace rocjitsu::consan {
namespace {

std::string hex_bytes(std::span<const uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (const uint8_t byte : bytes) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

void write_json_string(std::ostream &out, std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  out << '"';
  for (const unsigned char ch : value) {
    if (ch == '"' || ch == '\\') {
      out << '\\' << ch;
    } else if (ch < 32) {
      out << "\\u00" << digits[ch >> 4] << digits[ch & 15];
    } else {
      out << ch;
    }
  }
  out << '"';
}

std::string_view access_name(AccessKind access) {
  switch (access) {
  case AccessKind::Read:
    return "read";
  case AccessKind::Write:
    return "write";
  case AccessKind::Atomic:
    return "atomic";
  }
  return "unknown";
}

void write_flat_confidence(std::ostream &out, FlatAssumption assumption) {
  if (assumption == FlatAssumption::None)
    out << ",\"confidence\":\"exact\"";
  else {
    out << ",\"confidence\":\"conservative\",\"assumption\":";
    write_json_string(out, flat_assumption_text(assumption));
  }
}

void write_participant(std::ostream &out, const LdsRaceParticipant &value) {
  out << "{\"workgroup_x\":" << value.workgroup << ",\"wave\":" << value.wave
      << ",\"site\":" << value.site << ",\"range_id\":" << value.range_id
      << ",\"phase\":" << value.phase << ",\"sequence\":" << value.sequence
      << ",\"mask\":" << value.mask << ",\"offset\":" << value.offset
      << ",\"width\":" << value.width << ",\"original_offset\":" << value.original_offset
      << ",\"access\":\"" << access_name(value.access) << '"';
  if (value.origin == AccessOrigin::GroupFlat) {
    out << ",\"origin\":\"group_flat\"";
    write_flat_confidence(out, value.assumption);
  }
  if (value.transpose_exec)
    out << ",\"mask_role\":\"transpose_source_lanes\",\"anchor_exec\":" << *value.transpose_exec;
  out << '}';
}

void write_examples(std::ostream &out, const std::vector<LdsRaceFinding> &findings) {
  out << '[';
  bool first = true;
  for (const LdsRaceFinding &finding : findings) {
    if (!first) {
      out << ',';
    }
    first = false;
    out << "{\"kind\":\""
        << (finding.kind == LdsRaceFindingKind::CrossWave ? "cross_wave" : "instruction_collision")
        << "\",\"lane_pairs\":" << finding.lane_pairs;
    if (finding.kind == LdsRaceFindingKind::CrossWave) {
      out << ",\"ordering\":\""
          << (finding.ordering == LdsRaceOrdering::Ordered      ? "ordered"
              : finding.ordering == LdsRaceOrdering::Unresolved ? "ordering_unresolved"
                                                                : "unordered")
          << "\"";
    } else {
      out << ",\"reason\":\"ordinary_store_exact_mask\"";
    }
    if (finding.ordering == LdsRaceOrdering::Unresolved) {
      out << ",\"reason\":\"incomplete_synchronization_evidence\"";
    }
    out << ",\"first\":";
    write_participant(out, finding.first);
    if (finding.kind == LdsRaceFindingKind::CrossWave || finding.second.mask != 0) {
      out << ",\"second\":";
      write_participant(out, finding.second);
    }
    out << '}';
  }
  out << ']';
}

void write_flat_sites(std::ostream &out, const LdsRaceImageEntry &entry) {
  if (!entry.flat_sites.empty()) {
    out << ",\"flat_sites\":[";
    for (const SiteFacts &site : entry.flat_sites) {
      if (&site != &entry.flat_sites.front())
        out << ',';
      out << "{\"site\":" << site.id << ",\"original_offset\":" << site.original_offset;
      write_flat_confidence(out, site.assumption);
      out << '}';
    }
    out << ']';
  }
  if (!entry.flat_exclusions.empty()) {
    out << ",\"flat_exclusions\":[";
    uint32_t id = entry.observed_sites;
    for (const FlatExclusion &site : entry.flat_exclusions) {
      if (id != entry.observed_sites)
        out << ',';
      out << "{\"site\":" << id++ << ",\"original_offset\":" << site.original_offset
          << ",\"original_size\":" << site.original_size << ",\"reason\":\""
          << (site.reason == FlatExclusionReason::NonGroup ? "non_group"
              : site.reason == FlatExclusionReason::Strict ? "strict_excludes_likely"
                                                           : "unresolved")
          << '"';
      if (site.reason == FlatExclusionReason::Strict)
        write_flat_confidence(out, site.assumption);
      out << '}';
    }
    out << ']';
  }
}

// The legacy collision counters count findings remaining under the selected policy.
void write_collision_counts(std::ostream &out, uint64_t groups, uint64_t pairs,
                            uint64_t suppressed_groups, uint64_t suppressed_pairs) {
  out << ",\"raw_collision_groups\":" << groups + suppressed_groups
      << ",\"suppressed_collision_groups\":" << suppressed_groups
      << ",\"collision_groups\":" << groups
      << ",\"raw_collision_lane_pairs\":" << pairs + suppressed_pairs
      << ",\"suppressed_collision_lane_pairs\":" << suppressed_pairs
      << ",\"collision_lane_pairs\":" << pairs;
}

} // namespace

std::string render_lds_race_analysis(const LdsRaceResult &result) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::boolalpha;
  out << "{\"verdict\":\""
      << (result.cross_wave_pairs || result.collision_groups ? "findings" : "no_observed_findings")
      << "\",\"scope\":\"" << (result.has_group_flat ? "native_and_group_flat_lds_" : "native_lds_")
      << (result.has_dual_ranges ? "multi_range_" : "single_range_")
      << (result.has_transpose ? "transpose_" : "")
      << (result.scope != ScopeKind::NativeLdsNoSynchronization ? "epoch_workgroup_barriers"
                                                                : "no_sync_bounded_1d")
      << "\"" << ",\"run_identity\":\"" << hex_bytes(result.run_identity)
      << "\",\"original_sha256\":\"" << hex_bytes(result.original.sha256)
      << "\",\"original_bytes\":" << result.original.bytes << ",\"selection_profile\":\""
      << hex_bytes(result.selection_profile) << "\",\"entries\":[";
  bool first_entry = true;
  for (const LdsRaceImageEntry &entry : result.entries) {
    if (!first_entry) {
      out << ',';
    }
    first_entry = false;
    out << "{\"entry_key\":" << entry.key << ",\"entry\":";
    write_json_string(out, entry.name);
    out << ",\"disposition\":\""
        << (entry.disposition == EntryDisposition::Selected ? "selected" : "native_uncovered")
        << '"';
    if (entry.disposition == EntryDisposition::Selected) {
      out << ",\"uniform_store_proofs\":[";
      bool first_proof = true;
      for (const SiteFacts &site : entry.uniform_store_sites) {
        if (!first_proof)
          out << ',';
        first_proof = false;
        out << "{\"site\":" << site.id << ",\"original_offset\":" << site.original_offset
            << ",\"source\":\"block_local_broadcast\"}";
      }
      out << ']';
    }
    if (result.flat_policy && result.entries.size() > 1) {
      write_flat_sites(out, entry);
    }
    out << '}';
  }
  out << "],\"entry\":";
  write_json_string(out, result.entry);
  if (result.flat_policy) {
    out << ",\"flat_policy\":\""
        << (*result.flat_policy == FlatPolicy::Likely ? "likely" : "strict") << '\"';
    if (result.entries.size() == 1) {
      write_flat_sites(out, result.entries.front());
    }
  }
  out << ",\"target\":" << result.target << ",\"wave_size\":" << result.wave_size
      << ",\"execution_conditions\":" << result.execution_conditions
      << ",\"coverage_complete\":" << result.coverage_complete
      << ",\"ordering_unresolved_pairs\":" << result.ordering_unresolved_pairs
      << ",\"ordered_pairs\":" << result.ordered_pairs
      << ",\"cross_wave_pairs\":" << result.cross_wave_pairs
      << ",\"suppress_uniform_store_collisions\":" << result.suppress_uniform_store_collisions;
  write_collision_counts(out, result.collision_groups, result.collision_lane_pairs,
                         result.suppressed_collision_groups,
                         result.suppressed_collision_lane_pairs);
  out << ",\"executions\":[";
  bool first = true;
  for (const LdsRaceExecutionResult &execution : result.executions) {
    if (!first) {
      out << ',';
    }
    first = false;
    // Once any counter exhausts, do not present the difference of saturating
    // total-loss and address-loss counters as an exact dropped count; emit null.
    out << "{\"serial\":" << execution.serial << ",\"entry_key\":" << execution.entry_key
        << ",\"entry\":";
    write_json_string(out, execution.entry);
    out << ",\"workgroup_count\":" << execution.workgroup_count
        << ",\"waves_per_workgroup\":" << execution.waves_per_workgroup
        << ",\"valid_observations\":" << execution.valid_observations
        << ",\"rejected_records\":" << execution.rejected_records
        << ",\"lost_occurrences\":" << execution.lost_occurrences << ",\"dropped_representatives\":"
        << (execution.counter_exhausted_waves ? "null"
                                              : std::to_string(execution.dropped_representatives))
        << ",\"synchronization_loss\":" << execution.synchronization_loss
        << ",\"address_group_exhaustion\":" << execution.address_group_exhaustion
        << ",\"counter_exhausted_waves\":" << execution.counter_exhausted_waves
        << ",\"coverage_complete\":" << execution.coverage_complete
        << ",\"ordering_uncertain\":" << execution.ordering_uncertain
        << ",\"integrity_events\":" << execution.integrity_events
        << ",\"ordered_pairs\":" << execution.ordered_pairs
        << ",\"ordering_unresolved_pairs\":" << execution.ordering_unresolved_pairs
        << ",\"required_barrier_waves\":" << execution.required_barrier_waves
        << ",\"completed_barrier_waves\":" << execution.completed_barrier_waves
        << ",\"cross_wave_pairs\":" << execution.cross_wave_pairs
        << ",\"cross_wave_lane_pairs\":" << execution.cross_wave_lane_pairs;
    write_collision_counts(out, execution.collision_groups, execution.collision_lane_pairs,
                           execution.suppressed_collision_groups,
                           execution.suppressed_collision_lane_pairs);
    out << ",\"issues\":[";
    bool first_issue = true;
    for (const LdsRaceIntegrityIssue &issue : execution.issues) {
      if (!first_issue) {
        out << ',';
      }
      first_issue = false;
      out << "{\"wave\":";
      if (issue.wave == UINT32_MAX) {
        out << "null";
      } else {
        out << issue.wave;
      }
      out << ",\"site\":";
      if (issue.site == UINT32_MAX) {
        out << "null";
      } else {
        out << issue.site;
      }
      out << ",\"workgroup_x\":";
      if (issue.workgroup == UINT32_MAX) {
        out << "null";
      } else {
        out << issue.workgroup;
      }
      out << ",\"reason\":";
      write_json_string(out, issue.reason);
      out << '}';
    }
    out << "],\"cross_wave_examples\":";
    write_examples(out, execution.cross_wave_examples);
    out << ",\"unresolved_examples\":";
    write_examples(out, execution.unresolved_examples);
    out << ",\"ordered_examples\":";
    write_examples(out, execution.ordered_examples);
    out << ",\"collision_examples\":";
    write_examples(out, execution.collision_examples);
    out << '}';
  }
  out << "]}\n";
  return out.str();
}

} // namespace rocjitsu::consan

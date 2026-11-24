// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/lds_race.h"

#include "rocjitsu/code/consan/analysis/render_helpers.h"

#include <locale>
#include <sstream>

namespace rocjitsu::consan {
namespace {

using render_detail::hex_bytes;
using render_detail::write_json_string;

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

void write_participant(std::ostream &out, const LdsRaceParticipant &value) {
  out << "{\"workgroup_x\":" << value.workgroup << ",\"wave\":" << value.wave
      << ",\"site\":" << value.site << ",\"range_id\":" << value.range_id
      << ",\"phase\":" << value.phase << ",\"sequence\":" << value.sequence
      << ",\"mask\":" << value.mask << ",\"offset\":" << value.offset
      << ",\"width\":" << value.width << ",\"original_offset\":" << value.original_offset
      << ",\"access\":\"" << access_name(value.access) << '"';
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

} // namespace

std::string render_lds_race_analysis(const LdsRaceResult &result) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::boolalpha;
  out << "{\"verdict\":\""
      << (result.cross_wave_pairs || result.collision_groups ? "findings" : "no_observed_findings")
      << "\",\"scope\":\""
      << (result.has_dual_ranges ? "native_lds_multi_range_" : "native_lds_single_range_")
      << (result.has_transpose ? "transpose_" : "")
      << (result.scope != ScopeKind::NativeLdsNoSynchronization ? "epoch_workgroup_barriers"
                                                                : "no_sync_bounded_1d")
      << "\",\"lds_analysis\":\"" << (result.executions.empty() ? "not_performed" : "performed")
      << "\",\"run_identity\":\"" << hex_bytes(result.run_identity) << "\",\"original_sha256\":\""
      << hex_bytes(result.original.sha256) << "\",\"original_bytes\":" << result.original.bytes
      << ",\"selection_profile\":\"" << hex_bytes(result.selection_profile) << "\",\"entries\":[";
  bool first_entry = true;
  for (const LdsRaceImageEntry &entry : result.entries) {
    if (!first_entry) {
      out << ',';
    }
    first_entry = false;
    out << "{\"entry_key\":" << entry.key << ",\"entry\":";
    write_json_string(out, entry.name);
    out << ",\"disposition\":\""
        << (entry.disposition == EntryDisposition::Selected
                ? "selected"
                : (entry.no_probes_needed ? "no_probes_needed" : "native_uncovered"))
        << "\"";
    if (entry.no_probes_needed) {
      const NoProbesNeededProof &proof = *entry.no_probes_needed;
      out << ",\"no_probes_needed\":{\"version\":" << proof.version
          << ",\"scope\":\"lds\",\"instruction_count\":" << proof.instruction_count
          << ",\"code_bytes\":" << proof.code_bytes << ",\"code_sha256\":\""
          << hex_bytes(proof.code_sha256) << "\"}";
    }
    out << '}';
  }
  out << "],\"entry\":";
  write_json_string(out, result.entry);
  out << ",\"target\":" << result.target << ",\"wave_size\":" << result.wave_size
      << ",\"execution_conditions\":" << result.execution_conditions
      << ",\"coverage_complete\":" << result.coverage_complete
      << ",\"provided_entry_coverage_complete\":" << result.provided_entry_coverage_complete
      << ",\"ordering_unresolved_pairs\":" << result.ordering_unresolved_pairs
      << ",\"ordered_pairs\":" << result.ordered_pairs
      << ",\"cross_wave_pairs\":" << result.cross_wave_pairs
      << ",\"collision_groups\":" << result.collision_groups << ",\"executions\":[";
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
        << ",\"cross_wave_lane_pairs\":" << execution.cross_wave_lane_pairs
        << ",\"collision_groups\":" << execution.collision_groups
        << ",\"collision_lane_pairs\":" << execution.collision_lane_pairs << ",\"issues\":[";
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

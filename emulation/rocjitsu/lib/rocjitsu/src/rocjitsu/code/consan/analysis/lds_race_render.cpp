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
    if (ch == '"' || ch == '\\')
      out << '\\' << ch;
    else if (ch < 32)
      out << "\\u00" << digits[ch >> 4] << digits[ch & 15];
    else
      out << ch;
  }
  out << '"';
}

void participant(std::ostream &out, const LdsRaceParticipant &value) {
  out << "{\"wave\":" << value.wave << ",\"site\":" << value.site
      << ",\"sequence\":" << value.sequence << ",\"mask\":" << value.mask
      << ",\"offset\":" << value.offset << ",\"width\":" << value.width
      << ",\"original_offset\":" << value.original_offset << ",\"access\":\""
      << (value.access == AccessKind::Write ? "write" : "read") << "\"}";
}

void examples(std::ostream &out, const std::vector<LdsRaceFinding> &findings) {
  out << '[';
  bool first = true;
  for (const auto &finding : findings) {
    if (!first)
      out << ',';
    first = false;
    out << "{\"kind\":\""
        << (finding.kind == LdsRaceFindingKind::CrossWave ? "cross_wave" : "instruction_collision")
        << "\",\"lane_pairs\":" << finding.lane_pairs;
    if (finding.kind == LdsRaceFindingKind::CrossWave)
      out << ",\"ordering\":\"unordered\"";
    else
      out << ",\"reason\":\"ordinary_store_exact_mask\"";
    out << ",\"first\":";
    participant(out, finding.first);
    if (finding.kind == LdsRaceFindingKind::CrossWave) {
      out << ",\"second\":";
      participant(out, finding.second);
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
      << "\",\"scope\":\"native_lds_b32_no_sync_one_workgroup\"" << ",\"run_identity\":\""
      << hex_bytes(result.run_identity) << "\",\"original_sha256\":\""
      << hex_bytes(result.original.sha256) << "\",\"original_bytes\":" << result.original.bytes
      << ",\"entry\":";
  write_json_string(out, result.entry);
  out << ",\"target\":" << result.target << ",\"wave_size\":" << result.wave_size
      << ",\"execution_conditions\":" << result.execution_conditions
      << ",\"coverage_complete\":" << result.coverage_complete
      << ",\"cross_wave_pairs\":" << result.cross_wave_pairs
      << ",\"collision_groups\":" << result.collision_groups << ",\"executions\":[";
  bool first = true;
  for (const auto &execution : result.executions) {
    if (!first)
      out << ',';
    first = false;
    out << "{\"serial\":" << execution.serial
        << ",\"valid_observations\":" << execution.valid_observations
        << ",\"rejected_records\":" << execution.rejected_records
        << ",\"lost_occurrences\":" << execution.lost_occurrences
        << ",\"coverage_complete\":" << execution.coverage_complete
        << ",\"ordering_uncertain\":" << execution.ordering_uncertain
        << ",\"integrity_events\":" << execution.integrity_events
        << ",\"cross_wave_pairs\":" << execution.cross_wave_pairs
        << ",\"cross_wave_lane_pairs\":" << execution.cross_wave_lane_pairs
        << ",\"collision_groups\":" << execution.collision_groups
        << ",\"collision_lane_pairs\":" << execution.collision_lane_pairs << ",\"issues\":[";
    bool first_issue = true;
    for (const auto &issue : execution.issues) {
      if (!first_issue)
        out << ',';
      first_issue = false;
      out << "{\"wave\":";
      if (issue.wave == UINT32_MAX)
        out << "null";
      else
        out << issue.wave;
      out << ",\"site\":";
      if (issue.site == UINT32_MAX)
        out << "null";
      else
        out << issue.site;
      out << ",\"reason\":";
      write_json_string(out, issue.reason);
      out << '}';
    }
    out << "],\"cross_wave_examples\":";
    examples(out, execution.cross_wave_examples);
    out << ",\"collision_examples\":";
    examples(out, execution.collision_examples);
    out << '}';
  }
  out << "]}\n";
  return out.str();
}

} // namespace rocjitsu::consan

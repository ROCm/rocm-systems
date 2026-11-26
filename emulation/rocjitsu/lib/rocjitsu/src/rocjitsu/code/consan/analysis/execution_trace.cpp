// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/execution_trace.h"

#include "rocjitsu/code/consan/analysis/render_helpers.h"
#include "rocjitsu/code/consan/encoding.h"
#include "util/utf8.h"
#include <algorithm>
#include <locale>
#include <sstream>

namespace rocjitsu::consan {
namespace {

bool all_zero(std::span<const uint8_t> bytes) {
  return std::ranges::all_of(bytes, [](uint8_t value) { return value == 0; });
}

void issue(ExecutionTraceExecution &result, std::string message) {
  ++result.integrity_events;
  if (result.issues.size() < 8)
    result.issues.push_back(std::move(message));
}

} // namespace

FailureOr<ExecutionTraceResult>
analyze_execution_trace_capture(const Capture &capture, util::DiagnosticEmitter diagnostic) {
  if (validate_image_facts(capture.header.image, diagnostic).failed())
    return Result::failure();
  if (selected_scope(capture.header.image) != ScopeKind::ExecutionTrace ||
      capture.header.execution_conditions != kApplicationTraceExecutionConditions ||
      all_zero(capture.header.run_identity) ||
      capture.executions.size() > CaptureLimits{}.executions)
    return diagnostic.emit() << "invalid execution-trace scope, conditions, identity or budget";
  ExecutionTraceResult result;
  result.run_identity = capture.header.run_identity;
  result.original = capture.header.image.original;
  result.selection_profile = capture.header.image.selection_profile;
  result.transport_complete = !capture.executions.empty();
  for (const ImageEntryFacts &entry : capture.header.image.entries) {
    if (!util::is_valid_utf8(entry.binding.name))
      return diagnostic.emit() << "entry name is not valid UTF-8 for JSON output";
    result.entries.push_back(
        {.key = entry.key, .name = entry.binding.name, .disposition = entry.disposition});
  }
  std::map<uint32_t, uint32_t> generations;
  for (const CapturedExecution &execution : capture.executions) {
    const Manifest *facts = selected_manifest(capture.header.image, execution.entry_key);
    if (facts == nullptr || execution.serial != result.executions.size() + 1 ||
        execution.storage_generation <= generations[execution.storage_slot])
      return diagnostic.emit() << "invalid capture entry or execution identity";
    if (validate_captured_execution(*facts, execution, diagnostic).failed())
      return Result::failure();
    generations[execution.storage_slot] = execution.storage_generation;
    const LaunchLayout layout =
        plan_launch(*facts, execution.dimensions, execution.group_bytes, execution.private_bytes)
            .value();
    ExecutionTraceExecution analyzed = {.serial = execution.serial,
                                        .entry_key = execution.entry_key,
                                        .entry = facts->entry.name,
                                        .dimensions = execution.dimensions,
                                        .expected_rows = layout.wave_count,
                                        .issues = {}};
    const std::span<const uint8_t> report = execution.report;
    const bool parent_valid =
        wire::read_u64(report, wire::kReportSerial) == execution.serial &&
        wire::read_u32(report, wire::kReportEntry) == facts->entry.id &&
        wire::read_u32(report, wire::kReportRows) == layout.wave_count &&
        wire::read_u32(report, wire::kReportRowStride) == kRowHeaderBytes &&
        wire::read_u32(report, wire::kReportSites) == 0 &&
        all_zero(report.subspan(wire::kReportReservedBegin,
                                kReportHeaderBytes - wire::kReportReservedBegin));
    if (!parent_valid)
      issue(analyzed, "invalid_report_parent");
    const uint32_t waves_per_group =
        (execution.dimensions.workgroup[0] + facts->wave_size - 1) / facts->wave_size;
    for (uint32_t row_index = 0; row_index < layout.wave_count; ++row_index) {
      const std::span<const uint8_t> row =
          report.subspan(kReportHeaderBytes + row_index * kRowHeaderBytes, kRowHeaderBytes);
      const bool valid =
          wire::read_u64(row, wire::kRowSerial) == execution.serial &&
          wire::read_u32(row, wire::kRowEntry) == facts->entry.id &&
          wire::read_u32(row, wire::kRowWave) == row_index &&
          wire::read_u32(row, wire::kRowStatus) == (wire::kRowEntered | wire::kRowExited) &&
          wire::read_u32(row, wire::kRowLoss) == 0 &&
          wire::read_u32(row, wire::kRowFinalSequence) == 0 &&
          wire::read_u32(row, wire::kTraceRowWorkgroup) == row_index / waves_per_group &&
          wire::read_u32(row, wire::kTraceRowLocalWave) == row_index % waves_per_group &&
          wire::read_u32(row, wire::kTraceRowEntries) == 1 &&
          wire::read_u32(row, wire::kTraceRowExits) == 1 &&
          all_zero(row.subspan(wire::kTraceRowReservedBegin));
      if (valid && parent_valid) {
        ++analyzed.completed_rows;
        ++analyzed.entry_events;
        ++analyzed.exit_events;
      } else if (!valid)
        issue(analyzed, "invalid_entry_exit_row_" + std::to_string(row_index));
    }
    result.expected_rows += analyzed.expected_rows;
    result.completed_rows += analyzed.completed_rows;
    result.entry_events += analyzed.entry_events;
    result.exit_events += analyzed.exit_events;
    result.transport_complete &= analyzed.integrity_events == 0;
    result.executions.push_back(std::move(analyzed));
  }
  return result;
}

std::string render_execution_trace(const ExecutionTraceResult &result) {
  using render_detail::hex_bytes;
  using render_detail::write_json_string;
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::boolalpha
      << "{\"scope\":\"execution_trace\",\"lds_analysis\":\"not_performed\",\"run_identity\":\""
      << hex_bytes(result.run_identity) << "\",\"original_sha256\":\""
      << hex_bytes(result.original.sha256) << "\",\"original_bytes\":" << result.original.bytes
      << ",\"selection_profile\":\"" << hex_bytes(result.selection_profile)
      << "\",\"transport_complete\":" << result.transport_complete
      << ",\"expected_rows\":" << result.expected_rows
      << ",\"completed_rows\":" << result.completed_rows
      << ",\"entry_events\":" << result.entry_events << ",\"exit_events\":" << result.exit_events
      << ",\"entries\":[";
  bool first = true;
  for (const ExecutionTraceEntry &entry : result.entries) {
    if (!first)
      out << ',';
    first = false;
    out << "{\"entry_key\":" << entry.key << ",\"entry\":";
    write_json_string(out, entry.name);
    out << ",\"disposition\":\""
        << (entry.disposition == EntryDisposition::Selected ? "selected" : "native_uncovered")
        << "\"}";
  }
  out << "],\"executions\":[";
  first = true;
  for (const ExecutionTraceExecution &execution : result.executions) {
    if (!first)
      out << ',';
    first = false;
    out << "{\"serial\":" << execution.serial << ",\"entry_key\":" << execution.entry_key
        << ",\"entry\":";
    write_json_string(out, execution.entry);
    out << ",\"grid\":[" << execution.dimensions.grid[0] << ',' << execution.dimensions.grid[1]
        << ',' << execution.dimensions.grid[2] << "],\"workgroup\":["
        << execution.dimensions.workgroup[0] << ',' << execution.dimensions.workgroup[1] << ','
        << execution.dimensions.workgroup[2] << "],\"expected_rows\":" << execution.expected_rows
        << ",\"completed_rows\":" << execution.completed_rows
        << ",\"entry_events\":" << execution.entry_events
        << ",\"exit_events\":" << execution.exit_events
        << ",\"integrity_events\":" << execution.integrity_events << ",\"issues\":[";
    bool first_issue = true;
    for (const std::string &message : execution.issues) {
      if (!first_issue)
        out << ',';
      first_issue = false;
      write_json_string(out, message);
    }
    out << "]}";
  }
  out << "]}\n";
  return out.str();
}

} // namespace rocjitsu::consan

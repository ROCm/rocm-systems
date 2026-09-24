// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_report_renderer.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>

namespace rocjitsu::consan::hook {

std::string format_report_text(const char *format, ...) {
  std::va_list args;
  va_start(args, format);
  std::va_list size_args;
  va_copy(size_args, args);
  const int size = std::vsnprintf(nullptr, 0, format, size_args);
  va_end(size_args);
  if (size < 0) {
    va_end(args);
    return {};
  }
  std::string result(static_cast<size_t>(size), '\0');
  std::vsnprintf(result.data(), result.size() + 1u, format, args);
  va_end(args);
  return result;
}

std::vector<ReportDiagnostic> render_report(const ReportRenderInput &render_input) {
  std::vector<ReportDiagnostic> rendered;
  const auto append = [&rendered](ReportDiagnosticKind kind, const char *format, auto... args) {
    rendered.push_back({kind, format_report_text(format, args...)});
  };
  constexpr ReportDiagnosticKind kFailure = ReportDiagnosticKind::Failure;
  constexpr ReportDiagnosticKind kSummary = ReportDiagnosticKind::Summary;

  const ReportPipelineInput &input = render_input.pipeline;
  const DecodedReport &decoded = render_input.decoded;
  const ReportSummary &summary = render_input.summary;
  // Valid observations from a deliberately incomplete trace remain useful
  // diagnostics, but are never promoted to ordering evidence.
  for (size_t i = 0; i < std::min<size_t>(decoded.publications.events.size(), 64u); ++i) {
    const auto &event = decoded.publications.events[i];
    append(ReportDiagnosticKind::Detail,
           "ConSan publication event reader=%llu index=%zu sequence=%llu owner=%u lane=%u "
           "address=0x%llx bytes=%u operation=%u observed=%llu written=%llu release=%s acquire=%s "
           "complete=%s",
           static_cast<unsigned long long>(input.reader), i,
           static_cast<unsigned long long>(event.point.sequence), event.point.owner,
           event.point.lane, static_cast<unsigned long long>(event.address), event.bytes,
           static_cast<unsigned>(event.operation), static_cast<unsigned long long>(event.observed),
           static_cast<unsigned long long>(event.written), event.release ? "true" : "false",
           event.acquire ? "true" : "false",
           decoded.publications.status == PublicationDecodeStatus::Complete ? "true" : "false");
  }
  if (!decoded.complete()) {
    const ReportHeader &invalid_header = decoded.header;
    if (decoded.failure == ReportDecodeFailure::InvalidHeader) {
      append(kFailure,
             "ConSan auto report reader=%llu has invalid header magic=0x%08x "
             "abi=%u header_size=%u",
             static_cast<unsigned long long>(input.reader), invalid_header.magic,
             invalid_header.abi_version, invalid_header.header_size);
    } else if (decoded.failure == ReportDecodeFailure::GenerationMismatch) {
      append(kFailure,
             "ConSan auto report reader=%llu has mismatched allocation generation "
             "expected=%llu observed=%llu",
             static_cast<unsigned long long>(input.reader),
             static_cast<unsigned long long>(input.expected_generation.value_or(0)),
             static_cast<unsigned long long>(invalid_header.generation));
    } else if (decoded.failure == ReportDecodeFailure::PublicationEvidenceInvalid) {
      append(
          kFailure,
          "ConSan auto report reader=%llu has incomplete or malformed atomic publication evidence "
          "flags=%u events=%u capacity=%u dropped=%u clock=%llu",
          static_cast<unsigned long long>(input.reader), invalid_header.publication_flags,
          invalid_header.publication_event_count, invalid_header.publication_event_capacity,
          invalid_header.publication_dropped_count,
          static_cast<unsigned long long>(invalid_header.publication_clock));
    } else if (decoded.failure == ReportDecodeFailure::LayoutMismatch) {
      append(kFailure, "ConSan auto report reader=%llu has inconsistent ABI-v%u layout",
             static_cast<unsigned long long>(input.reader), kReportAbiVersion);
    } else {
      append(kFailure, "ConSan auto report reader=%llu has undersized host snapshot",
             static_cast<unsigned long long>(input.reader));
    }
    return rendered;
  }
  if (render_input.conflict_analysis == nullptr) {
    append(kFailure, "ConSan auto report reader=%llu has missing typed analysis",
           static_cast<unsigned long long>(input.reader));
    return rendered;
  }

  const ReportHeader &header = decoded.header;
  const DecodedEvidence &records = decoded.records;
  const ConflictAnalysis &analysis = *render_input.conflict_analysis;
  constexpr ReportDiagnosticKind kEvidence = ReportDiagnosticKind::Evidence;
  constexpr ReportDiagnosticKind kDetail = ReportDiagnosticKind::Detail;
  if (header.publication_clock || header.publication_flags || header.publication_event_count ||
      header.publication_dropped_count)
    append(kDetail,
           "ConSan publication reader=%llu flags=%u clock=%llu events=%u capacity=%u dropped=%u",
           static_cast<unsigned long long>(input.reader), header.publication_flags,
           static_cast<unsigned long long>(header.publication_clock),
           header.publication_event_count, header.publication_event_capacity,
           header.publication_dropped_count);
  for (const EvidenceIssue &issue : records.issues) {
    switch (issue.reason) {
    case EvidenceReason::MalformedWindow:
      rendered.push_back(
          {kEvidence,
           format_report_text(
               "ConSan malformed window index=%u state=%u generation=%llu/%llu "
               "dispatch=%llu/%llu epoch=%u first=%u entries=%u cluster_workgroup_id=%u "
               "snapshot=%u",
               issue.index, static_cast<uint32_t>(issue.words[0]),
               static_cast<unsigned long long>(issue.words[1]),
               static_cast<unsigned long long>(issue.words[2]),
               static_cast<unsigned long long>(issue.words[3]),
               static_cast<unsigned long long>(issue.words[4]),
               static_cast<uint32_t>(issue.words[5]), static_cast<uint32_t>(issue.words[6]),
               static_cast<uint32_t>(issue.words[7]), static_cast<uint32_t>(issue.words[8]),
               static_cast<uint32_t>(issue.words[9]))});
      break;
    case EvidenceReason::EmptyWatchpoint:
      rendered.push_back(
          {kEvidence, format_report_text("ConSan malformed empty watchpoint index=%u state=%u",
                                         issue.index, static_cast<uint32_t>(issue.words[0]))});
      break;
    case EvidenceReason::MalformedWatchpoint:
      rendered.push_back(
          {kEvidence, format_report_text("ConSan malformed packed watchpoint index=%u low=0x%08x "
                                         "high=0x%08x epoch=%u",
                                         issue.index, static_cast<uint32_t>(issue.words[0]),
                                         static_cast<uint32_t>(issue.words[1]),
                                         static_cast<uint32_t>(issue.words[2]))});
      break;
    }
  }

  const uint32_t diagnostic_count = static_cast<uint32_t>(
      std::min<uint64_t>(summary.conflict_count + summary.immediate_conflict_count,
                         std::numeric_limits<uint32_t>::max()));
  append(kSummary,
         "ConSan auto report reader=%llu addr=0x%llx bytes=%zu generation=%llu "
         "code_object=%s event_counter=%u diagnostics=%u"
         " watchpoints=%u visible=%zu sync_capacity=%u "
         "watchpoint_slots_examined=%llu visible_sync=%llu "
         "unsupported_sync=%llu malformed_sync=%llu "
         "pending_acquire_capacity=%u pending_acquires=%u "
         "pending_acquire_contention=%u pending_acquire_collisions=%u "
         "pending_acquire_malformed=%u pending_release_slots_examined=%llu "
         "conflicts=%u suppressed_uniform_write_conflicts=%u "
         "ordered_publication_pairs=%u incomplete_publication_pairs=%u "
         "immediate_conflicts=%llu claimed_windows=%llu "
         "dropped_windows=%llu saturated_windows=%llu "
         "stale_snapshots=%llu incomplete_snapshots=%llu "
         "changed_snapshots=%llu malformed_snapshots=%llu "
         "static_mapping_malformed=%llu"
         " conflict_examples=%zu conflict_pairs_without_example=%u fine_grained=%s",
         static_cast<unsigned long long>(input.reader),
         static_cast<unsigned long long>(input.source_address), input.size,
         static_cast<unsigned long long>(header.generation),
         input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
         header.event_counter, diagnostic_count, input.layout.watchpoint_capacity,
         records.evidence.size(), input.layout.sync_metadata_capacity,
         static_cast<unsigned long long>(records.watchpoint_slots_examined),
         static_cast<unsigned long long>(summary.visible_sync_metadata_count),
         static_cast<unsigned long long>(summary.unsupported_sync_count),
         static_cast<unsigned long long>(summary.malformed_sync_count),
         header.pending_acquire_capacity, header.pending_acquire_count,
         header.pending_acquire_contention_count, header.pending_acquire_collision_count,
         header.pending_acquire_malformed_count,
         static_cast<unsigned long long>(records.pending_release_slots_examined),
         analysis.conflict_count, analysis.suppressed_uniform_write_conflict_count,
         analysis.ordered_publication_pairs, analysis.incomplete_publication_pairs,
         static_cast<unsigned long long>(summary.immediate_conflict_count),
         static_cast<unsigned long long>(summary.claimed_window_count),
         static_cast<unsigned long long>(summary.dropped_window_count),
         static_cast<unsigned long long>(summary.saturated_window_count),
         static_cast<unsigned long long>(summary.stale_snapshot_count),
         static_cast<unsigned long long>(summary.incomplete_snapshot_count),
         static_cast<unsigned long long>(summary.changed_snapshot_count),
         static_cast<unsigned long long>(summary.malformed_snapshot_count),
         static_cast<unsigned long long>(summary.static_mapping_malformed_count),
         analysis.examples.size(),
         analysis.conflict_count - static_cast<uint32_t>(analysis.examples.size()),
         input.fine_grained ? "true" : "false");

  constexpr size_t kLogLimit = 64u;
  for (uint32_t i = 0; i < std::min(records.evidence.size(), kLogLimit); ++i) {
    const Evidence &entry = records.evidence[i];
    const AccessStaticMapping *mapping = entry.static_mapping;
    const uint64_t instruction_offset = mapping != nullptr ? mapping->instruction_offset : 0u;
    const uint32_t relative_slot = mapping != nullptr ? entry.index - mapping->first_slot : 0u;
    const uint32_t range = mapping != nullptr ? relative_slot / mapping->bank_count : 0u;
    const uint32_t bank = mapping != nullptr ? relative_slot % mapping->bank_count : 0u;
    const uint64_t emitted_probe_offset = mapping != nullptr ? mapping->emitted_probe_offset : 0u;
    const uint64_t relocated_guest_offset =
        mapping != nullptr ? mapping->relocated_guest_offset : 0u;
    const uint16_t scratch_vgpr =
        mapping != nullptr ? mapping->scratch_vgpr.value_or(std::numeric_limits<uint16_t>::max())
                           : std::numeric_limits<uint16_t>::max();
    rendered.push_back(
        {kDetail,
         format_report_text(
             "ConSan access reader=%llu index=%u kind=%u owner=%u epoch=%u "
             "generation=%u bytes=[%u,%u) consumed=%s dispatch=0x%llx "
             "workgroup=(%u,%u,%u) cluster_workgroup=%u instruction=0x%llx trampoline=0x%llx "
             "relocated_guest=0x%llx scratch_vgpr=%u range=%u bank=%u mapped=%s "
             "sync_class=%u sync_kind=%u sync_role=%u sync_scope=%u sync_outcome=%u "
             "sync_address=0x%llx sync_bytes=%u sync_epochs=%u/%u publication_sequence=%llu",
             static_cast<unsigned long long>(input.reader), entry.index,
             static_cast<uint32_t>(entry.entry.kind), entry.entry.owner_id, entry.entry.epoch,
             entry.entry.generation, entry.entry.start_byte,
             entry.entry.start_byte + entry.entry.byte_count,
             entry.entry.consumed ? "true" : "false",
             static_cast<unsigned long long>(entry.dispatch_id), entry.workgroup_x,
             entry.workgroup_y, entry.workgroup_z, entry.cluster_workgroup_id,
             static_cast<unsigned long long>(instruction_offset),
             static_cast<unsigned long long>(emitted_probe_offset),
             static_cast<unsigned long long>(relocated_guest_offset), scratch_vgpr, range, bank,
             mapping != nullptr ? "true" : "false",
             static_cast<uint32_t>(entry.sync.classification),
             static_cast<uint32_t>(entry.sync.metadata.kind),
             static_cast<uint32_t>(entry.sync.metadata.role),
             static_cast<uint32_t>(entry.sync.metadata.scope),
             static_cast<uint32_t>(entry.sync.metadata.outcome),
             static_cast<unsigned long long>(entry.sync.metadata.address),
             entry.sync.metadata.byte_count, entry.sync.metadata.epoch_before,
             entry.sync.metadata.epoch_after,
             static_cast<unsigned long long>(entry.publication_sequence))});
  }
  if (records.evidence.size() > kLogLimit) {
    rendered.push_back(
        {kDetail, format_report_text("ConSan access reader=%llu omitted=%zu after log limit=%zu",
                                     static_cast<unsigned long long>(input.reader),
                                     records.evidence.size() - kLogLimit, kLogLimit)});
  }
  for (const auto &[first, second] : analysis.examples) {
    const auto lane_text = [](uint64_t mask) {
      return mask ? format_report_text("0x%016llx", static_cast<unsigned long long>(mask))
                  : std::string("unavailable");
    };

    const auto instruction = [&](const Evidence &entry) {
      const auto *metadata = input.static_metadata;
      std::optional<uint64_t> offset;
      if (metadata) {
        // Check only retained diagnostics. The decoder's first matching
        // mapping is insufficient to promise a unique instruction location.
        for (const auto &mapping : metadata->mappings) {
          if (entry.index < mapping.first_slot ||
              static_cast<uint64_t>(entry.index - mapping.first_slot) >=
                  static_cast<uint64_t>(mapping.range_count) * mapping.bank_count)
            continue;
          if (offset && *offset != mapping.instruction_offset)
            return std::string("ambiguous");
          offset = mapping.instruction_offset;
        }
      } else if (entry.static_mapping) {
        offset = entry.static_mapping->instruction_offset;
      }
      return offset ? format_report_text("0x%llx", static_cast<unsigned long long>(*offset))
                    : std::string("unavailable");
    };
    const std::string first_instruction = instruction(first);
    const std::string second_instruction = instruction(second);
    rendered.push_back(
        {kDetail,
         format_report_text(
             "ConSan conflict reader=%llu first_index=%u second_index=%u "
             "first_kind=%u second_kind=%u first_owner=%u second_owner=%u epoch=%u "
             "generation=%u first_bytes=[%u,%u) second_bytes=[%u,%u) "
             "code_object=%.*s first_instruction=%s second_instruction=%s "
             "dispatch=0x%llx workgroup=(%u,%u,%u) cluster_workgroup=%u "
             "first_lanes=%s second_lanes=%s",
             static_cast<unsigned long long>(input.reader), first.index, second.index,
             static_cast<uint32_t>(first.entry.kind), static_cast<uint32_t>(second.entry.kind),
             first.entry.owner_id, second.entry.owner_id, second.entry.epoch,
             second.entry.generation, first.entry.start_byte,
             first.entry.start_byte + first.entry.byte_count, second.entry.start_byte,
             second.entry.start_byte + second.entry.byte_count,
             static_cast<int>(input.input_fingerprint.size()), input.input_fingerprint.data(),
             first_instruction.c_str(), second_instruction.c_str(),
             static_cast<unsigned long long>(second.dispatch_id), second.workgroup_x,
             second.workgroup_y, second.workgroup_z, second.cluster_workgroup_id,
             lane_text(first.exact_lane_mask).c_str(), lane_text(second.exact_lane_mask).c_str())});
  }
  return rendered;
}

} // namespace rocjitsu::consan::hook

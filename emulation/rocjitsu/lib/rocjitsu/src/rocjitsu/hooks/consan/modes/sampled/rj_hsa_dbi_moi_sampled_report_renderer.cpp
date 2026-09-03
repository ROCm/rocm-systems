// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_sampled_report_renderer.h"

#include <algorithm>
#include <limits>

namespace rocjitsu::consan_hook {

AutoMoiModeRendering render_auto_moi_sampled_report(
    const AutoMoiReportPipelineInput &input, const ConSanMoiReportHeader &header,
    const AutoMoiReportSummary &summary, const AutoMoiSampledDecodedReport &mode,
    const AutoMoiSampledConflictAnalysis &analysis) {
  AutoMoiModeRendering result;
  result.effective_diagnostic_count = header.diagnostic_count;
  result.summary_fields = format_auto_moi_report_text(
      " sampled_watchpoints=%u visible_sampled=%zu sampled_sync_capacity=%u "
      "sampled_watchpoint_slots_examined=%llu visible_sampled_sync=%llu "
      "sampled_unsupported_sync=%llu sampled_malformed_sync=%llu "
      "sampled_pending_acquire_capacity=%u sampled_pending_acquires=%u "
      "sampled_pending_acquire_contention=%u sampled_pending_acquire_collisions=%u "
      "sampled_pending_acquire_malformed=%u sampled_pending_release_slots_examined=%llu "
      "sampled_conflicts=%u sampled_immediate_conflicts=%llu sampled_claimed_windows=%llu "
      "sampled_dropped_windows=%llu sampled_saturated_windows=%llu "
      "sampled_stale_snapshots=%llu sampled_incomplete_snapshots=%llu "
      "sampled_changed_snapshots=%llu sampled_malformed_snapshots=%llu "
      "sampled_static_mapping_malformed=%llu",
      input.layout.sampled_watchpoint_capacity, mode.evidence.size(),
      input.layout.sampled_sync_metadata_capacity,
      static_cast<unsigned long long>(mode.watchpoint_slots_examined),
      static_cast<unsigned long long>(summary.visible_sampled_sync_metadata_count),
      static_cast<unsigned long long>(summary.sampled_unsupported_sync_count),
      static_cast<unsigned long long>(summary.sampled_malformed_sync_count),
      header.sampled_pending_acquire_capacity, header.sampled_pending_acquire_count,
      header.sampled_pending_acquire_contention_count,
      header.sampled_pending_acquire_collision_count,
      header.sampled_pending_acquire_malformed_count,
      static_cast<unsigned long long>(mode.pending_release_slots_examined), analysis.conflict_count,
      static_cast<unsigned long long>(summary.sampled_immediate_conflict_count),
      static_cast<unsigned long long>(summary.sampled_claimed_window_count),
      static_cast<unsigned long long>(summary.sampled_dropped_window_count),
      static_cast<unsigned long long>(summary.sampled_saturated_window_count),
      static_cast<unsigned long long>(summary.sampled_stale_snapshot_count),
      static_cast<unsigned long long>(summary.sampled_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.sampled_changed_snapshot_count),
      static_cast<unsigned long long>(summary.sampled_malformed_snapshot_count),
      static_cast<unsigned long long>(summary.sampled_static_mapping_malformed_count));

  constexpr AutoMoiReportDiagnosticKind kEvidence = AutoMoiReportDiagnosticKind::Evidence;
  constexpr AutoMoiReportDiagnosticKind kDetail = AutoMoiReportDiagnosticKind::Detail;
  for (const AutoMoiSampledEvidenceIssue &issue : mode.issues) {
    switch (issue.reason) {
    case AutoMoiSampledEvidenceReason::MalformedWindow:
      result.evidence.push_back(
          {kEvidence,
           format_auto_moi_report_text(
               "ConSan MOI sampled malformed window index=%u state=%u generation=%llu/%llu "
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
    case AutoMoiSampledEvidenceReason::EmptyWatchpoint:
      result.evidence.push_back(
          {kEvidence, format_auto_moi_report_text(
                          "ConSan MOI sampled malformed empty watchpoint index=%u state=%u",
                          issue.index, static_cast<uint32_t>(issue.words[0]))});
      break;
    case AutoMoiSampledEvidenceReason::MalformedWatchpoint:
      result.evidence.push_back(
          {kEvidence,
           format_auto_moi_report_text(
               "ConSan MOI sampled malformed packed watchpoint index=%u low=0x%08x "
               "high=0x%08x epoch=%u",
               issue.index, static_cast<uint32_t>(issue.words[0]),
               static_cast<uint32_t>(issue.words[1]), static_cast<uint32_t>(issue.words[2]))});
      break;
    }
  }

  constexpr size_t kSampledLogLimit = 64u;
  for (uint32_t i = 0; i < std::min(mode.evidence.size(), kSampledLogLimit); ++i) {
    const AutoMoiSampledEvidence &entry = mode.evidence[i];
    const AutoMoiSampledStaticMapping *mapping = entry.static_mapping;
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
    result.details.push_back(
        {kDetail, format_auto_moi_report_text(
                      "ConSan MOI auto sampled reader=%llu index=%u kind=%u owner=%u epoch=%u "
                      "generation=%u bytes=[%u,%u) consumed=%s dispatch=0x%llx "
                      "workgroup=(%u,%u,%u) instruction=0x%llx trampoline=0x%llx "
                      "relocated_guest=0x%llx scratch_vgpr=%u range=%u bank=%u mapped=%s "
                      "sync_class=%u sync_kind=%u sync_role=%u sync_scope=%u sync_outcome=%u "
                      "sync_address=0x%llx sync_bytes=%u sync_epochs=%u/%u",
                      static_cast<unsigned long long>(input.reader), entry.index,
                      static_cast<uint32_t>(entry.entry.kind), entry.entry.owner_id,
                      entry.entry.epoch, entry.entry.generation, entry.entry.start_byte,
                      entry.entry.start_byte + entry.entry.byte_count,
                      entry.entry.consumed ? "true" : "false",
                      static_cast<unsigned long long>(entry.dispatch_id), entry.workgroup_x,
                      entry.workgroup_y, entry.workgroup_z,
                      static_cast<unsigned long long>(instruction_offset),
                      static_cast<unsigned long long>(emitted_probe_offset),
                      static_cast<unsigned long long>(relocated_guest_offset), scratch_vgpr, range,
                      bank, mapping != nullptr ? "true" : "false",
                      static_cast<uint32_t>(entry.sync.classification),
                      static_cast<uint32_t>(entry.sync.metadata.kind),
                      static_cast<uint32_t>(entry.sync.metadata.role),
                      static_cast<uint32_t>(entry.sync.metadata.scope),
                      static_cast<uint32_t>(entry.sync.metadata.outcome),
                      static_cast<unsigned long long>(entry.sync.metadata.address),
                      entry.sync.metadata.byte_count, entry.sync.metadata.epoch_before,
                      entry.sync.metadata.epoch_after)});
  }
  if (mode.evidence.size() > kSampledLogLimit) {
    result.details.push_back(
        {kDetail, format_auto_moi_report_text(
                      "ConSan MOI auto sampled reader=%llu omitted=%zu after log limit=%zu",
                      static_cast<unsigned long long>(input.reader),
                      mode.evidence.size() - kSampledLogLimit, kSampledLogLimit)});
  }
  if (analysis.first_conflict) {
    const AutoMoiSampledEvidence &first = analysis.first_conflict->first;
    const AutoMoiSampledEvidence &second = analysis.first_conflict->second;
    result.details.push_back(
        {kDetail, format_auto_moi_report_text(
                      "ConSan MOI auto sampled conflict reader=%llu first_index=%u second_index=%u "
                      "first_kind=%u second_kind=%u first_owner=%u second_owner=%u epoch=%u "
                      "generation=%u first_bytes=[%u,%u) second_bytes=[%u,%u)",
                      static_cast<unsigned long long>(input.reader), first.index, second.index,
                      static_cast<uint32_t>(first.entry.kind),
                      static_cast<uint32_t>(second.entry.kind), first.entry.owner_id,
                      second.entry.owner_id, second.entry.epoch, second.entry.generation,
                      first.entry.start_byte, first.entry.start_byte + first.entry.byte_count,
                      second.entry.start_byte, second.entry.start_byte + second.entry.byte_count)});
  }
  return result;
}

} // namespace rocjitsu::consan_hook

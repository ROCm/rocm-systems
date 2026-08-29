// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_pipeline.h"

#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_analyzer.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::consan_hook {

AutoMoiReportSummary summarize_auto_moi_report(const AutoMoiReportPipelineInput &input,
                                               const AutoMoiReportSnapshot &snapshot,
                                               AutoMoiReportSummary summary) {
  const bool partition_mask_debug = [] {
    const char *debug = std::getenv("RJ_CONSAN_MOI_PARTITION_MASK_DEBUG");
    return debug != nullptr && std::string_view(debug) == "1";
  }();
  AutoMoiDecodedReport decoded =
      decode_auto_moi_report(input, snapshot, summary, partition_mask_debug);
  if (!decoded.complete()) {
    const ConSanMoiReportHeader &invalid_header = decoded.header;
    if (decoded.failure == AutoMoiReportDecodeFailure::InvalidHeader) {
      log_message(kLogInfo,
                  "ConSan MOI auto report reader=%llu has invalid header magic=0x%08x "
                  "abi=%u header_size=%u",
                  static_cast<unsigned long long>(input.reader), invalid_header.magic,
                  invalid_header.abi_version, invalid_header.header_size);
    } else if (decoded.failure == AutoMoiReportDecodeFailure::LayoutMismatch) {
      log_message(kLogInfo, "ConSan MOI auto report reader=%llu has inconsistent ABI-v%u layout",
                  static_cast<unsigned long long>(input.reader), kConSanMoiReportAbiVersion);
    } else {
      log_message(kLogInfo, "ConSan MOI auto report reader=%llu has undersized host snapshot",
                  static_cast<unsigned long long>(input.reader));
    }
    return decoded.summary;
  }

  summary = decoded.summary;
  const ConSanMoiEngine expected_engine = decoded.engine;
  const ConSanMoiReportHeader *header = &decoded.header;
  const ConSanMoiReportBufferLayout &expected_layout = input.layout;
  const uint32_t access_record_count = decoded.access_record_count;
  const uint32_t barrier_record_count = decoded.barrier_record_count;
  const uint32_t atomic_record_count = decoded.atomic_record_count;
  const uint32_t visible_records = decoded.visible_record_slot_count;
  const uint32_t visible_barriers = static_cast<uint32_t>(decoded.barrier_records.size());
  const uint32_t visible_atomics = static_cast<uint32_t>(decoded.atomic_records.size());
  const uint32_t visible_fences = static_cast<uint32_t>(decoded.fence_records.size());
  const uint32_t visible_diagnostics = static_cast<uint32_t>(decoded.diagnostics.size());
  const uint32_t effective_diagnostic_count = decoded.effective_diagnostic_count;
  const uint32_t dropped_records = static_cast<uint32_t>(summary.dropped_access_record_count);
  const uint32_t dropped_barriers = static_cast<uint32_t>(summary.dropped_barrier_record_count);
  const uint32_t dropped_atomics = static_cast<uint32_t>(summary.dropped_atomic_record_count);
  const uint32_t dropped_fences = static_cast<uint32_t>(summary.dropped_fence_record_count);
  const uint32_t dropped_diagnostics =
      static_cast<uint32_t>(summary.dropped_diagnostic_record_count);
  const uint32_t sampled_watchpoint_capacity = decoded.sampled_watchpoint_capacity;
  const uint32_t sampled_sync_metadata_capacity = decoded.sampled_sync_metadata_capacity;
  const uint64_t sampled_watchpoint_slots_examined = decoded.sampled_watchpoint_slots_examined;
  const uint64_t sampled_pending_release_slots_examined =
      decoded.sampled_pending_release_slots_examined;
  const uint32_t visible_sampled_sync_metadata =
      static_cast<uint32_t>(summary.visible_sampled_sync_metadata_count);
  const auto &visible_exact_shadow = decoded.exact_shadow;
  const auto &visible_inline_atomic_releases = decoded.inline_atomic_releases;
  const auto &visible_inline_acquired_tokens = decoded.inline_acquired_tokens;
  const auto &visible_sampled = decoded.sampled;
  const auto &replay_access_records = decoded.replay_access_records;
  const uint32_t committed_records = static_cast<uint32_t>(summary.visible_access_record_count);
  const ConSanMoiBarrierRecord *barriers = decoded.barrier_records.data();
  const ConSanMoiAtomicRecord *atomics = decoded.atomic_records.data();
  const ConSanMoiFenceRecord *fences = decoded.fence_records.data();
  const ConSanMoiAccessRecord *records = decoded.visible_access_slots.data();
  const ConSanMoiDiagnosticRecord *diagnostics = decoded.diagnostics.data();

  const AutoMoiSampledConflictAnalysis sampled_analysis = analyze_auto_moi_sampled_conflicts(
      decoded.sampled, decoded.sampled_synchronization_evidence_complete);
  const uint32_t sampled_conflicts = sampled_analysis.conflict_count;
  const auto &first_sampled_conflict = sampled_analysis.first_conflict;
  const AutoMoiRecordReplayAnalysis record_replay_analysis = analyze_auto_moi_record_replay(
      *header, expected_engine, decoded.replay_access_records, decoded.barrier_records,
      decoded.atomic_records, decoded.fence_records, input.record_replay_static_mappings,
      input.record_replay_static_mapping_malformed,
      expected_layout.record_replay_logical_access_range_count,
      expected_layout.record_replay_address_group_headroom);
  const RecordReplayPressureTelemetry &record_replay_pressure = record_replay_analysis.pressure;
  summary.record_replay_bank_saturation_count =
      record_replay_bank_saturation_count(*header, expected_engine);
  summary.record_replay_invalid_site_token_count = record_replay_pressure.invalid_site_token_count;
  summary.replay_conflict_count = record_replay_analysis.effective_conflict ? 1u : 0u;
  summary.replay_diagnostic_count = record_replay_analysis.effective_diagnostic_count;
  summary.replay_dropped_access_count = record_replay_analysis.replay.dropped_access_count;
  summary.replay_dropped_barrier_count = record_replay_analysis.replay.dropped_barrier_count;
  summary.replay_unsupported_access_count = record_replay_analysis.replay.unsupported_access_count;
  summary.replay_unsupported_atomic_count = record_replay_analysis.replay.unsupported_atomic_count;
  summary.replay_unsupported_fence_count = record_replay_analysis.replay.unsupported_fence_count;
  summary.replay_metadata_full_count = record_replay_analysis.replay.metadata_full ? 1u : 0u;
  summary.replay_diagnostic_capacity_exhausted_count =
      record_replay_analysis.replay.diagnostic_capacity_exhausted ? 1u : 0u;
  summary.sampled_conflict_count = sampled_conflicts;

  if (decoded.deferred_token_qualified_diagnostic_count != 0) {
    log_message(kLogInfo,
                "ConSan MOI deferred-token qualification reader=%llu ordered_diagnostics=%u",
                static_cast<unsigned long long>(input.reader),
                decoded.deferred_token_qualified_diagnostic_count);
  }
  bool rendered_first_exact_malformed = false;
  for (const AutoMoiReportEvidenceIssue &issue : decoded.issues) {
    switch (issue.reason) {
    case AutoMoiReportEvidenceReason::ExactMalformed:
      if (!rendered_first_exact_malformed) {
        log_message(kLogInfo,
                    "ConSan MOI first malformed exact snapshot reader=%llu index=%u "
                    "version_before=%u packed_access=0x%016llx dispatch_id=0x%016llx "
                    "byte_provenance=0x%08x version_after=%u",
                    static_cast<unsigned long long>(input.reader), issue.index,
                    static_cast<uint32_t>(issue.words[0]),
                    static_cast<unsigned long long>(issue.words[1]),
                    static_cast<unsigned long long>(issue.words[2]),
                    static_cast<uint32_t>(issue.words[3]), static_cast<uint32_t>(issue.words[4]));
        rendered_first_exact_malformed = true;
      }
      break;
    case AutoMoiReportEvidenceReason::ReleasePublishing:
      log_message(kLogInfo,
                  "ConSan MOI auto incomplete-inline-atomic-release reader=%llu index=%u "
                  "version=%u owner=%u epoch_plus_one=%u workgroup=%u address=0x%llx "
                  "dispatch=0x%llx",
                  static_cast<unsigned long long>(input.reader), issue.index,
                  static_cast<uint32_t>(issue.words[0]), static_cast<uint32_t>(issue.words[1]),
                  static_cast<uint32_t>(issue.words[2]), static_cast<uint32_t>(issue.words[3]),
                  static_cast<unsigned long long>(issue.words[4]),
                  static_cast<unsigned long long>(issue.words[5]));
      break;
    case AutoMoiReportEvidenceReason::SampledMalformedWindow:
      log_message(kLogInfo,
                  "ConSan MOI sampled malformed window index=%u state=%u generation=%llu/%llu "
                  "dispatch=%llu/%llu epoch=%u first=%u entries=%u "
                  "cluster_workgroup_id=%u snapshot=%u",
                  issue.index, static_cast<uint32_t>(issue.words[0]),
                  static_cast<unsigned long long>(issue.words[1]),
                  static_cast<unsigned long long>(issue.words[2]),
                  static_cast<unsigned long long>(issue.words[3]),
                  static_cast<unsigned long long>(issue.words[4]),
                  static_cast<uint32_t>(issue.words[5]), static_cast<uint32_t>(issue.words[6]),
                  static_cast<uint32_t>(issue.words[7]), static_cast<uint32_t>(issue.words[8]),
                  static_cast<uint32_t>(issue.words[9]));
      break;
    case AutoMoiReportEvidenceReason::SampledEmptyWatchpoint:
      log_message(kLogInfo, "ConSan MOI sampled malformed empty watchpoint index=%u state=%u",
                  issue.index, static_cast<uint32_t>(issue.words[0]));
      break;
    case AutoMoiReportEvidenceReason::SampledMalformedWatchpoint:
      log_message(kLogInfo,
                  "ConSan MOI sampled malformed packed watchpoint index=%u low=0x%08x "
                  "high=0x%08x epoch=%u",
                  issue.index, static_cast<uint32_t>(issue.words[0]),
                  static_cast<uint32_t>(issue.words[1]), static_cast<uint32_t>(issue.words[2]));
      break;
    case AutoMoiReportEvidenceReason::CompactDiagnosticTokenUnresolved:
      log_message(kLogInfo,
                  "ConSan MOI compact diagnostic token unresolved reader=%llu current=0x%x "
                  "tagged=0x%x well_formed=%s current_ambiguous=%s "
                  "prior_ambiguous=%s",
                  static_cast<unsigned long long>(input.reader), issue.index,
                  static_cast<uint32_t>(issue.words[0]), issue.words[1] != 0 ? "true" : "false",
                  issue.words[2] != 0 ? "true" : "false", issue.words[3] != 0 ? "true" : "false");
      break;
    default:
      break;
    }
  }
  if (partition_mask_debug) {
    const uint64_t group_mask = header->dispatch_id;
    const uint64_t first_exchange_address =
        static_cast<uint64_t>(header->access_record_count) |
        (static_cast<uint64_t>(header->barrier_record_count) << 32u);
    const uint64_t final_exchange_address = static_cast<uint64_t>(header->atomic_record_count) |
                                            (static_cast<uint64_t>(header->flags) << 32u);
    log_message(kLogInfo,
                "ConSan MOI partition-mask debug reader=%llu acceptance=false "
                "final_group=0x%016llx exchange_count=%u "
                "first_exchange_address=0x%016llx final_exchange_address=0x%016llx",
                static_cast<unsigned long long>(input.reader),
                static_cast<unsigned long long>(group_mask), header->event_counter,
                static_cast<unsigned long long>(first_exchange_address),
                static_cast<unsigned long long>(final_exchange_address));
  }
  log_message(
      kLogInfo,
      "ConSan MOI auto report reader=%llu addr=0x%llx bytes=%zu generation=%llu "
      "code_object=%s "
      "event_counter=%u access_records=%u visible_records=%u committed_records=%u "
      "dropped_records=%u "
      "capacity=%u dispatch_tokens=%u/%u record_replay_flags=0x%x "
      "record_replay_bank_saturated=%s "
      "record_replay_pressure_available=%s "
      "record_replay_pressure_unavailable_reason=%.*s "
      "record_replay_access_table_occupied=%llu record_replay_access_table_capacity=%llu "
      "record_replay_observed_sites=%llu record_replay_max_site_owner_address_groups=%llu "
      "record_replay_address_group_headroom=%u record_replay_logical_access_ranges=%u "
      "record_replay_max_site_token=%u record_replay_invalid_site_tokens=%llu "
      "barrier_records=%u visible_barriers=%u dropped_barriers=%u barrier_capacity=%u "
      "atomic_records=%u visible_atomics=%u dropped_atomics=%u atomic_capacity=%u "
      "fence_records=%u visible_fences=%u dropped_fences=%u fence_capacity=%u "
      "diagnostics=%u visible_diagnostics=%u dropped_diagnostics=%u "
      "diagnostic_capacity=%u "
      "exact_shadow_capacity=%u visible_exact_shadow=%zu "
      "exact_incomplete_snapshots=%llu exact_changed_snapshots=%llu "
      "exact_malformed_snapshots=%llu "
      "inline_atomic_release_capacity=%u visible_inline_atomic_releases=%zu "
      "release_incomplete_snapshots=%llu release_changed_snapshots=%llu "
      "release_overflow_snapshots=%llu release_source_incomplete_snapshots=%llu "
      "release_malformed_snapshots=%llu "
      "inline_acquired_token_capacity=%u visible_inline_acquired_tokens=%zu "
      "token_incomplete_snapshots=%llu token_changed_snapshots=%llu "
      "token_malformed_snapshots=%llu "
      "inline_undercoverage=%llu inline_overflow=%llu "
      "inline_unsupported=%llu inline_malformed=%llu "
      "sampled_watchpoints=%u visible_sampled=%zu sampled_sync_capacity=%u "
      "sampled_watchpoint_slots_examined=%llu "
      "visible_sampled_sync=%u sampled_unsupported_sync=%u sampled_malformed_sync=%llu "
      "sampled_pending_acquire_capacity=%u sampled_pending_acquires=%u "
      "sampled_pending_acquire_contention=%u "
      "sampled_pending_acquire_collisions=%u sampled_pending_acquire_malformed=%u "
      "sampled_pending_release_slots_examined=%llu "
      "sampled_conflicts=%u "
      "sampled_immediate_conflicts=%u sampled_claimed_windows=%u "
      "sampled_dropped_windows=%u sampled_saturated_windows=%u "
      "sampled_stale_snapshots=%llu sampled_incomplete_snapshots=%llu "
      "sampled_changed_snapshots=%llu sampled_malformed_snapshots=%llu "
      "sampled_static_mapping_malformed=%llu "
      "fine_grained=%s",
      static_cast<unsigned long long>(input.reader),
      static_cast<unsigned long long>(input.source_address), input.size,
      static_cast<unsigned long long>(header->generation),
      input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
      header->event_counter, access_record_count, visible_records, committed_records,
      dropped_records, header->access_record_capacity, header->record_replay_dispatch_token_count,
      header->record_replay_dispatch_token_capacity,
      expected_engine == ConSanMoiEngine::RecordReplay ? header->flags : 0u,
      summary.record_replay_bank_saturation_count != 0 ? "true" : "false",
      record_replay_pressure.available ? "true" : "false",
      static_cast<int>(
          record_replay_pressure_unavailable_reason_name(record_replay_pressure.unavailable_reason)
              .size()),
      record_replay_pressure_unavailable_reason_name(record_replay_pressure.unavailable_reason)
          .data(),
      static_cast<unsigned long long>(record_replay_pressure.occupied_access_record_count),
      static_cast<unsigned long long>(record_replay_pressure.access_record_capacity),
      static_cast<unsigned long long>(record_replay_pressure.observed_site_count),
      static_cast<unsigned long long>(
          record_replay_pressure.maximum_site_owner_address_group_count),
      record_replay_pressure.address_group_headroom,
      record_replay_pressure.logical_access_range_count, record_replay_pressure.maximum_site_token,
      static_cast<unsigned long long>(record_replay_pressure.invalid_site_token_count),
      barrier_record_count, visible_barriers, dropped_barriers, header->barrier_record_capacity,
      atomic_record_count, visible_atomics, dropped_atomics, header->atomic_record_capacity,
      header->fence_record_count, visible_fences, dropped_fences, input.fence_record_capacity,
      effective_diagnostic_count, visible_diagnostics, dropped_diagnostics,
      header->diagnostic_capacity, header->exact_shadow_entry_capacity, visible_exact_shadow.size(),
      static_cast<unsigned long long>(summary.exact_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.exact_changed_snapshot_count),
      static_cast<unsigned long long>(summary.exact_malformed_snapshot_count),
      header->inline_atomic_release_capacity, visible_inline_atomic_releases.size(),
      static_cast<unsigned long long>(summary.release_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.release_changed_snapshot_count),
      static_cast<unsigned long long>(summary.release_overflow_snapshot_count),
      static_cast<unsigned long long>(summary.release_source_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.release_malformed_snapshot_count),
      header->inline_acquired_epoch_token_capacity, visible_inline_acquired_tokens.size(),
      static_cast<unsigned long long>(summary.token_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.token_changed_snapshot_count),
      static_cast<unsigned long long>(summary.token_malformed_snapshot_count),
      static_cast<unsigned long long>(summary.inline_undercoverage_count),
      static_cast<unsigned long long>(summary.inline_overflow_count),
      static_cast<unsigned long long>(summary.inline_unsupported_count),
      static_cast<unsigned long long>(summary.inline_malformed_count), sampled_watchpoint_capacity,
      visible_sampled.size(), sampled_sync_metadata_capacity,
      static_cast<unsigned long long>(sampled_watchpoint_slots_examined),
      visible_sampled_sync_metadata, header->sampled_unsupported_sync_count,
      static_cast<unsigned long long>(summary.sampled_malformed_sync_count),
      header->sampled_pending_acquire_capacity, header->sampled_pending_acquire_count,
      header->sampled_pending_acquire_contention_count,
      header->sampled_pending_acquire_collision_count,
      header->sampled_pending_acquire_malformed_count,
      static_cast<unsigned long long>(sampled_pending_release_slots_examined), sampled_conflicts,
      static_cast<uint32_t>(summary.sampled_immediate_conflict_count),
      header->sampled_causal_window_count, header->sampled_dropped_window_count,
      header->sampled_saturated_window_count,
      static_cast<unsigned long long>(summary.sampled_stale_snapshot_count),
      static_cast<unsigned long long>(summary.sampled_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.sampled_changed_snapshot_count),
      static_cast<unsigned long long>(summary.sampled_malformed_snapshot_count),
      static_cast<unsigned long long>(summary.sampled_static_mapping_malformed_count),
      input.fine_grained ? "true" : "false");

  for (size_t i = 0; i < visible_inline_atomic_releases.size(); ++i) {
    const auto &release = visible_inline_atomic_releases[i];
    const auto &slot = release.slot;
    const auto &snapshot = release.snapshot;
    log_message(kLogInfo,
                "ConSan MOI auto inline-atomic-release reader=%llu index=%u version=%u "
                "owner=%u epoch_plus_one=%u workgroup=%u address=0x%llx dispatch=0x%llx "
                "snapshot_flags=%u snapshot_count=%u "
                "snapshot0_owner=%u snapshot0_epoch_plus_one=%u "
                "snapshot1_owner=%u snapshot1_epoch_plus_one=%u "
                "snapshot2_owner=%u snapshot2_epoch_plus_one=%u "
                "snapshot3_owner=%u snapshot3_epoch_plus_one=%u",
                static_cast<unsigned long long>(input.reader), release.index, slot.version,
                slot.owner_id, slot.epoch_plus_one, slot.workgroup_key,
                static_cast<unsigned long long>(slot.atomic_address),
                static_cast<unsigned long long>(slot.dispatch_id), snapshot.flags,
                snapshot.entry_count, snapshot.entries[0].ancestor_owner_id,
                snapshot.entries[0].ancestor_epoch_plus_one, snapshot.entries[1].ancestor_owner_id,
                snapshot.entries[1].ancestor_epoch_plus_one, snapshot.entries[2].ancestor_owner_id,
                snapshot.entries[2].ancestor_epoch_plus_one, snapshot.entries[3].ancestor_owner_id,
                snapshot.entries[3].ancestor_epoch_plus_one);
  }

  for (const auto &entry_token : visible_inline_acquired_tokens) {
    const auto &token = entry_token.token;
    log_message(
        kLogInfo,
        "ConSan MOI auto inline-acquired-token reader=%llu index=%u version=%u "
        "kind=%s consumer=%u producer=%u producer_epoch_plus_one=%u "
        "consumer_epoch_plus_one=%u workgroup=%u dispatch=0x%llx "
        "source_address=0x%llx source_version=%u",
        static_cast<unsigned long long>(input.reader), entry_token.index, token.version,
        token.kind == static_cast<uint32_t>(rocjitsu::ConSanMoiInlineTokenEvidenceKind::Direct)
            ? "direct"
        : token.kind == static_cast<uint32_t>(rocjitsu::ConSanMoiInlineTokenEvidenceKind::Inherited)
            ? "inherited"
            : "release-sequence",
        token.consumer_owner_id, token.producer_owner_id, token.producer_epoch_plus_one,
        token.consumer_epoch_plus_one, token.workgroup_key,
        static_cast<unsigned long long>(token.dispatch_id),
        static_cast<unsigned long long>(token.source_release_address),
        token.source_release_version);
  }

  const uint32_t fence_sample_count = consan_moi_auto_detail_log_count(visible_fences);
  for (uint32_t i = 0; i < fence_sample_count; ++i) {
    const rocjitsu::ConSanMoiFenceRecord &fence = fences[i];
    log_message(kLogInfo,
                "ConSan MOI auto fence reader=%llu index=%u event_index=%u owner=%u "
                "generation=%llu epoch=%u workgroup=(%u,%u,%u) inst=0x%x kind=%u scope=%u "
                "semantics=%u token=0x%016llx",
                static_cast<unsigned long long>(input.reader), i, fence.event_index, fence.owner_id,
                static_cast<unsigned long long>(fence.generation), fence.epoch, fence.workgroup_x,
                fence.workgroup_y, fence.workgroup_z, fence.instruction_offset,
                static_cast<unsigned>(fence.kind), fence.scope, fence.semantics,
                static_cast<unsigned long long>(fence.communication_token));
  }
  if (visible_fences > fence_sample_count) {
    log_message(kLogInfo, "ConSan MOI auto fence reader=%llu omitted=%u after log limit=%u",
                static_cast<unsigned long long>(input.reader), visible_fences - fence_sample_count,
                kConSanMoiAutoDetailLogLimit);
  }
  if (record_replay_analysis.shadow_bounded) {
    log_message(kLogInfo,
                "ConSan MOI replay shadow bounded reader=%llu generation=%llu code_object=%s "
                "required_shadow_entries=%llu limit=%llu shadow_entries=%llu",
                static_cast<unsigned long long>(input.reader),
                static_cast<unsigned long long>(header->generation),
                input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
                static_cast<unsigned long long>(record_replay_analysis.required_shadow_entry_count),
                static_cast<unsigned long long>(1u << 20u),
                static_cast<unsigned long long>(record_replay_analysis.replay_shadow_entry_count));
  }
  if (record_replay_analysis.replay_performed) {
    const ConSanMoiRecordReplayResult &replay = record_replay_analysis.replay;
    log_message(
        kLogInfo,
        "ConSan MOI auto replay reader=%llu generation=%llu code_object=%s "
        "replay_input_access=%zu published_access=%u processed_access=%u "
        "processed_barriers=%u processed_atomics=%u processed_fences=%u "
        "dropped_access=%u "
        "dropped_barriers=%u unsupported_access=%u unsupported_atomics=%u "
        "unsupported_fences=%u diagnostics=%u "
        "conflict=%s metadata_full=%s diagnostic_capacity_exhausted=%s "
        "release_metadata_max=%u acquired_metadata_max=%u "
        "diagnostic_capacity=%u replay_scratch_diagnostic_capacity=%u "
        "provenance_repaired=%u provenance_unresolved=%u "
        "disjoint_owner_suppressed=%u "
        "shadow_entries=%llu",
        static_cast<unsigned long long>(input.reader),
        static_cast<unsigned long long>(record_replay_analysis.replay_header.generation),
        input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
        replay_access_records.size(), replay.published_access_count, replay.processed_access_count,
        replay.processed_barrier_count, replay.processed_atomic_count, replay.processed_fence_count,
        replay.dropped_access_count, replay.dropped_barrier_count, replay.unsupported_access_count,
        replay.unsupported_atomic_count, replay.unsupported_fence_count,
        record_replay_analysis.effective_diagnostic_count,
        record_replay_analysis.effective_conflict ? "true" : "false",
        replay.metadata_full ? "true" : "false",
        replay.diagnostic_capacity_exhausted ? "true" : "false",
        replay.maximum_atomic_release_metadata_count, replay.maximum_acquired_epoch_metadata_count,
        header->diagnostic_capacity, record_replay_analysis.replay_header.diagnostic_capacity,
        record_replay_analysis.provenance.repaired_diagnostic_count,
        record_replay_analysis.provenance.unresolved_diagnostic_count,
        record_replay_analysis.disjoint_owner_suppressed_count,
        static_cast<unsigned long long>(record_replay_analysis.replay_shadow_entry_count));
    for (uint32_t index = 0; index < record_replay_analysis.diagnostics.size(); ++index) {
      const ConSanMoiDiagnosticRecord &diagnostic = record_replay_analysis.diagnostics[index];
      log_message(kLogInfo,
                  "ConSan MOI auto replay diagnostic reader=%llu index=%u kind=%u "
                  "code_object=%s "
                  "report_generation=%llu generation=%llu "
                  "epoch=%u first_owner=%u second_owner=%u first_inst=0x%x "
                  "second_inst=0x%x first_lds_known=%s first_lds=[%u,%u) "
                  "second_lds=[%u,%u) first_kind=%u second_kind=%u "
                  "first_lane_mask=0x%llx second_lane_mask=0x%llx",
                  static_cast<unsigned long long>(input.reader), index, diagnostic.kind,
                  input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
                  static_cast<unsigned long long>(record_replay_analysis.replay_header.generation),
                  static_cast<unsigned long long>(diagnostic.generation), diagnostic.epoch,
                  diagnostic.first_owner_id, diagnostic.second_owner_id,
                  diagnostic.first_instruction_offset, diagnostic.second_instruction_offset,
                  diagnostic.first_lds_byte_count != 0 ? "true" : "false",
                  diagnostic.first_lds_byte_offset,
                  diagnostic.first_lds_byte_offset + diagnostic.first_lds_byte_count,
                  diagnostic.second_lds_byte_offset,
                  diagnostic.second_lds_byte_offset + diagnostic.second_lds_byte_count,
                  diagnostic.first_access_kind, diagnostic.second_access_kind,
                  static_cast<unsigned long long>(diagnostic.first_lane_mask),
                  static_cast<unsigned long long>(diagnostic.second_lane_mask));
    }
  }
  uint32_t sampled_records = 0;
  for (uint32_t i = 0; i < visible_records && sampled_records < 4u; ++i) {
    const rocjitsu::ConSanMoiAccessRecord &record = records[i];
    if (record.claim_token == 0 &&
        record.access_kind == static_cast<uint32_t>(rocjitsu::ConSanMoiShadowAccessKind::Empty)) {
      continue;
    }
    log_message(
        kLogInfo,
        "ConSan MOI auto record reader=%llu index=%u event_index=%u kind=%u wave=%u "
        "claim_token=0x%016llx generation=%llu epoch=%u workgroup=(%u,%u,%u) "
        "inst=0x%x lds_offset=%u "
        "lds_bytes=%u "
        "cells=[%u,%u) "
        "lane_mask=0x%llx",
        static_cast<unsigned long long>(input.reader), i, record.event_index, record.access_kind,
        record.wave_id, static_cast<unsigned long long>(record.claim_token),
        static_cast<unsigned long long>(record.generation), record.epoch, record.workgroup_x,
        record.workgroup_y, record.workgroup_z, record.instruction_offset, record.lds_byte_offset,
        record.lds_byte_count, record.start_cell, record.start_cell + record.cell_count,
        static_cast<unsigned long long>(record.lane_mask));
    ++sampled_records;
  }

  const uint32_t barrier_sample_count = consan_moi_auto_detail_log_count(visible_barriers);
  for (uint32_t i = 0; i < barrier_sample_count; ++i) {
    const rocjitsu::ConSanMoiBarrierRecord &record = barriers[i];
    log_message(kLogInfo,
                "ConSan MOI auto barrier reader=%llu index=%u event_index=%u wave=%u "
                "inst=0x%x lane_mask=0x%llx",
                static_cast<unsigned long long>(input.reader), i, record.event_index,
                record.wave_id, record.instruction_offset,
                static_cast<unsigned long long>(record.lane_mask));
  }

  const uint32_t atomic_sample_count = consan_moi_auto_detail_log_count(visible_atomics);
  for (uint32_t i = 0; i < atomic_sample_count; ++i) {
    const rocjitsu::ConSanMoiAtomicRecord &record = atomics[i];
    log_message(kLogInfo,
                "ConSan MOI auto atomic reader=%llu index=%u event_index=%u kind=%u owner=%u "
                "generation=%llu epoch=%u workgroup=(%u,%u,%u) inst=0x%x address=0x%llx "
                "scope=%u semantics=%u operation=%u outcome=%u lane_mask=0x%llx "
                "success_lane_mask=0x%llx",
                static_cast<unsigned long long>(input.reader), i, record.event_index,
                static_cast<uint32_t>(record.kind), record.owner_id,
                static_cast<unsigned long long>(record.generation), record.epoch,
                record.workgroup_x, record.workgroup_y, record.workgroup_z,
                record.instruction_offset, static_cast<unsigned long long>(record.atomic_address),
                record.scope, record.semantics, static_cast<uint32_t>(record.operation),
                static_cast<uint32_t>(record.outcome),
                static_cast<unsigned long long>(record.lane_mask),
                static_cast<unsigned long long>(record.success_lane_mask));
  }
  const uint32_t diagnostic_sample_count = consan_moi_auto_detail_log_count(visible_diagnostics);
  for (uint32_t i = 0; i < diagnostic_sample_count; ++i) {
    const rocjitsu::ConSanMoiDiagnosticRecord &record = diagnostics[i];
    const char *backend_key_label =
        record.backend == static_cast<uint32_t>(rocjitsu::ConSanMoiEngine::RecordReplay)
            ? "event_index"
        : record.backend == static_cast<uint32_t>(rocjitsu::ConSanMoiEngine::InlineShadow)
            ? "workgroup"
            : "reserved";
    log_message(
        kLogInfo,
        "ConSan MOI auto diagnostic reader=%llu index=%u backend=%u kind=%u "
        "generation=%llu %s=%u first_epoch=%u second_epoch=%u "
        "first_owner=%u second_owner=%u first_inst=0x%x second_inst=0x%x "
        "first_kind=%u second_kind=%u first_lanes=0x%llx second_lanes=0x%llx "
        "first_lds=[%u,%u) second_lds=[%u,%u)",
        static_cast<unsigned long long>(input.reader), i, record.backend, record.kind,
        static_cast<unsigned long long>(record.generation), backend_key_label, record.reserved,
        record.first_epoch, record.epoch, record.first_owner_id, record.second_owner_id,
        record.first_instruction_offset, record.second_instruction_offset, record.first_access_kind,
        record.second_access_kind, static_cast<unsigned long long>(record.first_lane_mask),
        static_cast<unsigned long long>(record.second_lane_mask), record.first_lds_byte_offset,
        record.first_lds_byte_offset + record.first_lds_byte_count, record.second_lds_byte_offset,
        record.second_lds_byte_offset + record.second_lds_byte_count);
  }
  for (uint32_t i = 0; i < std::min<size_t>(visible_exact_shadow.size(), 4u); ++i) {
    const AutoMoiExactShadowEvidence &shadow_entry = visible_exact_shadow[i];
    log_message(kLogInfo,
                "ConSan MOI auto exact-shadow reader=%llu index=%u kind=%u owner=%u epoch=%u "
                "generation=%u inst=0x%x bytes=[%u,%u) lane=%u dispatch=0x%llx version=%u",
                static_cast<unsigned long long>(input.reader), shadow_entry.index,
                static_cast<uint32_t>(shadow_entry.entry.kind), shadow_entry.entry.owner_id,
                shadow_entry.entry.epoch, shadow_entry.entry.generation,
                shadow_entry.entry.instruction_offset, shadow_entry.byte_provenance.byte_offset,
                shadow_entry.byte_provenance.byte_offset + shadow_entry.byte_provenance.byte_count,
                shadow_entry.byte_provenance.representative_lane,
                static_cast<unsigned long long>(shadow_entry.dispatch_id), shadow_entry.version);
  }
  constexpr size_t kSampledLogLimit = 64u;
  for (uint32_t i = 0; i < std::min(visible_sampled.size(), kSampledLogLimit); ++i) {
    const AutoMoiSampledEvidence &sampled_entry = visible_sampled[i];
    const AutoMoiSampledStaticMapping *mapping = sampled_entry.static_mapping;
    const uint64_t instruction_offset = mapping != nullptr ? mapping->instruction_offset : 0u;
    const uint32_t relative_slot =
        mapping != nullptr ? sampled_entry.index - mapping->first_slot : 0u;
    const uint32_t range = mapping != nullptr ? relative_slot / mapping->bank_count : 0u;
    const uint32_t bank = mapping != nullptr ? relative_slot % mapping->bank_count : 0u;
    const uint64_t emitted_probe_offset = mapping != nullptr ? mapping->emitted_probe_offset : 0u;
    const uint64_t relocated_guest_offset =
        mapping != nullptr ? mapping->relocated_guest_offset : 0u;
    const uint16_t scratch_vgpr =
        mapping != nullptr ? mapping->scratch_vgpr.value_or(std::numeric_limits<uint16_t>::max())
                           : std::numeric_limits<uint16_t>::max();
    log_message(kLogInfo,
                "ConSan MOI auto sampled reader=%llu index=%u kind=%u owner=%u epoch=%u "
                "generation=%u bytes=[%u,%u) consumed=%s dispatch=0x%llx "
                "workgroup=(%u,%u,%u) instruction=0x%llx trampoline=0x%llx "
                "relocated_guest=0x%llx scratch_vgpr=%u range=%u bank=%u mapped=%s "
                "sync_class=%u sync_kind=%u sync_role=%u sync_scope=%u sync_outcome=%u "
                "sync_address=0x%llx sync_bytes=%u sync_epochs=%u/%u",
                static_cast<unsigned long long>(input.reader), sampled_entry.index,
                static_cast<uint32_t>(sampled_entry.entry.kind), sampled_entry.entry.owner_id,
                sampled_entry.entry.epoch, sampled_entry.entry.generation,
                sampled_entry.entry.start_byte,
                sampled_entry.entry.start_byte + sampled_entry.entry.byte_count,
                sampled_entry.entry.consumed ? "true" : "false",
                static_cast<unsigned long long>(sampled_entry.dispatch_id),
                sampled_entry.workgroup_x, sampled_entry.workgroup_y, sampled_entry.workgroup_z,
                static_cast<unsigned long long>(instruction_offset),
                static_cast<unsigned long long>(emitted_probe_offset),
                static_cast<unsigned long long>(relocated_guest_offset), scratch_vgpr, range, bank,
                mapping != nullptr ? "true" : "false",
                static_cast<uint32_t>(sampled_entry.sync.classification),
                static_cast<uint32_t>(sampled_entry.sync.metadata.kind),
                static_cast<uint32_t>(sampled_entry.sync.metadata.role),
                static_cast<uint32_t>(sampled_entry.sync.metadata.scope),
                static_cast<uint32_t>(sampled_entry.sync.metadata.outcome),
                static_cast<unsigned long long>(sampled_entry.sync.metadata.address),
                sampled_entry.sync.metadata.byte_count, sampled_entry.sync.metadata.epoch_before,
                sampled_entry.sync.metadata.epoch_after);
  }
  if (visible_sampled.size() > kSampledLogLimit)
    log_message(kLogInfo, "ConSan MOI auto sampled reader=%llu omitted=%zu after log limit=%zu",
                static_cast<unsigned long long>(input.reader),
                visible_sampled.size() - kSampledLogLimit, kSampledLogLimit);
  if (first_sampled_conflict) {
    const AutoMoiSampledEvidence &first = first_sampled_conflict->first;
    const AutoMoiSampledEvidence &second = first_sampled_conflict->second;
    const uint32_t first_end = first.entry.start_byte + first.entry.byte_count;
    const uint32_t second_end = second.entry.start_byte + second.entry.byte_count;
    log_message(kLogInfo,
                "ConSan MOI auto sampled conflict reader=%llu first_index=%u second_index=%u "
                "first_kind=%u second_kind=%u first_owner=%u second_owner=%u epoch=%u "
                "generation=%u first_bytes=[%u,%u) second_bytes=[%u,%u)",
                static_cast<unsigned long long>(input.reader), first.index, second.index,
                static_cast<uint32_t>(first.entry.kind), static_cast<uint32_t>(second.entry.kind),
                first.entry.owner_id, second.entry.owner_id, second.entry.epoch,
                second.entry.generation, first.entry.start_byte, first_end, second.entry.start_byte,
                second_end);
  }
  return summary;
}

} // namespace rocjitsu::consan_hook

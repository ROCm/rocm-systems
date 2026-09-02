// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_inline_shadow_report_decoder.h"

#include <utility>

namespace rocjitsu::consan_hook {

AutoMoiInlineShadowDecodeResult decode_auto_moi_inline_shadow_report(
    const AutoMoiReportPipelineInput &input, const ConSanMoiReportHeader &header,
    std::span<const uint8_t> report_bytes, uint32_t raw_visible_diagnostic_count,
    AutoMoiReportSummary &summary) {
  AutoMoiInlineShadowDecodeResult result;
  AutoMoiInlineShadowDecodedReport &decoded = result.decoded;
  const ConSanMoiReportBufferLayout &layout = input.layout;
  const uint8_t *bytes = report_bytes.data();

  const auto *exact_shadow = reinterpret_cast<const ConSanMoiInlineExactShadowSlot *>(
      bytes + layout.exact_shadow_entries_offset);
  for (uint32_t i = 0; i < header.exact_shadow_entry_capacity; ++i) {
    const volatile auto &slot = exact_shadow[i];
    const uint32_t version_before = slot.version;
    const uint64_t packed_access = slot.packed_access;
    const uint64_t dispatch_id = slot.dispatch_id;
    const uint32_t byte_provenance = slot.byte_provenance;
    const uint32_t version_after = slot.version;
    const auto snapshot = classify_consan_moi_inline_exact_snapshot(
        {version_before, packed_access, dispatch_id, byte_provenance, version_after});
    switch (snapshot.state) {
    case ConSanMoiInlineExactSnapshotState::Empty:
      break;
    case ConSanMoiInlineExactSnapshotState::Stable:
      decoded.exact_shadow.push_back(
          {i, snapshot.entry, snapshot.byte_provenance, snapshot.dispatch_id, snapshot.version});
      break;
    case ConSanMoiInlineExactSnapshotState::Publishing:
      ++summary.exact_incomplete_snapshot_count;
      break;
    case ConSanMoiInlineExactSnapshotState::ChangedDuringRead:
      ++summary.exact_changed_snapshot_count;
      break;
    case ConSanMoiInlineExactSnapshotState::Malformed:
      decoded.issues.push_back(
          {.reason = AutoMoiInlineShadowEvidenceReason::ExactMalformed,
           .index = i,
           .words = {version_before, packed_access, dispatch_id, byte_provenance, version_after}});
      ++summary.exact_malformed_snapshot_count;
      break;
    }
  }

  const auto *atomic_releases = reinterpret_cast<const ConSanMoiInlineAtomicReleaseSlot *>(
      bytes + layout.inline_atomic_release_slots_offset);
  const auto *causal_snapshots = reinterpret_cast<const ConSanMoiInlineCausalSnapshot *>(
      bytes + layout.inline_causal_snapshots_offset);
  for (uint32_t i = 0; i < layout.inline_atomic_release_capacity; ++i) {
    const volatile auto &slot = atomic_releases[i];
    const volatile auto &source_snapshot = causal_snapshots[i];
    ConSanMoiInlineReleaseSnapshotWords words;
    words.version_before = slot.version;
    words.slot.version = words.version_before;
    words.slot.owner_id = slot.owner_id;
    words.slot.epoch_plus_one = slot.epoch_plus_one;
    words.slot.workgroup_key = slot.workgroup_key;
    words.slot.atomic_address = slot.atomic_address;
    words.slot.dispatch_id = slot.dispatch_id;
    words.snapshot.entry_count = source_snapshot.entry_count;
    words.snapshot.flags = source_snapshot.flags;
    const auto *source_entries =
        reinterpret_cast<const volatile ConSanMoiInlineCausalSnapshotEntry *>(
            &source_snapshot.entries);
    for (uint32_t entry_index = 0; entry_index < kConSanMoiInlineCausalSnapshotEntryCapacity;
         ++entry_index) {
      words.snapshot.entries[entry_index].ancestor_owner_id =
          source_entries[entry_index].ancestor_owner_id;
      words.snapshot.entries[entry_index].ancestor_epoch_plus_one =
          source_entries[entry_index].ancestor_epoch_plus_one;
    }
    words.version_after = slot.version;
    const auto classified = classify_consan_moi_inline_release_snapshot(words);
    switch (classified.state) {
    case ConSanMoiInlineReleaseSnapshotState::Empty:
      break;
    case ConSanMoiInlineReleaseSnapshotState::Stable:
      decoded.atomic_releases.push_back({i, words.slot, words.snapshot});
      break;
    case ConSanMoiInlineReleaseSnapshotState::Publishing:
      ++summary.release_incomplete_snapshot_count;
      decoded.issues.push_back(
          {.reason = AutoMoiInlineShadowEvidenceReason::ReleasePublishing,
           .index = i,
           .words = {words.slot.version, words.slot.owner_id, words.slot.epoch_plus_one,
                     words.slot.workgroup_key, words.slot.atomic_address, words.slot.dispatch_id}});
      break;
    case ConSanMoiInlineReleaseSnapshotState::ChangedDuringRead:
      ++summary.release_changed_snapshot_count;
      break;
    case ConSanMoiInlineReleaseSnapshotState::CapacityOverflow:
      ++summary.release_overflow_snapshot_count;
      break;
    case ConSanMoiInlineReleaseSnapshotState::SourceIncomplete:
      ++summary.release_source_incomplete_snapshot_count;
      break;
    case ConSanMoiInlineReleaseSnapshotState::Malformed:
      ++summary.release_malformed_snapshot_count;
      break;
    }
  }

  std::vector<ConSanMoiInlineAcquiredEpochTokenSlot> stable_acquired_tokens;
  const auto *acquired_tokens =
      reinterpret_cast<const volatile ConSanMoiInlineAcquiredEpochTokenSlot *>(
          bytes + layout.inline_acquired_epoch_token_slots_offset);
  for (uint32_t i = 0; i < layout.inline_acquired_epoch_token_capacity; ++i) {
    const volatile auto &slot = acquired_tokens[i];
    ConSanMoiInlineAcquiredTokenSnapshot snapshot;
    snapshot.version_before = slot.version;
    snapshot.payload.version = snapshot.version_before;
    snapshot.payload.consumer_owner_id = slot.consumer_owner_id;
    snapshot.payload.producer_owner_id = slot.producer_owner_id;
    snapshot.payload.producer_epoch_plus_one = slot.producer_epoch_plus_one;
    snapshot.payload.workgroup_key = slot.workgroup_key;
    snapshot.payload.kind = slot.kind;
    snapshot.payload.dispatch_id = slot.dispatch_id;
    snapshot.payload.source_release_address = slot.source_release_address;
    snapshot.payload.source_release_version = slot.source_release_version;
    snapshot.payload.consumer_epoch_plus_one = slot.consumer_epoch_plus_one;
    snapshot.payload.reservation_version = slot.reservation_version;
    snapshot.version_after = slot.version;
    const auto classified = consan_moi_inline_classify_acquired_token(snapshot);
    switch (classified.state) {
    case ConSanMoiInlineAcquiredTokenState::Empty:
      break;
    case ConSanMoiInlineAcquiredTokenState::Stable:
      decoded.acquired_tokens.push_back({i, classified.token});
      stable_acquired_tokens.push_back(classified.token);
      break;
    case ConSanMoiInlineAcquiredTokenState::Publishing:
      ++summary.token_incomplete_snapshot_count;
      break;
    case ConSanMoiInlineAcquiredTokenState::Changed:
      ++summary.token_changed_snapshot_count;
      break;
    case ConSanMoiInlineAcquiredTokenState::Malformed:
      ++summary.token_malformed_snapshot_count;
      break;
    }
  }

  const auto *raw_diagnostics =
      reinterpret_cast<const ConSanMoiDiagnosticRecord *>(bytes + layout.diagnostic_records_offset);
  const bool deferred_token_evidence_complete =
      summary.token_incomplete_snapshot_count == 0 && summary.token_changed_snapshot_count == 0 &&
      summary.token_malformed_snapshot_count == 0 && header.inline_overflow_count == 0 &&
      header.inline_malformed_count == 0;
  const auto deferred_filter = consan_moi_filter_deferred_inline_diagnostics(
      std::span<const ConSanMoiDiagnosticRecord>(raw_diagnostics, raw_visible_diagnostic_count),
      header.diagnostic_count, stable_acquired_tokens, deferred_token_evidence_complete);
  decoded.deferred_token_qualified_diagnostic_count = deferred_filter.qualified_count;
  result.diagnostics.reserve(deferred_filter.visible_indices.size());
  for (uint32_t index : deferred_filter.visible_indices)
    result.diagnostics.push_back(raw_diagnostics[index]);

  const auto *inline_metadata =
      input.static_metadata ? std::get_if<AutoMoiInlineCompactStaticMetadata>(input.static_metadata)
                            : nullptr;
  const uint32_t compact_token_mapping_count =
      inline_metadata ? inline_metadata->mapping_count : 0u;
  if (!result.diagnostics.empty() && compact_token_mapping_count != 0u) {
    const auto *mappings = reinterpret_cast<const ConSanMoiCompactDiagnosticTokenMapping *>(
        bytes + layout.inline_compact_token_mappings_offset);
    for (ConSanMoiDiagnosticRecord &diagnostic : result.diagnostics) {
      const uint32_t tagged = diagnostic.first_instruction_offset;
      constexpr uint32_t kTokenPayloadMask = consan_moi_exact_shadow::max_compact_token;
      constexpr uint32_t kAllowedBits =
          consan_moi_exact_shadow::compact_diagnostic_token_tag | kTokenPayloadMask;
      if ((tagged & consan_moi_exact_shadow::compact_diagnostic_token_tag) == 0u)
        continue;
      const uint16_t token = static_cast<uint16_t>(tagged & kTokenPayloadMask);
      const bool well_formed = token != 0u && (tagged & ~kAllowedBits) == 0u;
      const ConSanMoiCompactDiagnosticTokenMapping *current = nullptr;
      const ConSanMoiCompactDiagnosticTokenMapping *resolved = nullptr;
      bool current_ambiguous = false;
      bool prior_ambiguous = false;
      if (well_formed) {
        for (uint32_t index = 0; index < compact_token_mapping_count; ++index) {
          const auto &mapping = mappings[index];
          if (mapping.instruction_offset != diagnostic.second_instruction_offset)
            continue;
          if (current != nullptr &&
              current->owner_descriptor_file_offset != mapping.owner_descriptor_file_offset) {
            current_ambiguous = true;
            break;
          }
          current = &mapping;
        }
        if (current != nullptr && !current_ambiguous) {
          for (uint32_t index = 0; index < compact_token_mapping_count; ++index) {
            const auto &mapping = mappings[index];
            if (mapping.owner_descriptor_file_offset != current->owner_descriptor_file_offset ||
                mapping.token != token)
              continue;
            if (resolved != nullptr && resolved->instruction_offset != mapping.instruction_offset) {
              prior_ambiguous = true;
              break;
            }
            resolved = &mapping;
          }
        }
      }
      if (!well_formed || current == nullptr || current_ambiguous || resolved == nullptr ||
          prior_ambiguous) {
        ++summary.inline_malformed_count;
        decoded.issues.push_back(
            {.reason = AutoMoiInlineShadowEvidenceReason::CompactDiagnosticTokenUnresolved,
             .index = diagnostic.second_instruction_offset,
             .words = {tagged, well_formed, current_ambiguous, prior_ambiguous}});
        continue;
      }
      diagnostic.first_instruction_offset = resolved->instruction_offset;
    }
  }

  summary.visible_inline_publication_count = header.event_counter;
  summary.visible_exact_shadow_entry_count = decoded.exact_shadow.size();
  summary.visible_inline_atomic_release_count = decoded.atomic_releases.size();
  summary.visible_inline_acquired_token_count = decoded.acquired_tokens.size();
  summary.inline_undercoverage_count = header.inline_undercoverage_count;
  summary.inline_overflow_count = header.inline_overflow_count;
  summary.inline_unsupported_count = header.inline_unsupported_count;
  summary.inline_malformed_count += header.inline_malformed_count;
  return result;
}

} // namespace rocjitsu::consan_hook

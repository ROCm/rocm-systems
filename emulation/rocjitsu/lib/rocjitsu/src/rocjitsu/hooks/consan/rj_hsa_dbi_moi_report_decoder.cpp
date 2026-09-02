// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_decoder.h"

#include <algorithm>
#include <array>
#include <limits>
#include <ranges>
#include <utility>

namespace rocjitsu::consan_hook {

AutoMoiDecodedReport decode_auto_moi_report(const AutoMoiReportPipelineInput &input,
                                            const AutoMoiReportSnapshot &snapshot,
                                            AutoMoiReportSummary initial_summary) {
  AutoMoiDecodedReport result;
  result.summary = initial_summary;
  AutoMoiReportSummary &summary = result.summary;
  if (snapshot.bytes.size() < sizeof(ConSanMoiReportHeader) || snapshot.bytes.size() < input.size) {
    result.failure = AutoMoiReportDecodeFailure::SnapshotTooSmall;
    return result;
  }
  const ConSanMoiEngine expected_engine = input.layout.engine;
  const void *report_ptr = snapshot.bytes.data();

  const auto *header = static_cast<const rocjitsu::ConSanMoiReportHeader *>(report_ptr);
  if (!rocjitsu::consan_moi_report_header_is_current(*header)) {
    result.header = *header;
    result.failure = AutoMoiReportDecodeFailure::InvalidHeader;
    return result;
  }
  const rocjitsu::ConSanMoiReportBufferLayout &expected_layout = input.layout;
  if (!rocjitsu::consan_moi_report_layout_matches_header(*header, expected_layout, expected_engine,
                                                         input.size)) {
    result.header = *header;
    result.failure = AutoMoiReportDecodeFailure::LayoutMismatch;
    return result;
  }
  const uint32_t access_record_count = header->access_record_count;
  const uint32_t barrier_record_count = header->barrier_record_count;
  const uint32_t atomic_record_count = header->atomic_record_count;
  const uint32_t visible_records = std::min(access_record_count, header->access_record_capacity);
  const uint32_t visible_barriers = std::min(barrier_record_count, header->barrier_record_capacity);
  const uint32_t visible_atomics = std::min(atomic_record_count, header->atomic_record_capacity);
  const uint32_t visible_fences =
      std::min(header->fence_record_count, expected_layout.fence_record_capacity);
  const uint32_t raw_visible_diagnostics =
      std::min(header->diagnostic_count, header->diagnostic_capacity);
  const uint32_t dropped_records =
      access_record_count > visible_records ? access_record_count - visible_records : 0;
  const uint32_t dropped_barriers =
      barrier_record_count > visible_barriers ? barrier_record_count - visible_barriers : 0;
  const uint32_t dropped_atomics =
      atomic_record_count > visible_atomics ? atomic_record_count - visible_atomics : 0;
  const uint32_t dropped_fences =
      expected_layout.fence_record_capacity != 0 && header->fence_record_count > visible_fences
          ? header->fence_record_count - visible_fences
          : 0;
  const uint32_t dropped_diagnostics = header->diagnostic_count > raw_visible_diagnostics
                                           ? header->diagnostic_count - raw_visible_diagnostics
                                           : 0;
  const auto *bytes = static_cast<const uint8_t *>(report_ptr);
  const auto *exact_shadow = reinterpret_cast<const rocjitsu::ConSanMoiInlineExactShadowSlot *>(
      bytes + expected_layout.exact_shadow_entries_offset);
  const auto *inline_atomic_releases =
      reinterpret_cast<const rocjitsu::ConSanMoiInlineAtomicReleaseSlot *>(
          bytes + expected_layout.inline_atomic_release_slots_offset);
  const uint32_t inline_atomic_release_capacity = expected_layout.inline_atomic_release_capacity;
  const auto *inline_acquired_tokens =
      reinterpret_cast<const volatile rocjitsu::ConSanMoiInlineAcquiredEpochTokenSlot *>(
          bytes + expected_layout.inline_acquired_epoch_token_slots_offset);
  const uint32_t inline_acquired_token_capacity =
      expected_layout.inline_acquired_epoch_token_capacity;
  const auto *inline_causal_snapshots =
      reinterpret_cast<const rocjitsu::ConSanMoiInlineCausalSnapshot *>(
          bytes + expected_layout.inline_causal_snapshots_offset);
  const auto *sampled_causal_windows =
      reinterpret_cast<const rocjitsu::ConSanMoiSampledCausalWindow *>(
          bytes + expected_layout.sampled_causal_windows_offset);
  const uint32_t sampled_watchpoint_capacity = expected_layout.sampled_watchpoint_capacity;
  const uint32_t sampled_causal_window_capacity = expected_layout.sampled_causal_window_capacity;
  const auto *sampled =
      reinterpret_cast<const uint64_t *>(bytes + expected_layout.sampled_watchpoints_offset);
  const auto *sampled_sync_words = reinterpret_cast<const volatile uint32_t *>(
      bytes + expected_layout.sampled_sync_metadata_offset);
  const uint32_t sampled_sync_metadata_capacity = expected_layout.sampled_sync_metadata_capacity;
  const auto *sampled_pending_acquires =
      reinterpret_cast<const volatile rocjitsu::ConSanMoiSampledPendingAcquireSlot *>(
          bytes + expected_layout.sampled_pending_acquires_offset);
  const uint32_t sampled_pending_acquire_capacity =
      expected_layout.sampled_pending_acquire_capacity;
  const uint32_t sampled_pending_acquire_owner_bank_count =
      rocjitsu::consan_moi_sampled_pending_acquire_owner_bank_count(
          sampled_pending_acquire_capacity, sampled_causal_window_capacity);
  const auto read_sampled_pending_acquire = [&](uint32_t pending_index) {
    rocjitsu::ConSanMoiSampledPendingAcquireView view{};
    if (pending_index >= sampled_pending_acquire_capacity)
      return view;
    const volatile auto &pending = sampled_pending_acquires[pending_index];
    view.version_before = pending.version;
    view.slot.version = view.version_before;
    view.slot.selected_slot = pending.selected_slot;
    view.slot.generation = pending.generation;
    view.slot.dispatch_id = pending.dispatch_id;
    view.slot.workgroup_x = pending.workgroup_x;
    view.slot.workgroup_y = pending.workgroup_y;
    view.slot.workgroup_z = pending.workgroup_z;
    view.slot.owner_id = pending.owner_id;
    view.slot.source_epoch = pending.source_epoch;
    view.slot.reserved = pending.reserved;
    view.slot.metadata.address = pending.metadata.address;
    view.slot.metadata.byte_count = pending.metadata.byte_count;
    view.slot.metadata.descriptor = pending.metadata.descriptor;
    view.slot.metadata.epoch_before = pending.metadata.epoch_before;
    view.slot.metadata.epoch_after = pending.metadata.epoch_after;
    view.version_after = pending.version;
    return view;
  };
  using ExactShadowEntry = AutoMoiExactShadowEvidence;
  std::vector<ExactShadowEntry> visible_exact_shadow;
  for (uint32_t i = 0; i < header->exact_shadow_entry_capacity; ++i) {
    const volatile auto &slot = exact_shadow[i];
    const uint32_t version_before = slot.version;
    const uint64_t packed_access = slot.packed_access;
    const uint64_t dispatch_id = slot.dispatch_id;
    const uint32_t byte_provenance = slot.byte_provenance;
    const uint32_t version_after = slot.version;
    const auto snapshot = rocjitsu::classify_consan_moi_inline_exact_snapshot(
        {version_before, packed_access, dispatch_id, byte_provenance, version_after});
    switch (snapshot.state) {
    case rocjitsu::ConSanMoiInlineExactSnapshotState::Empty:
      break;
    case rocjitsu::ConSanMoiInlineExactSnapshotState::Stable:
      visible_exact_shadow.push_back(
          {i, snapshot.entry, snapshot.byte_provenance, snapshot.dispatch_id, snapshot.version});
      break;
    case rocjitsu::ConSanMoiInlineExactSnapshotState::Publishing:
      ++summary.exact_incomplete_snapshot_count;
      break;
    case rocjitsu::ConSanMoiInlineExactSnapshotState::ChangedDuringRead:
      ++summary.exact_changed_snapshot_count;
      break;
    case rocjitsu::ConSanMoiInlineExactSnapshotState::Malformed:
      result.issues.push_back(
          {.reason = AutoMoiReportEvidenceReason::ExactMalformed,
           .index = i,
           .words = {version_before, packed_access, dispatch_id, byte_provenance, version_after}});
      ++summary.exact_malformed_snapshot_count;
      break;
    }
  }
  using InlineAtomicReleaseEntry = AutoMoiInlineAtomicReleaseEvidence;
  std::vector<InlineAtomicReleaseEntry> visible_inline_atomic_releases;
  for (uint32_t i = 0; i < inline_atomic_release_capacity; ++i) {
    const volatile auto &slot = inline_atomic_releases[i];
    const volatile auto &source_snapshot = inline_causal_snapshots[i];
    rocjitsu::ConSanMoiInlineReleaseSnapshotWords words;
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
        reinterpret_cast<const volatile rocjitsu::ConSanMoiInlineCausalSnapshotEntry *>(
            &source_snapshot.entries);
    for (uint32_t entry_index = 0;
         entry_index < rocjitsu::kConSanMoiInlineCausalSnapshotEntryCapacity; ++entry_index) {
      words.snapshot.entries[entry_index].ancestor_owner_id =
          source_entries[entry_index].ancestor_owner_id;
      words.snapshot.entries[entry_index].ancestor_epoch_plus_one =
          source_entries[entry_index].ancestor_epoch_plus_one;
    }
    words.version_after = slot.version;
    const auto classified = rocjitsu::classify_consan_moi_inline_release_snapshot(words);
    switch (classified.state) {
    case rocjitsu::ConSanMoiInlineReleaseSnapshotState::Empty:
      break;
    case rocjitsu::ConSanMoiInlineReleaseSnapshotState::Stable:
      visible_inline_atomic_releases.push_back({i, words.slot, words.snapshot});
      break;
    case rocjitsu::ConSanMoiInlineReleaseSnapshotState::Publishing:
      ++summary.release_incomplete_snapshot_count;
      result.issues.push_back(
          {.reason = AutoMoiReportEvidenceReason::ReleasePublishing,
           .index = i,
           .words = {words.slot.version, words.slot.owner_id, words.slot.epoch_plus_one,
                     words.slot.workgroup_key, words.slot.atomic_address, words.slot.dispatch_id}});
      break;
    case rocjitsu::ConSanMoiInlineReleaseSnapshotState::ChangedDuringRead:
      ++summary.release_changed_snapshot_count;
      break;
    case rocjitsu::ConSanMoiInlineReleaseSnapshotState::CapacityOverflow:
      ++summary.release_overflow_snapshot_count;
      break;
    case rocjitsu::ConSanMoiInlineReleaseSnapshotState::SourceIncomplete:
      ++summary.release_source_incomplete_snapshot_count;
      break;
    case rocjitsu::ConSanMoiInlineReleaseSnapshotState::Malformed:
      ++summary.release_malformed_snapshot_count;
      break;
    }
  }
  using InlineAcquiredTokenEntry = AutoMoiInlineAcquiredTokenEvidence;
  std::vector<InlineAcquiredTokenEntry> visible_inline_acquired_tokens;
  std::vector<rocjitsu::ConSanMoiInlineAcquiredEpochTokenSlot> stable_inline_acquired_tokens;
  for (uint32_t i = 0; i < inline_acquired_token_capacity; ++i) {
    const volatile auto &slot = inline_acquired_tokens[i];
    rocjitsu::ConSanMoiInlineAcquiredTokenSnapshot snapshot;
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
    const auto classified = rocjitsu::consan_moi_inline_classify_acquired_token(snapshot);
    switch (classified.state) {
    case rocjitsu::ConSanMoiInlineAcquiredTokenState::Empty:
      break;
    case rocjitsu::ConSanMoiInlineAcquiredTokenState::Stable:
      visible_inline_acquired_tokens.push_back({i, classified.token});
      stable_inline_acquired_tokens.push_back(classified.token);
      break;
    case rocjitsu::ConSanMoiInlineAcquiredTokenState::Publishing:
      ++summary.token_incomplete_snapshot_count;
      break;
    case rocjitsu::ConSanMoiInlineAcquiredTokenState::Changed:
      ++summary.token_changed_snapshot_count;
      break;
    case rocjitsu::ConSanMoiInlineAcquiredTokenState::Malformed:
      ++summary.token_malformed_snapshot_count;
      break;
    }
  }
  const auto *raw_diagnostics = reinterpret_cast<const rocjitsu::ConSanMoiDiagnosticRecord *>(
      bytes + expected_layout.diagnostic_records_offset);
  const bool deferred_token_evidence_complete =
      expected_engine == ConSanMoiEngine::InlineShadow &&
      summary.token_incomplete_snapshot_count == 0 && summary.token_changed_snapshot_count == 0 &&
      summary.token_malformed_snapshot_count == 0 && header->inline_overflow_count == 0 &&
      header->inline_malformed_count == 0;
  const auto deferred_filter = rocjitsu::consan_moi_filter_deferred_inline_diagnostics(
      std::span<const rocjitsu::ConSanMoiDiagnosticRecord>(raw_diagnostics,
                                                           raw_visible_diagnostics),
      header->diagnostic_count, stable_inline_acquired_tokens, deferred_token_evidence_complete);
  const auto &visible_diagnostic_indices = deferred_filter.visible_indices;
  const uint32_t deferred_token_qualified_diagnostics = deferred_filter.qualified_count;
  const uint32_t visible_diagnostics = static_cast<uint32_t>(visible_diagnostic_indices.size());
  const auto *sampled_metadata =
      input.static_metadata ? std::get_if<AutoMoiSampledStaticMetadata>(input.static_metadata)
                            : nullptr;
  const std::span<const AutoMoiSampledStaticMapping> sampled_static_mappings =
      sampled_metadata ? sampled_metadata->mappings
                       : std::span<const AutoMoiSampledStaticMapping>{};
  std::vector<AutoMoiSampledEvidence> visible_sampled;
  const auto sampled_static_mapping_for_slot = [&](uint32_t slot) {
    const auto mapping = std::ranges::find_if(
        sampled_static_mappings, [&](const AutoMoiSampledStaticMapping &candidate) {
          if (slot < candidate.first_slot)
            return false;
          const uint64_t relative = slot - candidate.first_slot;
          return relative < static_cast<uint64_t>(candidate.range_count) * candidate.bank_count;
        });
    return mapping == sampled_static_mappings.end() ? nullptr : &*mapping;
  };
  uint32_t visible_sampled_sync_metadata = 0;
  uint64_t sampled_watchpoint_slots_examined = 0;
  uint64_t sampled_pending_release_slots_examined = 0;
  const auto *sampled_words = reinterpret_cast<const volatile uint32_t *>(sampled);
  const uint32_t active_sampled_generation =
      static_cast<uint32_t>(header->generation) &
      rocjitsu::consan_moi_sampled_watchpoint::max_generation;
  const uint32_t sampled_ready_window_target =
      std::min(header->sampled_causal_window_count, sampled_watchpoint_capacity);
  uint32_t sampled_ready_windows_examined = 0;
  for (uint32_t i = 0; i < sampled_watchpoint_capacity &&
                       sampled_ready_windows_examined < sampled_ready_window_target;
       ++i) {
    ++sampled_watchpoint_slots_examined;
    if (i >= sampled_causal_window_capacity) {
      ++summary.sampled_malformed_snapshot_count;
      continue;
    }
    const volatile auto &window = sampled_causal_windows[i];
    const uint32_t state_before = window.publication_state;
    const uint32_t low_before = sampled_words[2u * i];
    const uint32_t high = sampled_words[2u * i + 1u];
    const uint32_t low_after = sampled_words[2u * i];
    const uint64_t window_generation = window.generation;
    const uint64_t window_dispatch_id = window.dispatch_id;
    const uint32_t window_x = window.workgroup_x;
    const uint32_t window_y = window.workgroup_y;
    const uint32_t window_z = window.workgroup_z;
    const uint32_t window_epoch = window.epoch;
    const uint32_t window_first_entry = window.first_entry;
    const uint32_t window_entry_count = window.entry_count;
    const uint32_t window_cluster_workgroup_id = window.cluster_workgroup_id;
    rocjitsu::ConSanMoiSampledSyncMetadataPacked sync_packed{};
    uint32_t sync_descriptor_before = 0;
    uint32_t sync_descriptor_after = 0;
    if (i < sampled_sync_metadata_capacity) {
      const size_t sync_word = static_cast<size_t>(i) * (sizeof(sync_packed) / sizeof(uint32_t));
      sync_descriptor_before = sampled_sync_words[sync_word + 3u];
      sync_packed.address = static_cast<uint64_t>(sampled_sync_words[sync_word]) |
                            (static_cast<uint64_t>(sampled_sync_words[sync_word + 1u]) << 32u);
      sync_packed.byte_count = sampled_sync_words[sync_word + 2u];
      sync_packed.epoch_before = sampled_sync_words[sync_word + 4u];
      sync_packed.epoch_after = sampled_sync_words[sync_word + 5u];
      sync_descriptor_after = sampled_sync_words[sync_word + 3u];
    }
    rocjitsu::ConSanMoiSampledSyncDecodeResult sync =
        i < sampled_sync_metadata_capacity
            ? rocjitsu::classify_consan_moi_sampled_sync_snapshot(
                  {sync_descriptor_before, sync_packed, sync_descriptor_after}, window_epoch)
            : rocjitsu::ConSanMoiSampledSyncDecodeResult{};
    const uint32_t state_after = window.publication_state;
    const rocjitsu::ConSanMoiSampledSnapshot snapshot =
        rocjitsu::classify_consan_moi_sampled_snapshot({low_before, high, low_after},
                                                       active_sampled_generation);
    if (state_before != state_after) {
      ++summary.sampled_changed_snapshot_count;
      continue;
    }
    const auto publication_state =
        static_cast<rocjitsu::ConSanMoiSampledCausalPublicationState>(state_after);
    if (publication_state == rocjitsu::ConSanMoiSampledCausalPublicationState::Empty) {
      if (snapshot.state != rocjitsu::ConSanMoiSampledSnapshotState::Empty ||
          sync.classification != rocjitsu::ConSanMoiSampledSyncClassification::Empty)
        ++summary.sampled_malformed_snapshot_count;
      continue;
    }
    if (publication_state == rocjitsu::ConSanMoiSampledCausalPublicationState::Publishing) {
      ++summary.sampled_incomplete_snapshot_count;
      continue;
    }
    if (publication_state == rocjitsu::ConSanMoiSampledCausalPublicationState::Ready)
      ++sampled_ready_windows_examined;
    if (publication_state != rocjitsu::ConSanMoiSampledCausalPublicationState::Ready ||
        window_generation != header->generation ||
        window_epoch > rocjitsu::consan_moi_sampled_watchpoint::max_epoch ||
        window_first_entry != i || window_entry_count != 1) {
      result.issues.push_back(
          {.reason = AutoMoiReportEvidenceReason::SampledMalformedWindow,
           .index = i,
           .words = {state_after, window_generation, header->generation, window_dispatch_id,
                     header->dispatch_id, window_epoch, window_first_entry, window_entry_count,
                     window_cluster_workgroup_id, static_cast<uint32_t>(snapshot.state)}});
      ++summary.sampled_malformed_snapshot_count;
      continue;
    }
    bool sync_snapshot_usable = true;
    if (sync.classification == rocjitsu::ConSanMoiSampledSyncClassification::ChangedDuringRead) {
      ++summary.sampled_changed_snapshot_count;
      sync_snapshot_usable = false;
    }
    if (sync.classification == rocjitsu::ConSanMoiSampledSyncClassification::Publishing) {
      ++summary.sampled_incomplete_snapshot_count;
      sync_snapshot_usable = false;
    }
    const bool has_watchpoint = snapshot.state != rocjitsu::ConSanMoiSampledSnapshotState::Empty;
    if (!has_watchpoint) {
      result.issues.push_back({.reason = AutoMoiReportEvidenceReason::SampledEmptyWatchpoint,
                               .index = i,
                               .words = {static_cast<uint32_t>(snapshot.state)}});
      ++summary.sampled_malformed_snapshot_count;
      continue;
    }
    const bool has_sync =
        sync.classification != rocjitsu::ConSanMoiSampledSyncClassification::Empty;
    if (has_sync && sync_snapshot_usable) {
      if (sync.classification != rocjitsu::ConSanMoiSampledSyncClassification::Valid) {
        ++summary.sampled_malformed_sync_count;
        sync_snapshot_usable = false;
      }
    }
    switch (snapshot.state) {
    case rocjitsu::ConSanMoiSampledSnapshotState::Empty:
      break;
    case rocjitsu::ConSanMoiSampledSnapshotState::Stable:
      if (snapshot.entry.epoch != window_epoch) {
        ++summary.sampled_malformed_snapshot_count;
        result.issues.push_back({.reason = AutoMoiReportEvidenceReason::SampledMalformedWatchpoint,
                                 .index = i,
                                 .words = {low_after, high, window_epoch, snapshot.entry.epoch}});
        break;
      }
      visible_sampled.push_back(
          {i, snapshot.entry, sync,
           static_cast<uint64_t>(low_after) | (static_cast<uint64_t>(high) << 32u),
           window_generation, window_dispatch_id, window_x, window_y, window_z, window_epoch,
           window_cluster_workgroup_id, sampled_static_mapping_for_slot(i), sync_snapshot_usable});
      break;
    case rocjitsu::ConSanMoiSampledSnapshotState::StaleGeneration:
      ++summary.sampled_stale_snapshot_count;
      break;
    case rocjitsu::ConSanMoiSampledSnapshotState::IncompletePublication:
      ++summary.sampled_incomplete_snapshot_count;
      break;
    case rocjitsu::ConSanMoiSampledSnapshotState::ChangedDuringRead:
      ++summary.sampled_changed_snapshot_count;
      break;
    case rocjitsu::ConSanMoiSampledSnapshotState::Malformed:
      result.issues.push_back({.reason = AutoMoiReportEvidenceReason::SampledMalformedWatchpoint,
                               .index = i,
                               .words = {low_after, high, window_epoch}});
      ++summary.sampled_malformed_snapshot_count;
      break;
    }
  }
  // A deferred acquire-release RMW carries a statically proven route back to
  // its preceding release-side access. The old report reader searched the
  // entire causal-window capacity once per visible access. Large framework
  // objects can publish thousands of visible entries, turning teardown into
  // a quadratic scan after the device result is already available. Index
  // visible release slots once, then scan each relevant owner bank once.
  if (header->sampled_pending_acquire_count != 0u &&
      sampled_pending_acquire_owner_bank_count != 0u && !visible_sampled.empty()) {
    const size_t missing_visible = visible_sampled.size();
    std::vector<size_t> visible_by_slot(sampled_watchpoint_capacity, missing_visible);
    std::array<bool, kConSanMoiSampledPendingAcquireOwnerBankCount> relevant_owner_banks{};
    for (size_t visible_index = 0; visible_index < visible_sampled.size(); ++visible_index) {
      const AutoMoiSampledEvidence &entry = visible_sampled[visible_index];
      if (entry.index >= visible_by_slot.size())
        continue;
      visible_by_slot[entry.index] = visible_index;
      relevant_owner_banks[entry.entry.owner_id & (sampled_pending_acquire_owner_bank_count - 1u)] =
          true;
    }
    std::vector<bool> release_sync_attached(visible_sampled.size(), false);
    std::vector<bool> release_sync_rejected(visible_sampled.size(), false);
    for (uint32_t acquire_slot = 0; acquire_slot < sampled_causal_window_capacity; ++acquire_slot) {
      for (uint32_t owner_bank = 0; owner_bank < sampled_pending_acquire_owner_bank_count;
           ++owner_bank) {
        if (!relevant_owner_banks[owner_bank])
          continue;
        const uint32_t pending_index =
            acquire_slot * sampled_pending_acquire_owner_bank_count + owner_bank;
        ++sampled_pending_release_slots_examined;
        const volatile auto &pending = sampled_pending_acquires[pending_index];
        const uint32_t version_before = pending.version;
        if (version_before == 0u || (version_before & 1u) != 0u)
          continue;
        const uint32_t reserved = pending.reserved;
        const uint32_t version_after = pending.version;
        if (version_before != version_after || reserved == 0u)
          continue;
        const uint32_t release_slot = reserved - 1u;
        if (release_slot >= visible_by_slot.size())
          continue;
        const size_t visible_index = visible_by_slot[release_slot];
        if (visible_index == missing_visible || release_sync_rejected[visible_index])
          continue;
        AutoMoiSampledEvidence &release = visible_sampled[visible_index];
        if ((release.entry.owner_id & (sampled_pending_acquire_owner_bank_count - 1u)) !=
            owner_bank)
          continue;
        if (release.sync.classification != rocjitsu::ConSanMoiSampledSyncClassification::Empty)
          continue;
        const auto release_pending = read_sampled_pending_acquire(pending_index);
        const auto release_join = rocjitsu::consan_moi_sampled_join_pending_release(
            release_pending,
            {release.generation, release.dispatch_id, release.workgroup_x, release.workgroup_y,
             release.workgroup_z, release.epoch, release.index, 1u,
             static_cast<uint32_t>(rocjitsu::ConSanMoiSampledCausalPublicationState::Ready),
             release.cluster_workgroup_id},
            release.packed_watchpoint, release.index, acquire_slot);
        if (release_join.state != rocjitsu::ConSanMoiSampledPendingAcquireState::Ready)
          continue;
        if (release_sync_attached[visible_index]) {
          ++summary.sampled_malformed_sync_count;
          release.sync = {};
          release.sync_snapshot_usable = false;
          release_sync_rejected[visible_index] = true;
          continue;
        }
        release.sync = release_join.sync;
        release_sync_attached[visible_index] = true;
      }
    }
  }
  // Preserve the original composition order: a deferred release fills only
  // empty direct metadata for its release-side access, then the acquire-side
  // join rejects a second sync role on the same access as malformed.
  if (header->sampled_pending_acquire_count != 0u &&
      sampled_pending_acquire_owner_bank_count != 0u) {
    for (AutoMoiSampledEvidence &entry : visible_sampled) {
      const uint32_t pending_index =
          entry.index * sampled_pending_acquire_owner_bank_count +
          (entry.entry.owner_id & (sampled_pending_acquire_owner_bank_count - 1u));
      if (pending_index >= sampled_pending_acquire_capacity) {
        ++summary.sampled_malformed_sync_count;
        entry.sync_snapshot_usable = false;
        continue;
      }
      const auto pending_view = read_sampled_pending_acquire(pending_index);
      const auto pending_join = rocjitsu::consan_moi_sampled_join_pending_acquire(
          pending_view,
          {entry.generation, entry.dispatch_id, entry.workgroup_x, entry.workgroup_y,
           entry.workgroup_z, entry.epoch, entry.index, 1u,
           static_cast<uint32_t>(rocjitsu::ConSanMoiSampledCausalPublicationState::Ready),
           entry.cluster_workgroup_id},
          entry.packed_watchpoint, entry.index);
      switch (pending_join.state) {
      case rocjitsu::ConSanMoiSampledPendingAcquireState::Empty:
        break;
      case rocjitsu::ConSanMoiSampledPendingAcquireState::Ready:
        if (entry.sync.classification == rocjitsu::ConSanMoiSampledSyncClassification::Empty)
          entry.sync = pending_join.sync;
        else {
          ++summary.sampled_malformed_sync_count;
          entry.sync_snapshot_usable = false;
        }
        break;
      case rocjitsu::ConSanMoiSampledPendingAcquireState::Publishing:
        ++summary.sampled_incomplete_snapshot_count;
        ++summary.sampled_malformed_sync_count;
        entry.sync_snapshot_usable = false;
        break;
      case rocjitsu::ConSanMoiSampledPendingAcquireState::ChangedDuringRead:
        ++summary.sampled_changed_snapshot_count;
        ++summary.sampled_malformed_sync_count;
        entry.sync_snapshot_usable = false;
        break;
      case rocjitsu::ConSanMoiSampledPendingAcquireState::Malformed:
      case rocjitsu::ConSanMoiSampledPendingAcquireState::IdentityMismatch:
      case rocjitsu::ConSanMoiSampledPendingAcquireState::FutureEpoch:
        ++summary.sampled_malformed_sync_count;
        entry.sync_snapshot_usable = false;
        break;
      }
    }
  }
  for (AutoMoiSampledEvidence &entry : visible_sampled) {
    if (!entry.sync_snapshot_usable)
      entry.sync = {};
  }
  visible_sampled_sync_metadata = static_cast<uint32_t>(
      std::ranges::count_if(visible_sampled, [](const AutoMoiSampledEvidence &entry) {
        return entry.sync_snapshot_usable &&
               entry.sync.classification != rocjitsu::ConSanMoiSampledSyncClassification::Empty;
      }));
  // A collision or capacity drop can leave a valid-looking first half in a
  // slot while discarding a different second publisher. Without per-slot
  // collision identity, disable all ordering suppression for that report.
  const bool sampled_sync_evidence_complete =
      rocjitsu::consan_moi_sampled_sync_report_is_complete(
          header->sampled_dropped_window_count, header->sampled_unsupported_sync_count,
          header->sampled_malformed_sync_count, header->sampled_pending_acquire_collision_count,
          header->sampled_pending_acquire_malformed_count) &&
      summary.sampled_malformed_sync_count == 0;

  const auto *records = reinterpret_cast<const ConSanMoiAccessRecord *>(
      bytes + expected_layout.access_records_offset);
  result.visible_access_slots.assign(records, records + visible_records);
  CompactRecordReplayAccessRecords compact_access_records =
      expected_engine == ConSanMoiEngine::RecordReplay
          ? compact_record_replay_access_records(result.visible_access_slots, header->event_counter)
          : CompactRecordReplayAccessRecords{};
  const uint32_t committed_records = compact_access_records.committed_record_count;

  const auto *barriers = reinterpret_cast<const ConSanMoiBarrierRecord *>(
      bytes + expected_layout.barrier_records_offset);
  result.barrier_records.assign(barriers, barriers + visible_barriers);
  const auto *atomics = reinterpret_cast<const ConSanMoiAtomicRecord *>(
      bytes + expected_layout.atomic_records_offset);
  result.atomic_records.assign(atomics, atomics + visible_atomics);
  const auto *fences =
      reinterpret_cast<const ConSanMoiFenceRecord *>(bytes + expected_layout.fence_records_offset);
  result.fence_records.assign(fences, fences + visible_fences);

  result.diagnostics.reserve(visible_diagnostics);
  for (uint32_t index : visible_diagnostic_indices)
    result.diagnostics.push_back(raw_diagnostics[index]);
  const auto *inline_metadata =
      input.static_metadata ? std::get_if<AutoMoiInlineCompactStaticMetadata>(input.static_metadata)
                            : nullptr;
  const uint32_t compact_token_mapping_count =
      inline_metadata ? inline_metadata->mapping_count : 0u;
  if (!result.diagnostics.empty() && compact_token_mapping_count != 0u) {
    const auto *mappings = reinterpret_cast<const ConSanMoiCompactDiagnosticTokenMapping *>(
        bytes + input.layout.inline_compact_token_mappings_offset);
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
        result.issues.push_back(
            {.reason = AutoMoiReportEvidenceReason::CompactDiagnosticTokenUnresolved,
             .index = diagnostic.second_instruction_offset,
             .words = {tagged, well_formed, current_ambiguous, prior_ambiguous}});
        continue;
      }
      diagnostic.first_instruction_offset = resolved->instruction_offset;
    }
  }

  summary.visible_access_record_count = committed_records;
  summary.visible_barrier_record_count = visible_barriers;
  summary.visible_atomic_record_count = visible_atomics;
  summary.visible_fence_record_count = visible_fences;
  summary.visible_diagnostic_record_count = visible_diagnostics;
  summary.visible_inline_publication_count =
      expected_engine == ConSanMoiEngine::InlineShadow ? header->event_counter : 0;
  summary.visible_exact_shadow_entry_count = visible_exact_shadow.size();
  summary.visible_inline_atomic_release_count = visible_inline_atomic_releases.size();
  summary.visible_inline_acquired_token_count = visible_inline_acquired_tokens.size();
  summary.visible_sampled_watchpoint_count = visible_sampled.size();
  summary.visible_sampled_sync_metadata_count = visible_sampled_sync_metadata;
  summary.dropped_access_record_count = dropped_records;
  summary.dropped_barrier_record_count = dropped_barriers;
  summary.dropped_atomic_record_count = dropped_atomics;
  summary.dropped_fence_record_count = dropped_fences;
  summary.dropped_diagnostic_record_count = dropped_diagnostics;
  summary.sampled_immediate_conflict_count =
      sampled_watchpoint_capacity != 0 ? header->event_counter : 0;
  summary.sampled_claimed_window_count =
      sampled_watchpoint_capacity != 0 ? header->sampled_causal_window_count : 0;
  summary.sampled_dropped_window_count =
      sampled_watchpoint_capacity != 0 ? header->sampled_dropped_window_count : 0;
  summary.sampled_saturated_window_count =
      sampled_watchpoint_capacity != 0 ? header->sampled_saturated_window_count : 0;
  summary.sampled_unsupported_sync_count =
      sampled_watchpoint_capacity != 0 ? header->sampled_unsupported_sync_count : 0;
  summary.sampled_malformed_sync_count +=
      sampled_watchpoint_capacity != 0 ? header->sampled_malformed_sync_count : 0;
  if (expected_engine == ConSanMoiEngine::InlineShadow) {
    summary.inline_undercoverage_count = header->inline_undercoverage_count;
    summary.inline_overflow_count = header->inline_overflow_count;
    summary.inline_unsupported_count = header->inline_unsupported_count;
    summary.inline_malformed_count += header->inline_malformed_count;
  }

  result.failure = AutoMoiReportDecodeFailure::None;
  result.header = *header;
  switch (expected_engine) {
  case ConSanMoiEngine::RecordReplay:
    result.mode = AutoMoiRecordReplayDecodedReport{
        .access_records = std::move(compact_access_records.replay_records)};
    break;
  case ConSanMoiEngine::InlineShadow:
    result.mode = AutoMoiInlineShadowDecodedReport{
        .deferred_token_qualified_diagnostic_count = deferred_token_qualified_diagnostics,
        .exact_shadow = std::move(visible_exact_shadow),
        .atomic_releases = std::move(visible_inline_atomic_releases),
        .acquired_tokens = std::move(visible_inline_acquired_tokens),
    };
    break;
  case ConSanMoiEngine::Sampled:
    result.mode = AutoMoiSampledDecodedReport{
        .watchpoint_slots_examined = sampled_watchpoint_slots_examined,
        .pending_release_slots_examined = sampled_pending_release_slots_examined,
        .synchronization_evidence_complete = sampled_sync_evidence_complete,
        .evidence = std::move(visible_sampled),
    };
    break;
  }
  return result;
}

} // namespace rocjitsu::consan_hook

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
  const ConSanMoiEngine expected_engine = input.inline_shadow    ? ConSanMoiEngine::InlineShadow
                                          : input.direct_sampled ? ConSanMoiEngine::Sampled
                                                                 : ConSanMoiEngine::RecordReplay;
  const void *report_ptr = snapshot.bytes.data();

  const auto *header = static_cast<const rocjitsu::ConSanMoiReportHeader *>(report_ptr);
  if (!rocjitsu::consan_moi_report_header_is_current(*header)) {
    log_message(kLogInfo,
                "ConSan MOI auto report reader=%llu has invalid header magic=0x%08x "
                "abi=%u header_size=%u",
                static_cast<unsigned long long>(input.reader), header->magic, header->abi_version,
                header->header_size);
    return summary;
  }
  const rocjitsu::ConSanMoiReportBufferLayout &expected_layout = input.layout;
  if (!rocjitsu::consan_moi_report_layout_matches_header(*header, expected_layout, expected_engine,
                                                         input.size)) {
    log_message(kLogInfo, "ConSan MOI auto report reader=%llu has inconsistent ABI-v%u layout",
                static_cast<unsigned long long>(input.reader),
                rocjitsu::kConSanMoiReportAbiVersion);
    return summary;
  }
  const bool partition_mask_debug = [] {
    const char *debug = std::getenv("RJ_CONSAN_MOI_PARTITION_MASK_DEBUG");
    return debug != nullptr && std::string_view(debug) == "1";
  }();
  // The bounded partition debugger reuses record-count words as an explicit
  // non-acceptance side channel. Keep those words out of the ordinary
  // overflow/accounting summary while preserving the registered capacities
  // that define the report layout.
  const uint32_t access_record_count = partition_mask_debug ? 0 : header->access_record_count;
  const uint32_t barrier_record_count = partition_mask_debug ? 0 : header->barrier_record_count;
  const uint32_t atomic_record_count = partition_mask_debug ? 0 : header->atomic_record_count;
  const uint32_t visible_records = std::min(access_record_count, header->access_record_capacity);
  const uint32_t visible_barriers = std::min(barrier_record_count, header->barrier_record_capacity);
  const uint32_t visible_atomics = std::min(atomic_record_count, header->atomic_record_capacity);
  const uint32_t visible_fences = std::min(header->fence_record_count, input.fence_record_capacity);
  const uint32_t raw_visible_diagnostics =
      std::min(header->diagnostic_count, header->diagnostic_capacity);
  const uint32_t dropped_records =
      access_record_count > visible_records ? access_record_count - visible_records : 0;
  const uint32_t dropped_barriers =
      barrier_record_count > visible_barriers ? barrier_record_count - visible_barriers : 0;
  const uint32_t dropped_atomics =
      atomic_record_count > visible_atomics ? atomic_record_count - visible_atomics : 0;
  const uint32_t dropped_fences =
      input.fence_record_capacity != 0 && header->fence_record_count > visible_fences
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
  const uint32_t inline_atomic_release_capacity =
      input.inline_shadow ? header->inline_atomic_release_capacity : 0;
  const auto *inline_acquired_tokens =
      reinterpret_cast<const volatile rocjitsu::ConSanMoiInlineAcquiredEpochTokenSlot *>(
          bytes + expected_layout.inline_acquired_epoch_token_slots_offset);
  const uint32_t inline_acquired_token_capacity =
      input.inline_shadow ? header->inline_acquired_epoch_token_capacity : 0;
  const auto *inline_causal_snapshots =
      reinterpret_cast<const rocjitsu::ConSanMoiInlineCausalSnapshot *>(
          bytes + expected_layout.inline_causal_snapshots_offset);
  const auto *sampled_causal_windows =
      reinterpret_cast<const rocjitsu::ConSanMoiSampledCausalWindow *>(
          bytes + expected_layout.sampled_causal_windows_offset);
  const uint32_t sampled_watchpoint_capacity =
      input.direct_sampled ? header->sampled_watchpoint_capacity : 0;
  const uint32_t sampled_causal_window_capacity =
      input.direct_sampled ? header->sampled_causal_window_capacity : 0;
  const auto *sampled =
      reinterpret_cast<const uint64_t *>(bytes + expected_layout.sampled_watchpoints_offset);
  const auto *sampled_sync_words = reinterpret_cast<const volatile uint32_t *>(
      bytes + expected_layout.sampled_sync_metadata_offset);
  const uint32_t sampled_sync_metadata_capacity =
      input.direct_sampled ? header->sampled_sync_metadata_capacity : 0;
  const auto *sampled_pending_acquires =
      reinterpret_cast<const volatile rocjitsu::ConSanMoiSampledPendingAcquireSlot *>(
          bytes + expected_layout.sampled_pending_acquires_offset);
  const uint32_t sampled_pending_acquire_capacity =
      input.direct_sampled ? header->sampled_pending_acquire_capacity : 0;
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
  struct ExactShadowEntry {
    uint32_t index = 0;
    rocjitsu::ConSanMoiExactShadowEntry entry;
    rocjitsu::ConSanMoiExactByteCellProvenance byte_provenance;
    uint64_t dispatch_id = 0;
    uint32_t version = 0;
  };
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
      if (summary.exact_malformed_snapshot_count == 0) {
        log_message(kLogInfo,
                    "ConSan MOI first malformed exact snapshot reader=%llu index=%u "
                    "version_before=%u packed_access=0x%016llx dispatch_id=0x%016llx "
                    "byte_provenance=0x%08x version_after=%u",
                    static_cast<unsigned long long>(input.reader), i, version_before,
                    static_cast<unsigned long long>(packed_access),
                    static_cast<unsigned long long>(dispatch_id), byte_provenance, version_after);
      }
      ++summary.exact_malformed_snapshot_count;
      break;
    }
  }
  struct InlineAtomicReleaseEntry {
    uint32_t index = 0;
    rocjitsu::ConSanMoiInlineAtomicReleaseSlot slot;
    rocjitsu::ConSanMoiInlineCausalSnapshot snapshot;
  };
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
      log_message(kLogInfo,
                  "ConSan MOI auto incomplete-inline-atomic-release reader=%llu index=%u "
                  "version=%u owner=%u epoch_plus_one=%u workgroup=%u address=0x%llx "
                  "dispatch=0x%llx",
                  static_cast<unsigned long long>(input.reader), i, words.slot.version,
                  words.slot.owner_id, words.slot.epoch_plus_one, words.slot.workgroup_key,
                  static_cast<unsigned long long>(words.slot.atomic_address),
                  static_cast<unsigned long long>(words.slot.dispatch_id));
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
  struct InlineAcquiredTokenEntry {
    uint32_t index = 0;
    rocjitsu::ConSanMoiInlineAcquiredEpochTokenSlot token;
  };
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
      input.inline_shadow && summary.token_incomplete_snapshot_count == 0 &&
      summary.token_changed_snapshot_count == 0 && summary.token_malformed_snapshot_count == 0 &&
      header->inline_overflow_count == 0 && header->inline_malformed_count == 0;
  const auto deferred_filter = rocjitsu::consan_moi_filter_deferred_inline_diagnostics(
      std::span<const rocjitsu::ConSanMoiDiagnosticRecord>(raw_diagnostics,
                                                           raw_visible_diagnostics),
      header->diagnostic_count, stable_inline_acquired_tokens, deferred_token_evidence_complete);
  const auto &visible_diagnostic_indices = deferred_filter.visible_indices;
  const uint32_t deferred_token_qualified_diagnostics = deferred_filter.qualified_count;
  const uint32_t visible_diagnostics = static_cast<uint32_t>(visible_diagnostic_indices.size());
  const uint32_t effective_diagnostic_count = deferred_filter.effective_diagnostic_count;
  if (deferred_token_qualified_diagnostics != 0) {
    log_message(
        kLogInfo, "ConSan MOI deferred-token qualification reader=%llu ordered_diagnostics=%u",
        static_cast<unsigned long long>(input.reader), deferred_token_qualified_diagnostics);
  }
  using SampledEntry = AutoMoiSampledEvidence;
  std::vector<SampledEntry> visible_sampled;
  const auto sampled_static_mapping_for_slot = [&](uint32_t slot) {
    const auto mapping = std::ranges::find_if(
        input.sampled_static_mappings, [&](const AutoMoiSampledStaticMapping &candidate) {
          if (slot < candidate.first_slot)
            return false;
          const uint64_t relative = slot - candidate.first_slot;
          return relative < static_cast<uint64_t>(candidate.range_count) * candidate.bank_count;
        });
    return mapping == input.sampled_static_mappings.end() ? nullptr : &*mapping;
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
      log_message(kLogInfo,
                  "ConSan MOI sampled malformed window index=%u state=%u generation=%llu/%llu "
                  "dispatch=%llu/%llu epoch=%u first=%u entries=%u "
                  "cluster_workgroup_id=%u snapshot=%u",
                  i, state_after, static_cast<unsigned long long>(window_generation),
                  static_cast<unsigned long long>(header->generation),
                  static_cast<unsigned long long>(window_dispatch_id),
                  static_cast<unsigned long long>(header->dispatch_id), window_epoch,
                  window_first_entry, window_entry_count, window_cluster_workgroup_id,
                  static_cast<uint32_t>(snapshot.state));
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
      log_message(kLogInfo, "ConSan MOI sampled malformed empty watchpoint index=%u state=%u", i,
                  static_cast<uint32_t>(snapshot.state));
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
      log_message(kLogInfo,
                  "ConSan MOI sampled malformed packed watchpoint index=%u low=0x%08x "
                  "high=0x%08x epoch=%u",
                  i, low_after, high, window_epoch);
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
      const SampledEntry &entry = visible_sampled[visible_index];
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
        SampledEntry &release = visible_sampled[visible_index];
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
    for (SampledEntry &entry : visible_sampled) {
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
  for (SampledEntry &entry : visible_sampled) {
    if (!entry.sync_snapshot_usable)
      entry.sync = {};
  }
  visible_sampled_sync_metadata =
      static_cast<uint32_t>(std::ranges::count_if(visible_sampled, [](const SampledEntry &entry) {
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
  const AutoMoiSampledConflictAnalysis sampled_analysis =
      analyze_auto_moi_sampled_conflicts(visible_sampled, sampled_sync_evidence_complete);
  const uint32_t sampled_conflicts = sampled_analysis.conflict_count;
  const auto &first_sampled_conflict = sampled_analysis.first_conflict;
  const auto *records = reinterpret_cast<const rocjitsu::ConSanMoiAccessRecord *>(
      bytes + expected_layout.access_records_offset);
  rocjitsu::consan_hook::CompactRecordReplayAccessRecords compact_access_records;
  if (expected_engine == ConSanMoiEngine::RecordReplay) {
    compact_access_records = rocjitsu::consan_hook::compact_record_replay_access_records(
        std::span<const rocjitsu::ConSanMoiAccessRecord>(records, visible_records),
        header->event_counter);
  }
  const uint32_t committed_records = compact_access_records.committed_record_count;
  std::vector<rocjitsu::ConSanMoiAccessRecord> &replay_access_records =
      compact_access_records.replay_records;
  const auto *barriers = reinterpret_cast<const rocjitsu::ConSanMoiBarrierRecord *>(
      static_cast<const uint8_t *>(report_ptr) + expected_layout.barrier_records_offset);
  const auto *atomics = reinterpret_cast<const rocjitsu::ConSanMoiAtomicRecord *>(
      static_cast<const uint8_t *>(report_ptr) + expected_layout.atomic_records_offset);
  const auto *fences = reinterpret_cast<const rocjitsu::ConSanMoiFenceRecord *>(
      static_cast<const uint8_t *>(report_ptr) + expected_layout.fence_records_offset);
  const AutoMoiRecordReplayAnalysis record_replay_analysis = analyze_auto_moi_record_replay(
      *header, expected_engine, replay_access_records,
      std::span<const ConSanMoiBarrierRecord>(barriers, visible_barriers),
      std::span<const ConSanMoiAtomicRecord>(atomics, visible_atomics),
      std::span<const ConSanMoiFenceRecord>(fences, visible_fences),
      input.record_replay_static_mappings, input.record_replay_static_mapping_malformed,
      expected_layout.record_replay_logical_access_range_count,
      expected_layout.record_replay_address_group_headroom);
  const RecordReplayPressureTelemetry &record_replay_pressure = record_replay_analysis.pressure;
  summary.visible_access_record_count = committed_records;
  summary.visible_barrier_record_count = visible_barriers;
  summary.visible_atomic_record_count = visible_atomics;
  summary.visible_fence_record_count = visible_fences;
  summary.visible_diagnostic_record_count = visible_diagnostics;
  summary.visible_inline_publication_count =
      input.inline_shadow && !partition_mask_debug ? header->event_counter : 0;
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
  summary.inline_undercoverage_count =
      input.inline_shadow && !partition_mask_debug ? header->inline_undercoverage_count : 0;
  summary.inline_overflow_count = input.inline_shadow ? header->inline_overflow_count : 0;
  summary.inline_unsupported_count = input.inline_shadow ? header->inline_unsupported_count : 0;
  summary.inline_malformed_count = input.inline_shadow ? header->inline_malformed_count : 0;
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

  std::vector<rocjitsu::ConSanMoiDiagnosticRecord> resolved_diagnostics;
  resolved_diagnostics.reserve(visible_diagnostics);
  for (uint32_t index : visible_diagnostic_indices)
    resolved_diagnostics.push_back(raw_diagnostics[index]);
  const auto *diagnostics = resolved_diagnostics.data();
  if (visible_diagnostics != 0u && input.compact_token_mapping_count != 0u) {
    const auto *mappings =
        reinterpret_cast<const rocjitsu::ConSanMoiCompactDiagnosticTokenMapping *>(
            static_cast<const uint8_t *>(report_ptr) +
            input.layout.inline_compact_token_mappings_offset);
    for (rocjitsu::ConSanMoiDiagnosticRecord &diagnostic : resolved_diagnostics) {
      const uint32_t tagged = diagnostic.first_instruction_offset;
      constexpr uint32_t kTokenPayloadMask = rocjitsu::consan_moi_exact_shadow::max_compact_token;
      constexpr uint32_t kAllowedBits =
          rocjitsu::consan_moi_exact_shadow::compact_diagnostic_token_tag | kTokenPayloadMask;
      if ((tagged & rocjitsu::consan_moi_exact_shadow::compact_diagnostic_token_tag) == 0u)
        continue;
      const uint16_t token = static_cast<uint16_t>(tagged & kTokenPayloadMask);
      const bool well_formed = token != 0u && (tagged & ~kAllowedBits) == 0u;
      const rocjitsu::ConSanMoiCompactDiagnosticTokenMapping *current = nullptr;
      const rocjitsu::ConSanMoiCompactDiagnosticTokenMapping *resolved = nullptr;
      bool current_ambiguous = false;
      bool prior_ambiguous = false;
      if (well_formed) {
        for (uint32_t i = 0; i < input.compact_token_mapping_count; ++i) {
          const auto &mapping = mappings[i];
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
          for (uint32_t i = 0; i < input.compact_token_mapping_count; ++i) {
            const auto &mapping = mappings[i];
            if (mapping.owner_descriptor_file_offset != current->owner_descriptor_file_offset ||
                mapping.token != token) {
              continue;
            }
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
        log_message(kLogInfo,
                    "ConSan MOI compact diagnostic token unresolved reader=%llu current=0x%x "
                    "tagged=0x%x well_formed=%s current_ambiguous=%s "
                    "prior_ambiguous=%s",
                    static_cast<unsigned long long>(input.reader),
                    diagnostic.second_instruction_offset, tagged, well_formed ? "true" : "false",
                    current_ambiguous ? "true" : "false", prior_ambiguous ? "true" : "false");
        continue;
      }
      diagnostic.first_instruction_offset = resolved->instruction_offset;
    }
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
    const ExactShadowEntry &shadow_entry = visible_exact_shadow[i];
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
    const SampledEntry &sampled_entry = visible_sampled[i];
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
    const SampledEntry &first = first_sampled_conflict->first;
    const SampledEntry &second = first_sampled_conflict->second;
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

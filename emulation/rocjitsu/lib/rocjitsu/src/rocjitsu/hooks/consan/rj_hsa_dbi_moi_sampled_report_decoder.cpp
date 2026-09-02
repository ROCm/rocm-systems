// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_sampled_report_decoder.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <utility>

namespace rocjitsu::consan_hook {

AutoMoiSampledDecodedReport decode_auto_moi_sampled_report(
    const AutoMoiReportPipelineInput &input, const ConSanMoiReportHeader &report_header,
    std::span<const uint8_t> report_bytes, AutoMoiReportSummary &summary) {
  const auto *header = &report_header;
  const ConSanMoiReportBufferLayout &layout = input.layout;
  const uint8_t *bytes = report_bytes.data();
  const auto *sampled_causal_windows = reinterpret_cast<const ConSanMoiSampledCausalWindow *>(
      bytes + layout.sampled_causal_windows_offset);
  const uint32_t sampled_watchpoint_capacity = layout.sampled_watchpoint_capacity;
  const uint32_t sampled_causal_window_capacity = layout.sampled_causal_window_capacity;
  const auto *sampled =
      reinterpret_cast<const uint64_t *>(bytes + layout.sampled_watchpoints_offset);
  const auto *sampled_sync_words =
      reinterpret_cast<const volatile uint32_t *>(bytes + layout.sampled_sync_metadata_offset);
  const uint32_t sampled_sync_metadata_capacity = layout.sampled_sync_metadata_capacity;
  const auto *sampled_pending_acquires =
      reinterpret_cast<const volatile ConSanMoiSampledPendingAcquireSlot *>(
          bytes + layout.sampled_pending_acquires_offset);
  const uint32_t sampled_pending_acquire_capacity = layout.sampled_pending_acquire_capacity;
  const uint32_t sampled_pending_acquire_owner_bank_count =
      consan_moi_sampled_pending_acquire_owner_bank_count(sampled_pending_acquire_capacity,
                                                          sampled_causal_window_capacity);
  const auto read_sampled_pending_acquire = [&](uint32_t pending_index) {
    ConSanMoiSampledPendingAcquireView view{};
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
  const auto *sampled_metadata =
      input.static_metadata ? std::get_if<AutoMoiSampledStaticMetadata>(input.static_metadata)
                            : nullptr;
  const std::span<const AutoMoiSampledStaticMapping> sampled_static_mappings =
      sampled_metadata ? sampled_metadata->mappings
                       : std::span<const AutoMoiSampledStaticMapping>{};
  std::vector<AutoMoiSampledEvidence> visible_sampled;
  std::vector<AutoMoiSampledEvidenceIssue> sampled_issues;
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
      static_cast<uint32_t>(header->generation) & consan_moi_sampled_watchpoint::max_generation;
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
    ConSanMoiSampledSyncMetadataPacked sync_packed{};
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
    ConSanMoiSampledSyncDecodeResult sync =
        i < sampled_sync_metadata_capacity
            ? classify_consan_moi_sampled_sync_snapshot(
                  {sync_descriptor_before, sync_packed, sync_descriptor_after}, window_epoch)
            : ConSanMoiSampledSyncDecodeResult{};
    const uint32_t state_after = window.publication_state;
    const ConSanMoiSampledSnapshot snapshot = classify_consan_moi_sampled_snapshot(
        {low_before, high, low_after}, active_sampled_generation);
    if (state_before != state_after) {
      ++summary.sampled_changed_snapshot_count;
      continue;
    }
    const auto publication_state = static_cast<ConSanMoiSampledCausalPublicationState>(state_after);
    if (publication_state == ConSanMoiSampledCausalPublicationState::Empty) {
      if (snapshot.state != ConSanMoiSampledSnapshotState::Empty ||
          sync.classification != ConSanMoiSampledSyncClassification::Empty)
        ++summary.sampled_malformed_snapshot_count;
      continue;
    }
    if (publication_state == ConSanMoiSampledCausalPublicationState::Publishing) {
      ++summary.sampled_incomplete_snapshot_count;
      continue;
    }
    if (publication_state == ConSanMoiSampledCausalPublicationState::Ready)
      ++sampled_ready_windows_examined;
    if (publication_state != ConSanMoiSampledCausalPublicationState::Ready ||
        window_generation != header->generation ||
        window_epoch > consan_moi_sampled_watchpoint::max_epoch || window_first_entry != i ||
        window_entry_count != 1) {
      sampled_issues.push_back(
          {.reason = AutoMoiSampledEvidenceReason::MalformedWindow,
           .index = i,
           .words = {state_after, window_generation, header->generation, window_dispatch_id,
                     header->dispatch_id, window_epoch, window_first_entry, window_entry_count,
                     window_cluster_workgroup_id, static_cast<uint32_t>(snapshot.state)}});
      ++summary.sampled_malformed_snapshot_count;
      continue;
    }
    bool sync_snapshot_usable = true;
    if (sync.classification == ConSanMoiSampledSyncClassification::ChangedDuringRead) {
      ++summary.sampled_changed_snapshot_count;
      sync_snapshot_usable = false;
    }
    if (sync.classification == ConSanMoiSampledSyncClassification::Publishing) {
      ++summary.sampled_incomplete_snapshot_count;
      sync_snapshot_usable = false;
    }
    const bool has_watchpoint = snapshot.state != ConSanMoiSampledSnapshotState::Empty;
    if (!has_watchpoint) {
      sampled_issues.push_back({.reason = AutoMoiSampledEvidenceReason::EmptyWatchpoint,
                                .index = i,
                                .words = {static_cast<uint32_t>(snapshot.state)}});
      ++summary.sampled_malformed_snapshot_count;
      continue;
    }
    const bool has_sync = sync.classification != ConSanMoiSampledSyncClassification::Empty;
    if (has_sync && sync_snapshot_usable &&
        sync.classification != ConSanMoiSampledSyncClassification::Valid) {
      ++summary.sampled_malformed_sync_count;
      sync_snapshot_usable = false;
    }
    switch (snapshot.state) {
    case ConSanMoiSampledSnapshotState::Empty:
      break;
    case ConSanMoiSampledSnapshotState::Stable:
      if (snapshot.entry.epoch != window_epoch) {
        ++summary.sampled_malformed_snapshot_count;
        sampled_issues.push_back({.reason = AutoMoiSampledEvidenceReason::MalformedWatchpoint,
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
    case ConSanMoiSampledSnapshotState::StaleGeneration:
      ++summary.sampled_stale_snapshot_count;
      break;
    case ConSanMoiSampledSnapshotState::IncompletePublication:
      ++summary.sampled_incomplete_snapshot_count;
      break;
    case ConSanMoiSampledSnapshotState::ChangedDuringRead:
      ++summary.sampled_changed_snapshot_count;
      break;
    case ConSanMoiSampledSnapshotState::Malformed:
      sampled_issues.push_back({.reason = AutoMoiSampledEvidenceReason::MalformedWatchpoint,
                                .index = i,
                                .words = {low_after, high, window_epoch}});
      ++summary.sampled_malformed_snapshot_count;
      break;
    }
  }

  // A deferred acquire-release RMW carries a statically proven route back to
  // its preceding release-side access. Index visible release slots once, then
  // scan each relevant owner bank once.
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
        if (release.sync.classification != ConSanMoiSampledSyncClassification::Empty)
          continue;
        const auto release_pending = read_sampled_pending_acquire(pending_index);
        const auto release_join = consan_moi_sampled_join_pending_release(
            release_pending,
            {release.generation, release.dispatch_id, release.workgroup_x, release.workgroup_y,
             release.workgroup_z, release.epoch, release.index, 1u,
             static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready),
             release.cluster_workgroup_id},
            release.packed_watchpoint, release.index, acquire_slot);
        if (release_join.state != ConSanMoiSampledPendingAcquireState::Ready)
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

  // A deferred release fills only empty direct metadata for its release-side
  // access, then the acquire-side join rejects a second sync role.
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
      const auto pending_join = consan_moi_sampled_join_pending_acquire(
          pending_view,
          {entry.generation, entry.dispatch_id, entry.workgroup_x, entry.workgroup_y,
           entry.workgroup_z, entry.epoch, entry.index, 1u,
           static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready),
           entry.cluster_workgroup_id},
          entry.packed_watchpoint, entry.index);
      switch (pending_join.state) {
      case ConSanMoiSampledPendingAcquireState::Empty:
        break;
      case ConSanMoiSampledPendingAcquireState::Ready:
        if (entry.sync.classification == ConSanMoiSampledSyncClassification::Empty)
          entry.sync = pending_join.sync;
        else {
          ++summary.sampled_malformed_sync_count;
          entry.sync_snapshot_usable = false;
        }
        break;
      case ConSanMoiSampledPendingAcquireState::Publishing:
        ++summary.sampled_incomplete_snapshot_count;
        ++summary.sampled_malformed_sync_count;
        entry.sync_snapshot_usable = false;
        break;
      case ConSanMoiSampledPendingAcquireState::ChangedDuringRead:
        ++summary.sampled_changed_snapshot_count;
        ++summary.sampled_malformed_sync_count;
        entry.sync_snapshot_usable = false;
        break;
      case ConSanMoiSampledPendingAcquireState::Malformed:
      case ConSanMoiSampledPendingAcquireState::IdentityMismatch:
      case ConSanMoiSampledPendingAcquireState::FutureEpoch:
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
               entry.sync.classification != ConSanMoiSampledSyncClassification::Empty;
      }));
  const bool sampled_sync_evidence_complete =
      consan_moi_sampled_sync_report_is_complete(
          header->sampled_dropped_window_count, header->sampled_unsupported_sync_count,
          header->sampled_malformed_sync_count, header->sampled_pending_acquire_collision_count,
          header->sampled_pending_acquire_malformed_count) &&
      summary.sampled_malformed_sync_count == 0;

  summary.visible_sampled_watchpoint_count = visible_sampled.size();
  summary.visible_sampled_sync_metadata_count = visible_sampled_sync_metadata;
  summary.sampled_immediate_conflict_count = header->event_counter;
  summary.sampled_claimed_window_count = header->sampled_causal_window_count;
  summary.sampled_dropped_window_count = header->sampled_dropped_window_count;
  summary.sampled_saturated_window_count = header->sampled_saturated_window_count;
  summary.sampled_unsupported_sync_count = header->sampled_unsupported_sync_count;
  summary.sampled_malformed_sync_count += header->sampled_malformed_sync_count;

  return {
      .watchpoint_slots_examined = sampled_watchpoint_slots_examined,
      .pending_release_slots_examined = sampled_pending_release_slots_examined,
      .synchronization_evidence_complete = sampled_sync_evidence_complete,
      .evidence = std::move(visible_sampled),
      .issues = std::move(sampled_issues),
  };
}

} // namespace rocjitsu::consan_hook

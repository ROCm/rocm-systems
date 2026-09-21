// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_evidence_decoder.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <utility>

namespace rocjitsu::consan::hook {

DecodedEvidence decode_evidence(const ReportPipelineInput &input, const ReportHeader &report_header,
                                std::span<const uint8_t> report_bytes, ReportSummary &summary) {
  const auto *header = &report_header;
  const ReportBufferLayout &layout = input.layout;
  const uint8_t *bytes = report_bytes.data();
  const auto *causal_windows =
      reinterpret_cast<const CausalWindow *>(bytes + layout.causal_windows_offset);
  const uint32_t watchpoint_capacity = layout.watchpoint_capacity;
  const uint32_t causal_window_capacity = layout.causal_window_capacity;
  const auto *watchpoints = reinterpret_cast<const uint64_t *>(bytes + layout.watchpoints_offset);
  const auto *sync_words =
      reinterpret_cast<const volatile uint32_t *>(bytes + layout.sync_metadata_offset);
  const uint32_t sync_metadata_capacity = layout.sync_metadata_capacity;
  const auto *pending_acquires =
      reinterpret_cast<const volatile PendingAcquireSlot *>(bytes + layout.pending_acquires_offset);
  const uint32_t pending_acquire_capacity = layout.pending_acquire_capacity;
  const uint32_t pending_acquire_owner_bank_count =
      consan::pending_acquire_owner_bank_count(pending_acquire_capacity, causal_window_capacity);
  const auto read_pending_acquire = [&](uint32_t pending_index) {
    PendingAcquireView view{};
    if (pending_index >= pending_acquire_capacity)
      return view;
    const volatile auto &pending = pending_acquires[pending_index];
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
  const auto *metadata = input.static_metadata;
  const std::span<const AccessStaticMapping> static_mappings =
      metadata ? metadata->mappings : std::span<const AccessStaticMapping>{};
  std::vector<Evidence> visible;
  std::vector<EvidenceIssue> issues;
  const auto static_mapping_for_slot = [&](uint32_t slot) {
    const auto mapping =
        std::ranges::find_if(static_mappings, [&](const AccessStaticMapping &candidate) {
          if (slot < candidate.first_slot)
            return false;
          const uint64_t relative = slot - candidate.first_slot;
          return relative < static_cast<uint64_t>(candidate.range_count) * candidate.bank_count;
        });
    if (mapping == static_mappings.end())
      return static_cast<const AccessStaticMapping *>(nullptr);
    if (mapping->uniform_lds_store) {
      // A proof may suppress a diagnostic only with unambiguous attribution.
      if (metadata->malformed || std::ranges::any_of(static_mappings, [&](const auto &other) {
            return &other != &*mapping && slot >= other.first_slot &&
                   uint64_t{slot - other.first_slot} <
                       uint64_t{other.range_count} * other.bank_count;
          }))
        return static_cast<const AccessStaticMapping *>(nullptr);
    }
    return &*mapping;
  };
  uint32_t visible_sync_metadata = 0;
  uint64_t watchpoint_slots_examined = 0;
  uint64_t pending_release_slots_examined = 0;
  const auto *words = reinterpret_cast<const volatile uint32_t *>(watchpoints);
  const uint32_t active_generation =
      static_cast<uint32_t>(header->generation) & watchpoint::max_generation;
  const uint32_t ready_window_target = std::min(header->causal_window_count, watchpoint_capacity);
  uint32_t ready_windows_examined = 0;
  for (uint32_t i = 0; i < watchpoint_capacity && ready_windows_examined < ready_window_target;
       ++i) {
    ++watchpoint_slots_examined;
    if (i >= causal_window_capacity) {
      ++summary.malformed_snapshot_count;
      continue;
    }
    const volatile auto &window = causal_windows[i];
    const uint32_t state_before = window.publication_state;
    const uint32_t low_before = words[2u * i];
    const uint32_t high = words[2u * i + 1u];
    const uint32_t low_after = words[2u * i];
    const uint64_t window_generation = window.generation;
    const uint64_t window_dispatch_id = window.dispatch_id;
    const uint32_t window_x = window.workgroup_x;
    const uint32_t window_y = window.workgroup_y;
    const uint32_t window_z = window.workgroup_z;
    const uint32_t window_epoch = window.epoch;
    const uint32_t window_first_entry = window.first_entry;
    const uint32_t window_entry_count = window.entry_count;
    const uint32_t window_cluster_workgroup_id = window.cluster_workgroup_id;
    const uint64_t exact_lane_mask = window.exact_lane_mask;
    SyncMetadataPacked sync_packed{};
    uint32_t sync_descriptor_before = 0;
    uint32_t sync_descriptor_after = 0;
    if (i < sync_metadata_capacity) {
      const size_t sync_word = static_cast<size_t>(i) * (sizeof(sync_packed) / sizeof(uint32_t));
      sync_descriptor_before = sync_words[sync_word + 3u];
      sync_packed.address = static_cast<uint64_t>(sync_words[sync_word]) |
                            (static_cast<uint64_t>(sync_words[sync_word + 1u]) << 32u);
      sync_packed.byte_count = sync_words[sync_word + 2u];
      sync_packed.epoch_before = sync_words[sync_word + 4u];
      sync_packed.epoch_after = sync_words[sync_word + 5u];
      sync_descriptor_after = sync_words[sync_word + 3u];
    }
    SyncDecodeResult sync =
        i < sync_metadata_capacity
            ? classify_sync_snapshot({sync_descriptor_before, sync_packed, sync_descriptor_after},
                                     window_epoch)
            : SyncDecodeResult{};
    const uint32_t state_after = window.publication_state;
    const Snapshot snapshot = classify_snapshot({low_before, high, low_after}, active_generation);
    if (state_before != state_after) {
      ++summary.changed_snapshot_count;
      continue;
    }
    const auto publication_state = static_cast<CausalPublicationState>(state_after);
    if (publication_state == CausalPublicationState::Empty) {
      if (snapshot.state != SnapshotState::Empty ||
          sync.classification != SyncClassification::Empty)
        ++summary.malformed_snapshot_count;
      continue;
    }
    if (publication_state == CausalPublicationState::Publishing) {
      ++summary.incomplete_snapshot_count;
      continue;
    }
    if (publication_state == CausalPublicationState::Ready)
      ++ready_windows_examined;
    if (publication_state != CausalPublicationState::Ready ||
        window_generation != header->generation || window_epoch > watchpoint::max_epoch ||
        window_first_entry != i || window_entry_count != 1) {
      issues.push_back(
          {.reason = EvidenceReason::MalformedWindow,
           .index = i,
           .words = {state_after, window_generation, header->generation, window_dispatch_id,
                     header->dispatch_id, window_epoch, window_first_entry, window_entry_count,
                     window_cluster_workgroup_id, static_cast<uint32_t>(snapshot.state)}});
      ++summary.malformed_snapshot_count;
      continue;
    }
    bool sync_snapshot_usable = true;
    if (sync.classification == SyncClassification::ChangedDuringRead) {
      ++summary.changed_snapshot_count;
      sync_snapshot_usable = false;
    }
    if (sync.classification == SyncClassification::Publishing) {
      ++summary.incomplete_snapshot_count;
      sync_snapshot_usable = false;
    }
    const bool has_watchpoint = snapshot.state != SnapshotState::Empty;
    if (!has_watchpoint) {
      issues.push_back({.reason = EvidenceReason::EmptyWatchpoint,
                        .index = i,
                        .words = {static_cast<uint32_t>(snapshot.state)}});
      ++summary.malformed_snapshot_count;
      continue;
    }
    const bool has_sync = sync.classification != SyncClassification::Empty;
    if (has_sync && sync_snapshot_usable && sync.classification != SyncClassification::Valid) {
      ++summary.malformed_sync_count;
      sync_snapshot_usable = false;
    }
    switch (snapshot.state) {
    case SnapshotState::Empty:
      break;
    case SnapshotState::Stable:
      if (snapshot.entry.epoch != window_epoch) {
        ++summary.malformed_snapshot_count;
        issues.push_back({.reason = EvidenceReason::MalformedWatchpoint,
                          .index = i,
                          .words = {low_after, high, window_epoch, snapshot.entry.epoch}});
        break;
      }
      visible.push_back({i, snapshot.entry, sync,
                         static_cast<uint64_t>(low_after) | (static_cast<uint64_t>(high) << 32u),
                         window_generation, window_dispatch_id, window_x, window_y, window_z,
                         window_epoch, window_cluster_workgroup_id, sync_snapshot_usable,
                         static_mapping_for_slot(i), exact_lane_mask});
      break;
    case SnapshotState::StaleGeneration:
      ++summary.stale_snapshot_count;
      break;
    case SnapshotState::IncompletePublication:
      ++summary.incomplete_snapshot_count;
      break;
    case SnapshotState::ChangedDuringRead:
      ++summary.changed_snapshot_count;
      break;
    case SnapshotState::Malformed:
      issues.push_back({.reason = EvidenceReason::MalformedWatchpoint,
                        .index = i,
                        .words = {low_after, high, window_epoch}});
      ++summary.malformed_snapshot_count;
      break;
    }
  }

  // A deferred acquire-release RMW carries a statically proven route back to
  // its preceding release-side access. Index visible release slots once, then
  // scan each relevant owner bank once.
  if (header->pending_acquire_count != 0u && pending_acquire_owner_bank_count != 0u &&
      !visible.empty()) {
    const size_t missing_visible = visible.size();
    std::vector<size_t> visible_by_slot(watchpoint_capacity, missing_visible);
    std::array<bool, kPendingAcquireOwnerBankCount> relevant_owner_banks{};
    for (size_t visible_index = 0; visible_index < visible.size(); ++visible_index) {
      const Evidence &entry = visible[visible_index];
      if (entry.index >= visible_by_slot.size())
        continue;
      visible_by_slot[entry.index] = visible_index;
      relevant_owner_banks[entry.entry.owner_id & (pending_acquire_owner_bank_count - 1u)] = true;
    }
    std::vector<bool> release_sync_attached(visible.size(), false);
    std::vector<bool> release_sync_rejected(visible.size(), false);
    for (uint32_t acquire_slot = 0; acquire_slot < causal_window_capacity; ++acquire_slot) {
      for (uint32_t owner_bank = 0; owner_bank < pending_acquire_owner_bank_count; ++owner_bank) {
        if (!relevant_owner_banks[owner_bank])
          continue;
        const uint32_t pending_index = acquire_slot * pending_acquire_owner_bank_count + owner_bank;
        ++pending_release_slots_examined;
        const volatile auto &pending = pending_acquires[pending_index];
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
        Evidence &release = visible[visible_index];
        if ((release.entry.owner_id & (pending_acquire_owner_bank_count - 1u)) != owner_bank)
          continue;
        if (release.sync.classification != SyncClassification::Empty)
          continue;
        const auto release_pending = read_pending_acquire(pending_index);
        const auto release_join = join_pending_release(
            release_pending,
            {release.generation, release.dispatch_id, release.workgroup_x, release.workgroup_y,
             release.workgroup_z, release.epoch, release.index, 1u,
             static_cast<uint32_t>(CausalPublicationState::Ready), release.cluster_workgroup_id},
            release.packed_watchpoint, release.index, acquire_slot);
        if (release_join.state != PendingAcquireState::Ready)
          continue;
        if (release_sync_attached[visible_index]) {
          ++summary.malformed_sync_count;
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
  if (header->pending_acquire_count != 0u && pending_acquire_owner_bank_count != 0u) {
    for (Evidence &entry : visible) {
      const uint32_t pending_index =
          entry.index * pending_acquire_owner_bank_count +
          (entry.entry.owner_id & (pending_acquire_owner_bank_count - 1u));
      if (pending_index >= pending_acquire_capacity) {
        ++summary.malformed_sync_count;
        entry.sync_snapshot_usable = false;
        continue;
      }
      const auto pending_view = read_pending_acquire(pending_index);
      const auto pending_join = join_pending_acquire(
          pending_view,
          {entry.generation, entry.dispatch_id, entry.workgroup_x, entry.workgroup_y,
           entry.workgroup_z, entry.epoch, entry.index, 1u,
           static_cast<uint32_t>(CausalPublicationState::Ready), entry.cluster_workgroup_id},
          entry.packed_watchpoint, entry.index);
      switch (pending_join.state) {
      case PendingAcquireState::Empty:
        break;
      case PendingAcquireState::Ready:
        if (entry.sync.classification == SyncClassification::Empty)
          entry.sync = pending_join.sync;
        else {
          ++summary.malformed_sync_count;
          entry.sync_snapshot_usable = false;
        }
        break;
      case PendingAcquireState::Publishing:
        ++summary.incomplete_snapshot_count;
        ++summary.malformed_sync_count;
        entry.sync_snapshot_usable = false;
        break;
      case PendingAcquireState::ChangedDuringRead:
        ++summary.changed_snapshot_count;
        ++summary.malformed_sync_count;
        entry.sync_snapshot_usable = false;
        break;
      case PendingAcquireState::IdentityMismatch:
      case PendingAcquireState::FutureEpoch:
        // The pending table is hashed by selected slot and owner bank. A
        // stable record for another workgroup/dispatch, or for a later epoch
        // of this identity, is an ordinary non-match rather than malformed
        // evidence for the visible window being decoded.
        break;
      case PendingAcquireState::Malformed:
        ++summary.malformed_sync_count;
        entry.sync_snapshot_usable = false;
        break;
      }
    }
  }
  for (Evidence &entry : visible) {
    if (!entry.sync_snapshot_usable)
      entry.sync = {};
  }
  visible_sync_metadata =
      static_cast<uint32_t>(std::ranges::count_if(visible, [](const Evidence &entry) {
        return entry.sync_snapshot_usable && entry.sync.classification != SyncClassification::Empty;
      }));
  const bool sync_evidence_complete =
      sync_report_is_complete(header->dropped_window_count, header->unsupported_sync_count,
                              header->malformed_sync_count, header->pending_acquire_collision_count,
                              header->pending_acquire_malformed_count) &&
      summary.malformed_sync_count == 0;

  summary.visible_watchpoint_count = visible.size();
  summary.visible_sync_metadata_count = visible_sync_metadata;
  summary.immediate_conflict_count = header->event_counter;
  summary.claimed_window_count = header->causal_window_count;
  summary.dropped_window_count = header->dropped_window_count;
  summary.saturated_window_count = header->saturated_window_count;
  summary.epoch_exhaustion_count = header->epoch_exhaustion_count;
  summary.unsupported_sync_count = header->unsupported_sync_count;
  summary.malformed_sync_count += header->malformed_sync_count;

  return {
      .watchpoint_slots_examined = watchpoint_slots_examined,
      .pending_release_slots_examined = pending_release_slots_examined,
      .synchronization_evidence_complete = sync_evidence_complete,
      .evidence = std::move(visible),
      .issues = std::move(issues),
  };
}

} // namespace rocjitsu::consan::hook

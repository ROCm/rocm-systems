// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_sampled_model_test_support.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

ConSanMoiSampledSyncPublishResult
consan_moi_sampled_publish_sync_metadata(std::span<ConSanMoiSampledSyncMetadataPacked> slots,
                                         uint32_t selected_slot,
                                         const ConSanMoiSampledSyncMetadata &metadata) {
  const ConSanMoiSampledSyncEncodeResult encoded =
      encode_consan_moi_sampled_sync_metadata(metadata);
  if (encoded.classification != ConSanMoiSampledSyncClassification::Valid)
    return {ConSanMoiSampledSyncPublishOutcome::Rejected, encoded.classification, selected_slot};
  if (selected_slot >= slots.size())
    return {ConSanMoiSampledSyncPublishOutcome::CapacityExhausted,
            ConSanMoiSampledSyncClassification::Valid, selected_slot};

  ConSanMoiSampledSyncMetadataPacked &slot = slots[selected_slot];
  const ConSanMoiSampledSyncDecodeResult existing = decode_consan_moi_sampled_sync_metadata(slot);
  if (existing.classification == ConSanMoiSampledSyncClassification::Empty) {
    slot = encoded.packed;
    return {ConSanMoiSampledSyncPublishOutcome::Published,
            ConSanMoiSampledSyncClassification::Valid, selected_slot};
  }
  if (existing.classification != ConSanMoiSampledSyncClassification::Valid)
    return {ConSanMoiSampledSyncPublishOutcome::MalformedSlot, existing.classification,
            selected_slot};
  if (slot == encoded.packed)
    return {ConSanMoiSampledSyncPublishOutcome::Existing, ConSanMoiSampledSyncClassification::Valid,
            selected_slot};
  return {ConSanMoiSampledSyncPublishOutcome::Collision, ConSanMoiSampledSyncClassification::Valid,
          selected_slot};
}

ConSanMoiSampledPublishResult
consan_moi_sampled_publish_access_records(const ConSanMoiReportHeader &header,
                                          std::span<const ConSanMoiAccessRecord> access_records,
                                          std::span<uint64_t> sampled_watchpoint_entries) {
  auto span_size_u32 = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };
  auto decode_access_kind = [](uint32_t value) {
    switch (static_cast<ConSanMoiShadowAccessKind>(value)) {
    case ConSanMoiShadowAccessKind::Read:
    case ConSanMoiShadowAccessKind::Write:
    case ConSanMoiShadowAccessKind::ReadWrite:
    case ConSanMoiShadowAccessKind::Atomic:
      return static_cast<ConSanMoiShadowAccessKind>(value);
    case ConSanMoiShadowAccessKind::Empty:
      return ConSanMoiShadowAccessKind::Empty;
    }
    return ConSanMoiShadowAccessKind::Empty;
  };

  ConSanMoiSampledPublishResult publish;
  const uint32_t access_count = std::min({header.access_record_count, header.access_record_capacity,
                                          span_size_u32(access_records.size())});
  const uint32_t sampled_capacity = std::min(header.sampled_watchpoint_capacity,
                                             span_size_u32(sampled_watchpoint_entries.size()));

  for (uint32_t i = 0; i < access_count; ++i) {
    ++publish.processed_access_count;
    if (publish.published_entry_count >= sampled_capacity) {
      publish.sampled_capacity_exhausted = true;
      continue;
    }

    const ConSanMoiAccessRecord &record = access_records[i];
    const uint32_t start_byte = record.lds_byte_count != 0
                                    ? record.lds_byte_offset
                                    : record.start_cell * consan_moi_exact_shadow::granule_bytes;
    const uint32_t byte_count = record.lds_byte_count != 0
                                    ? record.lds_byte_count
                                    : record.cell_count * consan_moi_exact_shadow::granule_bytes;
    if (byte_count == 0)
      continue;

    sampled_watchpoint_entries[publish.published_entry_count] =
        pack_consan_moi_sampled_watchpoint_entry(
            decode_access_kind(record.access_kind), record.wave_id, record.epoch,
            static_cast<uint32_t>(record.generation != 0 ? record.generation : header.generation),
            start_byte, byte_count);
    ++publish.published_entry_count;
  }
  return publish;
}

ConSanMoiSampledPublishResult
consan_moi_sampled_publish_causal_windows(const ConSanMoiReportHeader &header,
                                          std::span<const ConSanMoiAccessRecord> access_records,
                                          uint32_t selection_stride, uint32_t selection_offset,
                                          std::span<uint64_t> sampled_watchpoint_entries,
                                          std::span<ConSanMoiSampledCausalWindow> causal_windows) {
  struct WindowKey {
    uint64_t generation = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t epoch = 0;

    bool operator==(const WindowKey &) const = default;
  };
  struct WindowKeyHash {
    size_t operator()(const WindowKey &key) const {
      uint64_t hash = consan_moi_sampled_causal_mix(0x243f6a8885a308d3ull, key.generation);
      hash = consan_moi_sampled_causal_mix(hash, key.x);
      hash = consan_moi_sampled_causal_mix(hash, key.y);
      hash = consan_moi_sampled_causal_mix(hash, key.z);
      return static_cast<size_t>(consan_moi_sampled_causal_mix(hash, key.epoch));
    }
  };
  struct Item {
    ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
    uint32_t owner = 0;
    uint32_t start_byte = 0;
    uint32_t byte_count = 0;
  };
  struct Window {
    WindowKey key;
    std::vector<Item> items;
    bool malformed = false;
  };
  const auto span_size_u32 = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };
  const auto decode_kind = [](uint32_t value) {
    const auto kind = static_cast<ConSanMoiShadowAccessKind>(value);
    return kind == ConSanMoiShadowAccessKind::Read || kind == ConSanMoiShadowAccessKind::Write ||
                   kind == ConSanMoiShadowAccessKind::Atomic
               ? kind
               : ConSanMoiShadowAccessKind::Empty;
  };

  ConSanMoiSampledPublishResult publish;
  if (selection_stride == 0 || selection_stride > 1024u ||
      (selection_stride & (selection_stride - 1u)) != 0 || selection_offset >= selection_stride) {
    publish.invalid_selection = true;
    return publish;
  }

  const uint32_t access_count = std::min({header.access_record_count, header.access_record_capacity,
                                          span_size_u32(access_records.size())});
  const uint32_t entry_capacity = std::min(header.sampled_watchpoint_capacity,
                                           span_size_u32(sampled_watchpoint_entries.size()));
  const uint32_t window_capacity = span_size_u32(causal_windows.size());
  std::vector<Window> windows;
  std::unordered_map<WindowKey, size_t, WindowKeyHash> by_key;
  for (uint32_t index = 0; index < access_count; ++index) {
    ++publish.processed_access_count;
    const ConSanMoiAccessRecord &record = access_records[index];
    const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
    const WindowKey key{generation, record.workgroup_x, record.workgroup_y, record.workgroup_z,
                        record.epoch};
    auto [position, inserted] = by_key.emplace(key, windows.size());
    if (inserted)
      windows.push_back(Window{key, {}, false});
    Window &window = windows[position->second];
    const uint32_t start_byte = record.lds_byte_count != 0
                                    ? record.lds_byte_offset
                                    : record.start_cell * consan_moi_exact_shadow::granule_bytes;
    const uint32_t byte_count = record.lds_byte_count != 0
                                    ? record.lds_byte_count
                                    : record.cell_count * consan_moi_exact_shadow::granule_bytes;
    const ConSanMoiShadowAccessKind kind = decode_kind(record.access_kind);
    const bool malformed =
        generation != header.generation ||
        record.wave_id > consan_moi_sampled_watchpoint::max_owner ||
        record.epoch > consan_moi_sampled_watchpoint::max_epoch ||
        kind == ConSanMoiShadowAccessKind::Empty || byte_count == 0 ||
        byte_count > consan_moi_sampled_watchpoint::max_byte_count ||
        start_byte > consan_moi_sampled_watchpoint::max_start_byte ||
        byte_count > consan_moi_sampled_watchpoint::max_start_byte + 1u - start_byte;
    window.malformed |= malformed;
    window.items.push_back(Item{kind, record.wave_id, start_byte, byte_count});
  }

  for (const Window &window : windows) {
    ++publish.eligible_window_count;
    if (!consan_moi_sampled_causal_window_selected(
            window.key.generation, header.dispatch_id, window.key.x, window.key.y, window.key.z,
            window.key.epoch, selection_stride, selection_offset)) {
      continue;
    }
    ++publish.selected_window_count;
    if (window.malformed) {
      ++publish.malformed_window_count;
      continue;
    }
    if (publish.published_window_count >= window_capacity) {
      publish.window_capacity_exhausted = true;
      continue;
    }
    if (window.items.size() > entry_capacity - publish.published_entry_count) {
      publish.sampled_capacity_exhausted = true;
      continue;
    }
    const uint32_t first_entry = publish.published_entry_count;
    for (const Item &item : window.items) {
      sampled_watchpoint_entries[publish.published_entry_count++] =
          pack_consan_moi_sampled_watchpoint_entry(item.kind, item.owner, window.key.epoch,
                                                   static_cast<uint32_t>(window.key.generation),
                                                   item.start_byte, item.byte_count);
    }
    causal_windows[publish.published_window_count++] = ConSanMoiSampledCausalWindow{
        window.key.generation,
        header.dispatch_id,
        window.key.x,
        window.key.y,
        window.key.z,
        window.key.epoch,
        first_entry,
        static_cast<uint32_t>(window.items.size()),
        static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready),
        0};
  }
  return publish;
}

ConSanMoiSampledReplayResult
consan_moi_sampled_replay_entries(ConSanMoiReportHeader &header,
                                  std::span<const uint64_t> sampled_watchpoint_entries,
                                  std::span<ConSanMoiDiagnosticRecord> diagnostic_records) {
  std::vector<ConSanMoiSampledSnapshotWords> snapshots;
  snapshots.reserve(sampled_watchpoint_entries.size());
  for (uint64_t packed : sampled_watchpoint_entries) {
    const uint32_t low = static_cast<uint32_t>(packed);
    snapshots.push_back({low, static_cast<uint32_t>(packed >> 32u), low});
  }
  return consan_moi_sampled_replay_snapshots(header, snapshots, diagnostic_records);
}

ConSanMoiSampledReplayResult
consan_moi_sampled_replay_snapshots(ConSanMoiReportHeader &header,
                                    std::span<const ConSanMoiSampledSnapshotWords> snapshots,
                                    std::span<ConSanMoiDiagnosticRecord> diagnostic_records) {
  auto span_size_u32 = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };

  ConSanMoiSampledReplayResult replay;
  const uint32_t entry_count =
      std::min(header.sampled_watchpoint_capacity, span_size_u32(snapshots.size()));
  const uint32_t diagnostic_capacity =
      std::min(header.diagnostic_capacity, span_size_u32(diagnostic_records.size()));
  const uint32_t active_generation =
      static_cast<uint32_t>(header.generation) & consan_moi_sampled_watchpoint::max_generation;

  std::vector<std::optional<ConSanMoiSampledWatchpointEntry>> stable_entries(entry_count);

  for (uint32_t i = 0; i < entry_count; ++i) {
    ++replay.processed_entry_count;
    const ConSanMoiSampledSnapshot snapshot =
        classify_consan_moi_sampled_snapshot(snapshots[i], active_generation);
    switch (snapshot.state) {
    case ConSanMoiSampledSnapshotState::Empty:
      ++replay.empty_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::StaleGeneration:
      ++replay.stale_generation_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::IncompletePublication:
      ++replay.incomplete_publication_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::ChangedDuringRead:
      ++replay.changed_during_read_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::Malformed:
      ++replay.malformed_entry_count;
      continue;
    case ConSanMoiSampledSnapshotState::Stable:
      break;
    }
    stable_entries[i] = snapshot.entry;
    const ConSanMoiSampledWatchpointEntry &current = *stable_entries[i];

    for (uint32_t prior_index = 0; prior_index < i; ++prior_index) {
      if (!stable_entries[prior_index] ||
          !consan_moi_sampled_watchpoints_conflict(current, *stable_entries[prior_index]))
        continue;

      replay.conflict = true;
      ConSanMoiDiagnosticRecord diagnostic;
      diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict);
      diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::Sampled);
      diagnostic.generation = current.generation;
      diagnostic.epoch = current.epoch;
      diagnostic.first_owner_id = stable_entries[prior_index]->owner_id;
      diagnostic.second_owner_id = current.owner_id;
      diagnostic.first_access_kind = static_cast<uint32_t>(stable_entries[prior_index]->kind);
      diagnostic.second_access_kind = static_cast<uint32_t>(current.kind);
      if (header.diagnostic_count < diagnostic_capacity) {
        diagnostic_records[header.diagnostic_count] = diagnostic;
        ++header.diagnostic_count;
        ++replay.emitted_diagnostic_count;
      } else {
        replay.diagnostic_capacity_exhausted = true;
      }
      return replay;
    }
  }
  return replay;
}

ConSanMoiSampledReplayResult consan_moi_sampled_replay_causal_windows(
    ConSanMoiReportHeader &header, std::span<const uint64_t> sampled_watchpoint_entries,
    std::span<const ConSanMoiSampledCausalWindow> causal_windows,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records) {
  ConSanMoiSampledReplayResult replay;
  const uint32_t entry_capacity =
      std::min(header.sampled_watchpoint_capacity,
               sampled_watchpoint_entries.size() > std::numeric_limits<uint32_t>::max()
                   ? std::numeric_limits<uint32_t>::max()
                   : static_cast<uint32_t>(sampled_watchpoint_entries.size()));
  uint32_t expected_first = 0;
  std::vector<ConSanMoiSampledWatchpointEntry> decoded;
  decoded.reserve(entry_capacity);
  for (const ConSanMoiSampledCausalWindow &window : causal_windows) {
    if (window.publication_state !=
            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready) ||
        window.generation != header.generation || window.dispatch_id != header.dispatch_id ||
        window.epoch > consan_moi_sampled_watchpoint::max_epoch || window.entry_count == 0 ||
        window.first_entry != expected_first ||
        window.entry_count > entry_capacity - expected_first) {
      replay.invalid_causal_metadata = true;
      return replay;
    }
    expected_first += window.entry_count;
  }
  for (uint32_t index = 0; index < expected_first; ++index) {
    const ConSanMoiSampledWatchpointEntry entry =
        decode_consan_moi_sampled_watchpoint_entry(sampled_watchpoint_entries[index]);
    if (!entry.valid || entry.consumed ||
        (entry.kind != ConSanMoiShadowAccessKind::Read &&
         entry.kind != ConSanMoiShadowAccessKind::Write &&
         entry.kind != ConSanMoiShadowAccessKind::Atomic) ||
        entry.generation != (static_cast<uint32_t>(header.generation) &
                             consan_moi_sampled_watchpoint::max_generation)) {
      replay.invalid_causal_metadata = true;
      return replay;
    }
    decoded.push_back(entry);
  }
  for (const ConSanMoiSampledCausalWindow &window : causal_windows) {
    for (uint32_t index = window.first_entry; index < window.first_entry + window.entry_count;
         ++index) {
      if (decoded[index].epoch != window.epoch) {
        replay.invalid_causal_metadata = true;
        return replay;
      }
    }
  }

  const uint32_t diagnostic_capacity = std::min(
      header.diagnostic_capacity, diagnostic_records.size() > std::numeric_limits<uint32_t>::max()
                                      ? std::numeric_limits<uint32_t>::max()
                                      : static_cast<uint32_t>(diagnostic_records.size()));
  for (const ConSanMoiSampledCausalWindow &window : causal_windows) {
    ++replay.processed_window_count;
    for (uint32_t current_index = window.first_entry;
         current_index < window.first_entry + window.entry_count; ++current_index) {
      ++replay.processed_entry_count;
      const ConSanMoiSampledWatchpointEntry &current = decoded[current_index];
      for (uint32_t prior_index = window.first_entry; prior_index < current_index; ++prior_index) {
        const ConSanMoiSampledWatchpointEntry &prior = decoded[prior_index];
        if (!consan_moi_sampled_watchpoints_conflict(current, prior))
          continue;
        replay.conflict = true;
        ConSanMoiDiagnosticRecord diagnostic;
        diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict);
        diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::Sampled);
        diagnostic.generation = current.generation;
        diagnostic.epoch = current.epoch;
        diagnostic.first_owner_id = prior.owner_id;
        diagnostic.second_owner_id = current.owner_id;
        diagnostic.first_access_kind = static_cast<uint32_t>(prior.kind);
        diagnostic.second_access_kind = static_cast<uint32_t>(current.kind);
        if (header.diagnostic_count < diagnostic_capacity) {
          diagnostic_records[header.diagnostic_count++] = diagnostic;
          ++replay.emitted_diagnostic_count;
        } else {
          replay.diagnostic_capacity_exhausted = true;
        }
        return replay;
      }
    }
  }
  return replay;
}

ConSanMoiSampledClaimResult
consan_moi_sampled_begin_causal_claim(std::span<ConSanMoiSampledCausalWindow> windows,
                                      const ConSanMoiSampledCausalKey &key) {
  ConSanMoiSampledClaimResult result;
  if (windows.empty() || key.epoch > consan_moi_sampled_watchpoint::max_epoch)
    return result;

  uint64_t hash = consan_moi_sampled_causal_mix(0x243f6a8885a308d3ull, key.generation);
  hash = consan_moi_sampled_causal_mix(hash, key.dispatch_id);
  hash = consan_moi_sampled_causal_mix(hash, key.workgroup_x);
  hash = consan_moi_sampled_causal_mix(hash, key.workgroup_y);
  hash = consan_moi_sampled_causal_mix(hash, key.workgroup_z);
  hash = consan_moi_sampled_causal_mix(hash, key.epoch);
  hash = consan_moi_sampled_causal_mix(hash, key.cluster_workgroup_id);
  const uint32_t start = static_cast<uint32_t>(hash % windows.size());
  for (uint32_t probe = 0; probe < windows.size(); ++probe) {
    result.slot = (start + probe) % static_cast<uint32_t>(windows.size());
    ConSanMoiSampledCausalWindow &window = windows[result.slot];
    std::atomic_ref<uint32_t> state(window.publication_state);
    uint32_t observed = state.load(std::memory_order_acquire);
    if (observed == static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing)) {
      result.outcome = ConSanMoiSampledClaimOutcome::Busy;
      return result;
    }
    if (observed == static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)) {
      const ConSanMoiSampledCausalKey existing{
          window.generation,  window.dispatch_id, window.workgroup_x,         window.workgroup_y,
          window.workgroup_z, window.epoch,       window.cluster_workgroup_id};
      if (existing == key) {
        result.outcome = ConSanMoiSampledClaimOutcome::Existing;
        return result;
      }
      ++result.collision_count;
      continue;
    }
    if (observed != static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Empty)) {
      ++result.malformed_slot_count;
      continue;
    }
    if (!state.compare_exchange_strong(
            observed, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      result.outcome = ConSanMoiSampledClaimOutcome::Busy;
      return result;
    }
    result.outcome = ConSanMoiSampledClaimOutcome::Claimed;
    return result;
  }
  result.outcome = ConSanMoiSampledClaimOutcome::CapacityExhausted;
  return result;
}

bool consan_moi_sampled_commit_causal_claim(ConSanMoiSampledCausalWindow &window,
                                            const ConSanMoiSampledCausalKey &key,
                                            uint32_t first_entry, uint32_t entry_count) {
  std::atomic_ref<uint32_t> state(window.publication_state);
  if (state.load(std::memory_order_acquire) !=
          static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing) ||
      entry_count == 0)
    return false;
  window.generation = key.generation;
  window.dispatch_id = key.dispatch_id;
  window.workgroup_x = key.workgroup_x;
  window.workgroup_y = key.workgroup_y;
  window.workgroup_z = key.workgroup_z;
  window.epoch = key.epoch;
  window.first_entry = first_entry;
  window.entry_count = entry_count;
  window.cluster_workgroup_id = key.cluster_workgroup_id;
  state.store(static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready),
              std::memory_order_release);
  return true;
}

bool consan_moi_sampled_abort_causal_claim(ConSanMoiSampledCausalWindow &window) {
  std::atomic_ref<uint32_t> state(window.publication_state);
  uint32_t expected = static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing);
  return state.compare_exchange_strong(
      expected, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Malformed),
      std::memory_order_release, std::memory_order_relaxed);
}

} // namespace rocjitsu

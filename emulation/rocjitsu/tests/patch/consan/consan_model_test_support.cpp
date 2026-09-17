// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_model_test_support.h"

#include <atomic>

namespace rocjitsu::consan {

SyncPublishResult publish_sync_metadata(std::span<SyncMetadataPacked> slots, uint32_t selected_slot,
                                        const SyncMetadata &metadata) {
  const SyncEncodeResult encoded = encode_sync_metadata(metadata);
  if (encoded.classification != SyncClassification::Valid)
    return {SyncPublishOutcome::Rejected, encoded.classification, selected_slot};
  if (selected_slot >= slots.size())
    return {SyncPublishOutcome::CapacityExhausted, SyncClassification::Valid, selected_slot};

  SyncMetadataPacked &slot = slots[selected_slot];
  const SyncDecodeResult existing = decode_sync_metadata(slot);
  if (existing.classification == SyncClassification::Empty) {
    slot = encoded.packed;
    return {SyncPublishOutcome::Published, SyncClassification::Valid, selected_slot};
  }
  if (existing.classification != SyncClassification::Valid)
    return {SyncPublishOutcome::MalformedSlot, existing.classification, selected_slot};
  if (slot == encoded.packed)
    return {SyncPublishOutcome::Existing, SyncClassification::Valid, selected_slot};
  return {SyncPublishOutcome::Collision, SyncClassification::Valid, selected_slot};
}

ClaimResult begin_causal_claim(std::span<CausalWindow> windows, const CausalKey &key) {
  ClaimResult result;
  if (windows.empty() || key.epoch > watchpoint::max_epoch)
    return result;

  uint64_t hash = causal_mix(0x243f6a8885a308d3ull, key.generation);
  hash = causal_mix(hash, key.dispatch_id);
  hash = causal_mix(hash, key.workgroup_x);
  hash = causal_mix(hash, key.workgroup_y);
  hash = causal_mix(hash, key.workgroup_z);
  hash = causal_mix(hash, key.epoch);
  hash = causal_mix(hash, key.cluster_workgroup_id);
  const uint32_t start = static_cast<uint32_t>(hash % windows.size());
  for (uint32_t probe = 0; probe < windows.size(); ++probe) {
    result.slot = (start + probe) % static_cast<uint32_t>(windows.size());
    CausalWindow &window = windows[result.slot];
    std::atomic_ref<uint32_t> state(window.publication_state);
    uint32_t observed = state.load(std::memory_order_acquire);
    if (observed == static_cast<uint32_t>(CausalPublicationState::Publishing)) {
      result.outcome = ClaimOutcome::Busy;
      return result;
    }
    if (observed == static_cast<uint32_t>(CausalPublicationState::Ready)) {
      const CausalKey existing{window.generation,          window.dispatch_id, window.workgroup_x,
                               window.workgroup_y,         window.workgroup_z, window.epoch,
                               window.cluster_workgroup_id};
      if (existing == key) {
        result.outcome = ClaimOutcome::Existing;
        return result;
      }
      ++result.collision_count;
      continue;
    }
    if (observed != static_cast<uint32_t>(CausalPublicationState::Empty)) {
      ++result.malformed_slot_count;
      continue;
    }
    if (!state.compare_exchange_strong(observed,
                                       static_cast<uint32_t>(CausalPublicationState::Publishing),
                                       std::memory_order_acq_rel, std::memory_order_acquire)) {
      result.outcome = ClaimOutcome::Busy;
      return result;
    }
    result.outcome = ClaimOutcome::Claimed;
    return result;
  }
  result.outcome = ClaimOutcome::CapacityExhausted;
  return result;
}

bool commit_causal_claim(CausalWindow &window, const CausalKey &key, uint32_t first_entry,
                         uint32_t entry_count) {
  std::atomic_ref<uint32_t> state(window.publication_state);
  if (state.load(std::memory_order_acquire) !=
          static_cast<uint32_t>(CausalPublicationState::Publishing) ||
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
  state.store(static_cast<uint32_t>(CausalPublicationState::Ready), std::memory_order_release);
  return true;
}

bool abort_causal_claim(CausalWindow &window) {
  std::atomic_ref<uint32_t> state(window.publication_state);
  uint32_t expected = static_cast<uint32_t>(CausalPublicationState::Publishing);
  return state.compare_exchange_strong(expected,
                                       static_cast<uint32_t>(CausalPublicationState::Malformed),
                                       std::memory_order_release, std::memory_order_relaxed);
}

} // namespace rocjitsu::consan

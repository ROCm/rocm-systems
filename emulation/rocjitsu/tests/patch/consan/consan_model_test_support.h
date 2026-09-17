// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_report.h"

namespace rocjitsu::consan {

struct CausalKey {
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t epoch = 0;
  uint32_t cluster_workgroup_id = 0;

  [[nodiscard]] constexpr bool operator==(const CausalKey &) const = default;
};

enum class ClaimOutcome : uint32_t {
  Claimed,
  Existing,
  Busy,
  CapacityExhausted,
  Invalid,
};

struct ClaimResult {
  ClaimOutcome outcome = ClaimOutcome::Invalid;
  uint32_t slot = 0;
  uint32_t collision_count = 0;
  uint32_t malformed_slot_count = 0;
};

enum class SyncPublishOutcome : uint8_t {
  Published,
  Existing,
  Collision,
  CapacityExhausted,
  Rejected,
  MalformedSlot,
};

struct SyncPublishResult {
  SyncPublishOutcome outcome = SyncPublishOutcome::Rejected;
  SyncClassification classification = SyncClassification::Malformed;
  uint32_t slot = 0;
};

/// Independent host reference for the device-side causal-window selector.
[[nodiscard]] constexpr uint64_t causal_mix(uint64_t hash, uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  hash ^= hash >> 30u;
  hash *= 0xbf58476d1ce4e5b9ull;
  hash ^= hash >> 27u;
  hash *= 0x94d049bb133111ebull;
  return hash ^ (hash >> 31u);
}

/// Independent host encoder for the complete device-published ConSan
/// watchpoint ABI. The byte-count field primitive remains shared with the
/// production emitter.
[[nodiscard]] constexpr uint64_t pack_watchpoint_entry(ShadowAccessKind kind, uint32_t owner_id,
                                                       uint32_t epoch, uint32_t generation,
                                                       uint32_t start_byte, uint32_t byte_count,
                                                       bool consumed = false) {
  return watchpoint::valid_mask | (consumed ? watchpoint::consumed_mask : uint64_t{0}) |
         ((static_cast<uint64_t>(kind) & low_bit_mask(watchpoint::access_kind_bits))
          << watchpoint::access_kind_shift) |
         ((static_cast<uint64_t>(owner_id) & low_bit_mask(watchpoint::owner_bits))
          << watchpoint::owner_shift) |
         ((static_cast<uint64_t>(epoch) & low_bit_mask(watchpoint::epoch_bits))
          << watchpoint::epoch_shift) |
         ((static_cast<uint64_t>(generation) & low_bit_mask(watchpoint::generation_bits))
          << watchpoint::generation_shift) |
         ((static_cast<uint64_t>(start_byte) & low_bit_mask(watchpoint::start_byte_bits))
          << watchpoint::start_byte_shift) |
         ((static_cast<uint64_t>(encode_byte_count(byte_count)) &
           low_bit_mask(watchpoint::count_bits))
          << watchpoint::count_shift);
}

[[nodiscard]] SyncPublishResult publish_sync_metadata(std::span<SyncMetadataPacked> slots,
                                                      uint32_t selected_slot,
                                                      const SyncMetadata &metadata);

[[nodiscard]] ClaimResult begin_causal_claim(std::span<CausalWindow> windows, const CausalKey &key);

[[nodiscard]] bool commit_causal_claim(CausalWindow &window, const CausalKey &key,
                                       uint32_t first_entry, uint32_t entry_count);

[[nodiscard]] bool abort_causal_claim(CausalWindow &window);

} // namespace rocjitsu::consan

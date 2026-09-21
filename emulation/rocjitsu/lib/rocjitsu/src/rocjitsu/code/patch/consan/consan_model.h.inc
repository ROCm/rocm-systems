// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

namespace watchpoint {

inline constexpr uint32_t valid_shift = 0;
inline constexpr uint32_t consumed_shift = 1;
inline constexpr uint32_t access_kind_shift = 2;
inline constexpr uint32_t access_kind_bits = 3;
inline constexpr uint32_t owner_shift = 5;
inline constexpr uint32_t owner_bits = 10;
inline constexpr uint32_t epoch_shift = 15;
inline constexpr uint32_t epoch_bits = 10;
inline constexpr uint32_t generation_shift = 25;
// Keep publication a single 64-bit atomic exchange, but retain the full
// 20-bit LDS byte address (including gfx1250's LDS above 256 KiB). The compact
// generation is only a tag: causal windows and atomic attachments validate
// the full 64-bit generation before accepting evidence. Never use this tag
// alone to establish identity across report allocations.
// Four-byte cells are insufficient here: cooperative kernels commonly assign
// adjacent FP16 halves of one dword to different waves.
inline constexpr uint32_t generation_bits = 14;
inline constexpr uint32_t start_byte_shift = generation_shift + generation_bits;
inline constexpr uint32_t start_byte_bits = 20;
inline constexpr uint32_t count_shift = 59;
inline constexpr uint32_t count_bits = 5;
static_assert(start_byte_shift + start_byte_bits == count_shift);
static_assert(count_shift + count_bits == 64);

inline constexpr uint64_t valid_mask = uint64_t{1} << valid_shift;
inline constexpr uint64_t consumed_mask = uint64_t{1} << consumed_shift;
inline constexpr uint64_t access_kind_mask = bit_field_mask(access_kind_shift, access_kind_bits);
inline constexpr uint64_t owner_mask = bit_field_mask(owner_shift, owner_bits);
inline constexpr uint64_t epoch_mask = bit_field_mask(epoch_shift, epoch_bits);
inline constexpr uint64_t generation_mask = bit_field_mask(generation_shift, generation_bits);
inline constexpr uint64_t start_byte_mask = bit_field_mask(start_byte_shift, start_byte_bits);
inline constexpr uint64_t count_mask = bit_field_mask(count_shift, count_bits);
inline constexpr uint64_t epoch_generation_mask = epoch_mask | generation_mask;

inline constexpr uint32_t max_owner = (1u << owner_bits) - 1u;
inline constexpr uint32_t max_epoch = (1u << epoch_bits) - 1u;
// Persistent owner state has a full dword. This sentinel must never be packed
// into an evidence entry: doing so would alias the first epoch.
inline constexpr uint32_t exhausted_epoch = max_epoch + 1u;
inline constexpr uint32_t max_generation = (1u << generation_bits) - 1u;

} // namespace watchpoint

struct WatchpointEntry {
  bool valid = false;
  bool consumed = false;
  ShadowAccessKind kind = ShadowAccessKind::Empty;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t generation = 0;
  uint32_t start_byte = 0;
  uint32_t byte_count = 0;
};

struct AtomicAttachmentKey {
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t epoch = 0;
  uint32_t owner_id = 0;
  uint32_t cluster_workgroup_id = 0;
};

/// Host mirror of the device-side fail-closed attachment predicate.
[[nodiscard]] bool atomic_attachment_matches(const CausalWindow &window, uint64_t packed_watchpoint,
                                             uint32_t slot, const AtomicAttachmentKey &key);

// Device probes publish sampled entries with one aligned 64-bit atomic
// exchange. Fine-grained host readers still sample the low word on both sides
// of the high word so that a concurrent replacement is classified instead of
// being mistaken for a stable entry.
struct SnapshotWords {
  uint32_t low_before = 0;
  uint32_t high = 0;
  uint32_t low_after = 0;
};

enum class SnapshotState : uint8_t {
  Empty,
  Stable,
  StaleGeneration,
  IncompletePublication,
  ChangedDuringRead,
  Malformed,
};

struct Snapshot {
  SnapshotState state = SnapshotState::Empty;
  WatchpointEntry entry;
};

[[nodiscard]] constexpr uint32_t encode_byte_count(uint32_t byte_count) {
  return byte_count == 0 ? 0 : byte_count - 1u;
}

[[nodiscard]] constexpr WatchpointEntry decode_watchpoint_entry(uint64_t value) {
  return WatchpointEntry{
      (value & watchpoint::valid_mask) != 0,
      (value & watchpoint::consumed_mask) != 0,
      static_cast<ShadowAccessKind>((value & watchpoint::access_kind_mask) >>
                                    watchpoint::access_kind_shift),
      static_cast<uint32_t>((value & watchpoint::owner_mask) >> watchpoint::owner_shift),
      static_cast<uint32_t>((value & watchpoint::epoch_mask) >> watchpoint::epoch_shift),
      static_cast<uint32_t>((value & watchpoint::generation_mask) >> watchpoint::generation_shift),
      static_cast<uint32_t>((value & watchpoint::start_byte_mask) >> watchpoint::start_byte_shift),
      static_cast<uint32_t>(((value & watchpoint::count_mask) >> watchpoint::count_shift) + 1u),
  };
}

[[nodiscard]] constexpr Snapshot classify_snapshot(SnapshotWords words,
                                                   uint32_t active_generation) {
  if (words.low_before != words.low_after)
    return {SnapshotState::ChangedDuringRead, {}};

  const uint64_t packed =
      static_cast<uint64_t>(words.low_after) | (static_cast<uint64_t>(words.high) << 32u);
  const WatchpointEntry entry = decode_watchpoint_entry(packed);
  if (!entry.valid) {
    return {packed == 0 ? SnapshotState::Empty : SnapshotState::IncompletePublication, entry};
  }
  if (entry.generation != (active_generation & watchpoint::max_generation))
    return {SnapshotState::StaleGeneration, entry};
  if (entry.kind != ShadowAccessKind::Read && entry.kind != ShadowAccessKind::Write &&
      entry.kind != ShadowAccessKind::Atomic)
    return {SnapshotState::Malformed, entry};
  return {SnapshotState::Stable, entry};
}

[[nodiscard]] constexpr bool watchpoints_conflict(const WatchpointEntry &current,
                                                  const WatchpointEntry &prior) {
  return current.valid && prior.valid && !prior.consumed && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         shadow_kind_conflicts(current.kind, prior.kind) &&
         ranges_overlap(current.start_byte, current.byte_count, prior.start_byte, prior.byte_count);
}

/// Classifies the semantic host representation without truncating invalid
/// enum or range values into the packed descriptor.
[[nodiscard]] SyncClassification classify_sync_metadata(const SyncMetadata &metadata);

[[nodiscard]] SyncEncodeResult encode_sync_metadata(const SyncMetadata &metadata);

[[nodiscard]] SyncDecodeResult decode_sync_metadata(const SyncMetadataPacked &packed);

/// Returns true only for an exact, complete workgroup barrier sequence that
/// ConSan may use to advance its selected causal epoch. This deliberately
/// excludes individual signal/wait halves, dynamic IDs, cyclic forms, scalar
/// clauses, unresolved owners, and ambiguous membership. Shared physical
/// sequences may have several proven kernel owners; lowering validates their
/// common ABI inputs and requires a preceding selected window for every owner.
[[nodiscard]] bool qualifies_barrier_sequence(const SyncSequence &sequence, bool owner_proven);

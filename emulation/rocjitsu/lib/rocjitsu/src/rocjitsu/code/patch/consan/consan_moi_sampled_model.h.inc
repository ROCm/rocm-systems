// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

namespace consan_moi_sampled_watchpoint {

inline constexpr uint32_t valid_shift = 0;
inline constexpr uint32_t consumed_shift = 1;
inline constexpr uint32_t access_kind_shift = 2;
inline constexpr uint32_t access_kind_bits = 3;
inline constexpr uint32_t owner_shift = 5;
inline constexpr uint32_t owner_bits = 10;
inline constexpr uint32_t epoch_shift = 15;
inline constexpr uint32_t epoch_bits = 10;
inline constexpr uint32_t generation_shift = 25;
// A sampled entry is published with one 64-bit atomic exchange. Eighteen
// generation bits still distinguish 262,144 report generations in one
// process, while leaving enough space to retain an exact LDS byte range.
// Four-byte cells are insufficient here: cooperative kernels commonly assign
// adjacent FP16 halves of one dword to different waves.
inline constexpr uint32_t generation_bits = 18;
inline constexpr uint32_t start_byte_shift = 43;
inline constexpr uint32_t start_byte_bits = 16;
inline constexpr uint32_t count_shift = 59;
inline constexpr uint32_t count_bits = 5;

inline constexpr uint64_t valid_mask = uint64_t{1} << valid_shift;
inline constexpr uint64_t consumed_mask = uint64_t{1} << consumed_shift;
inline constexpr uint64_t access_kind_mask =
    consan_moi_bit_field_mask(access_kind_shift, access_kind_bits);
inline constexpr uint64_t owner_mask = consan_moi_bit_field_mask(owner_shift, owner_bits);
inline constexpr uint64_t epoch_mask = consan_moi_bit_field_mask(epoch_shift, epoch_bits);
inline constexpr uint64_t generation_mask =
    consan_moi_bit_field_mask(generation_shift, generation_bits);
inline constexpr uint64_t start_byte_mask =
    consan_moi_bit_field_mask(start_byte_shift, start_byte_bits);
inline constexpr uint64_t count_mask = consan_moi_bit_field_mask(count_shift, count_bits);
inline constexpr uint64_t epoch_generation_mask = epoch_mask | generation_mask;

inline constexpr uint32_t max_owner = (1u << owner_bits) - 1u;
inline constexpr uint32_t max_epoch = (1u << epoch_bits) - 1u;
inline constexpr uint32_t max_generation = (1u << generation_bits) - 1u;
inline constexpr uint32_t max_start_byte = (1u << start_byte_bits) - 1u;
inline constexpr uint32_t max_byte_count = 1u << count_bits;

} // namespace consan_moi_sampled_watchpoint

[[nodiscard]] constexpr uint64_t consan_moi_sampled_causal_mix(uint64_t hash, uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  hash ^= hash >> 30u;
  hash *= 0xbf58476d1ce4e5b9ull;
  hash ^= hash >> 27u;
  hash *= 0x94d049bb133111ebull;
  return hash ^ (hash >> 31u);
}

[[nodiscard]] constexpr bool consan_moi_sampled_causal_window_selected(
    uint64_t generation, uint64_t dispatch_id, uint32_t workgroup_x, uint32_t workgroup_y,
    uint32_t workgroup_z, uint32_t epoch, uint32_t stride, uint32_t offset) {
  if (stride == 0 || stride > 1024u || (stride & (stride - 1u)) != 0 || offset >= stride)
    return false;
  uint64_t hash = consan_moi_sampled_causal_mix(0x243f6a8885a308d3ull, generation);
  hash = consan_moi_sampled_causal_mix(hash, dispatch_id);
  hash = consan_moi_sampled_causal_mix(hash, workgroup_x);
  hash = consan_moi_sampled_causal_mix(hash, workgroup_y);
  hash = consan_moi_sampled_causal_mix(hash, workgroup_z);
  hash = consan_moi_sampled_causal_mix(hash, epoch);
  return (hash & (stride - 1u)) == offset;
}

struct ConSanMoiSampledWatchpointEntry {
  bool valid = false;
  bool consumed = false;
  ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t generation = 0;
  uint32_t start_byte = 0;
  uint32_t byte_count = 0;
};

struct ConSanMoiSampledAtomicAttachmentKey {
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
[[nodiscard]] bool
consan_moi_sampled_atomic_attachment_matches(const ConSanMoiSampledCausalWindow &window,
                                             uint64_t packed_watchpoint, uint32_t slot,
                                             const ConSanMoiSampledAtomicAttachmentKey &key);

// Device probes publish sampled entries with one aligned 64-bit atomic
// exchange. Fine-grained host readers still sample the low word on both sides
// of the high word so that a concurrent replacement is classified instead of
// being mistaken for a stable entry.
struct ConSanMoiSampledSnapshotWords {
  uint32_t low_before = 0;
  uint32_t high = 0;
  uint32_t low_after = 0;
};

enum class ConSanMoiSampledSnapshotState : uint8_t {
  Empty,
  Stable,
  StaleGeneration,
  IncompletePublication,
  ChangedDuringRead,
  Malformed,
};

struct ConSanMoiSampledSnapshot {
  ConSanMoiSampledSnapshotState state = ConSanMoiSampledSnapshotState::Empty;
  ConSanMoiSampledWatchpointEntry entry;
};

[[nodiscard]] constexpr uint32_t encode_consan_moi_sampled_byte_count(uint32_t byte_count) {
  return byte_count == 0 ? 0 : byte_count - 1u;
}

[[nodiscard]] constexpr uint64_t
pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind kind, uint32_t owner_id,
                                         uint32_t epoch, uint32_t generation, uint32_t start_byte,
                                         uint32_t byte_count, bool consumed = false) {
  return consan_moi_sampled_watchpoint::valid_mask |
         (consumed ? consan_moi_sampled_watchpoint::consumed_mask : uint64_t{0}) |
         ((static_cast<uint64_t>(kind) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::access_kind_bits))
          << consan_moi_sampled_watchpoint::access_kind_shift) |
         ((static_cast<uint64_t>(owner_id) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::owner_bits))
          << consan_moi_sampled_watchpoint::owner_shift) |
         ((static_cast<uint64_t>(epoch) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::epoch_bits))
          << consan_moi_sampled_watchpoint::epoch_shift) |
         ((static_cast<uint64_t>(generation) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::generation_bits))
          << consan_moi_sampled_watchpoint::generation_shift) |
         ((static_cast<uint64_t>(start_byte) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::start_byte_bits))
          << consan_moi_sampled_watchpoint::start_byte_shift) |
         ((static_cast<uint64_t>(encode_consan_moi_sampled_byte_count(byte_count)) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::count_bits))
          << consan_moi_sampled_watchpoint::count_shift);
}

[[nodiscard]] constexpr ConSanMoiSampledWatchpointEntry
decode_consan_moi_sampled_watchpoint_entry(uint64_t value) {
  return ConSanMoiSampledWatchpointEntry{
      (value & consan_moi_sampled_watchpoint::valid_mask) != 0,
      (value & consan_moi_sampled_watchpoint::consumed_mask) != 0,
      static_cast<ConSanMoiShadowAccessKind>(
          (value & consan_moi_sampled_watchpoint::access_kind_mask) >>
          consan_moi_sampled_watchpoint::access_kind_shift),
      static_cast<uint32_t>((value & consan_moi_sampled_watchpoint::owner_mask) >>
                            consan_moi_sampled_watchpoint::owner_shift),
      static_cast<uint32_t>((value & consan_moi_sampled_watchpoint::epoch_mask) >>
                            consan_moi_sampled_watchpoint::epoch_shift),
      static_cast<uint32_t>((value & consan_moi_sampled_watchpoint::generation_mask) >>
                            consan_moi_sampled_watchpoint::generation_shift),
      static_cast<uint32_t>((value & consan_moi_sampled_watchpoint::start_byte_mask) >>
                            consan_moi_sampled_watchpoint::start_byte_shift),
      static_cast<uint32_t>(((value & consan_moi_sampled_watchpoint::count_mask) >>
                             consan_moi_sampled_watchpoint::count_shift) +
                            1u),
  };
}

[[nodiscard]] constexpr ConSanMoiSampledSnapshot
classify_consan_moi_sampled_snapshot(ConSanMoiSampledSnapshotWords words,
                                     uint32_t active_generation) {
  if (words.low_before != words.low_after)
    return {ConSanMoiSampledSnapshotState::ChangedDuringRead, {}};

  const uint64_t packed =
      static_cast<uint64_t>(words.low_after) | (static_cast<uint64_t>(words.high) << 32u);
  const ConSanMoiSampledWatchpointEntry entry = decode_consan_moi_sampled_watchpoint_entry(packed);
  if (!entry.valid) {
    return {packed == 0 ? ConSanMoiSampledSnapshotState::Empty
                        : ConSanMoiSampledSnapshotState::IncompletePublication,
            entry};
  }
  if (entry.generation != (active_generation & consan_moi_sampled_watchpoint::max_generation))
    return {ConSanMoiSampledSnapshotState::StaleGeneration, entry};
  if (entry.kind != ConSanMoiShadowAccessKind::Read &&
      entry.kind != ConSanMoiShadowAccessKind::Write &&
      entry.kind != ConSanMoiShadowAccessKind::Atomic)
    return {ConSanMoiSampledSnapshotState::Malformed, entry};
  return {ConSanMoiSampledSnapshotState::Stable, entry};
}

[[nodiscard]] constexpr bool
consan_moi_sampled_watchpoints_conflict(const ConSanMoiSampledWatchpointEntry &current,
                                        const ConSanMoiSampledWatchpointEntry &prior) {
  return current.valid && prior.valid && !prior.consumed && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         consan_moi_shadow_kind_conflicts(current.kind, prior.kind) &&
         consan_moi_ranges_overlap(current.start_byte, current.byte_count, prior.start_byte,
                                   prior.byte_count);
}

/// Classifies the semantic host representation without truncating invalid
/// enum or range values into the packed descriptor.
[[nodiscard]] ConSanMoiSampledSyncClassification
classify_consan_moi_sampled_sync_metadata(const ConSanMoiSampledSyncMetadata &metadata);

[[nodiscard]] ConSanMoiSampledSyncEncodeResult
encode_consan_moi_sampled_sync_metadata(const ConSanMoiSampledSyncMetadata &metadata);

[[nodiscard]] ConSanMoiSampledSyncDecodeResult
decode_consan_moi_sampled_sync_metadata(const ConSanMoiSampledSyncMetadataPacked &packed);

/// Returns true only for an exact, complete workgroup barrier sequence that
/// Sampled may use to advance its selected causal epoch. This deliberately
/// excludes individual signal/wait halves, dynamic IDs, cyclic forms, scalar
/// clauses, unresolved owners, and ambiguous membership. Shared physical
/// sequences may have several proven kernel owners; lowering validates their
/// common ABI inputs and requires a preceding selected window for every owner.
[[nodiscard]] bool
consan_moi_sampled_qualifies_barrier_sequence(const ConSanSyncSequence &sequence);

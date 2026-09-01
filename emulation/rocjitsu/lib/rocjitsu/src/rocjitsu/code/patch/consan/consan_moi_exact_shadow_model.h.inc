// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

namespace consan_moi_exact_shadow {

inline constexpr uint32_t access_kind_shift = 0;
inline constexpr uint32_t access_kind_bits = 3;
inline constexpr uint32_t owner_shift = 3;
inline constexpr uint32_t owner_bits = 10;
inline constexpr uint32_t epoch_shift = 13;
inline constexpr uint32_t epoch_bits = 10;
inline constexpr uint32_t generation_shift = 23;
inline constexpr uint32_t generation_bits = 20;
inline constexpr uint32_t instruction_offset_shift = 43;
inline constexpr uint32_t instruction_offset_bits = 21;

inline constexpr uint64_t access_kind_mask =
    consan_moi_bit_field_mask(access_kind_shift, access_kind_bits);
inline constexpr uint64_t owner_mask = consan_moi_bit_field_mask(owner_shift, owner_bits);
inline constexpr uint64_t epoch_mask = consan_moi_bit_field_mask(epoch_shift, epoch_bits);
inline constexpr uint64_t generation_mask =
    consan_moi_bit_field_mask(generation_shift, generation_bits);
inline constexpr uint64_t instruction_offset_mask =
    consan_moi_bit_field_mask(instruction_offset_shift, instruction_offset_bits);
inline constexpr uint64_t epoch_generation_mask = epoch_mask | generation_mask;

inline constexpr uint32_t max_owner = (1u << owner_bits) - 1u;
inline constexpr uint32_t max_epoch = (1u << epoch_bits) - 1u;
inline constexpr uint32_t max_generation = (1u << generation_bits) - 1u;
inline constexpr uint32_t max_instruction_offset = (1u << instruction_offset_bits) - 1u;

// A workgroup-local shadow does not need the generation field used to
// disambiguate process-global cells. gfx1250 compact local cells retain the
// canonical kind/owner/epoch low 23 bits and use the remaining nine bits for
// an object-unique site token. Token zero remains the all-zero empty state.
inline constexpr uint32_t compact_token_shift = generation_shift;
inline constexpr uint32_t compact_token_bits = 9;
inline constexpr uint32_t max_compact_token = (1u << compact_token_bits) - 1u;
inline constexpr uint32_t compact_diagnostic_token_tag = 0x80000000u;

} // namespace consan_moi_exact_shadow

namespace consan_moi_exact_byte_cell {

static_assert(consan_moi_shadow_cell::granule_bytes == 4u,
              "exact byte-cell field widths and lookup packing require four-byte cells");
inline constexpr uint32_t byte_mask_shift = 0;
inline constexpr uint32_t byte_mask_bits = consan_moi_shadow_cell::granule_bytes;
inline constexpr uint32_t byte_offset_shift = byte_mask_shift + byte_mask_bits;
inline constexpr uint32_t byte_offset_bits = 2;
inline constexpr uint32_t byte_end_minus_one_shift = byte_offset_shift + byte_offset_bits;
inline constexpr uint32_t byte_end_minus_one_bits = 2;
inline constexpr uint32_t lane_plus_one_shift = byte_end_minus_one_shift + byte_end_minus_one_bits;
inline constexpr uint32_t lane_plus_one_bits = 7;
inline constexpr uint32_t access_byte_count_minus_one_shift =
    lane_plus_one_shift + lane_plus_one_bits;
inline constexpr uint32_t access_byte_count_minus_one_bits = 4;
inline constexpr uint32_t relative_cell_index_shift =
    access_byte_count_minus_one_shift + access_byte_count_minus_one_bits;
inline constexpr uint32_t relative_cell_index_bits = 4;
inline constexpr uint32_t byte_mask_mask =
    static_cast<uint32_t>(consan_moi_bit_field_mask(byte_mask_shift, byte_mask_bits));
inline constexpr uint32_t byte_offset_mask =
    static_cast<uint32_t>(consan_moi_bit_field_mask(byte_offset_shift, byte_offset_bits));
inline constexpr uint32_t byte_end_minus_one_mask = static_cast<uint32_t>(
    consan_moi_bit_field_mask(byte_end_minus_one_shift, byte_end_minus_one_bits));
inline constexpr uint32_t lane_plus_one_mask =
    static_cast<uint32_t>(consan_moi_bit_field_mask(lane_plus_one_shift, lane_plus_one_bits));
inline constexpr uint32_t access_byte_count_minus_one_mask = static_cast<uint32_t>(
    consan_moi_bit_field_mask(access_byte_count_minus_one_shift, access_byte_count_minus_one_bits));
inline constexpr uint32_t relative_cell_index_mask = static_cast<uint32_t>(
    consan_moi_bit_field_mask(relative_cell_index_shift, relative_cell_index_bits));
inline constexpr uint32_t packed_mask = byte_mask_mask | byte_offset_mask |
                                        byte_end_minus_one_mask | lane_plus_one_mask |
                                        access_byte_count_minus_one_mask | relative_cell_index_mask;
inline constexpr uint32_t maximum_lane = 63;
inline constexpr uint32_t maximum_access_byte_count = 16;
inline constexpr uint32_t maximum_relative_cell_index = 15;

} // namespace consan_moi_exact_byte_cell

struct ConSanMoiExactByteCellProvenance {
  uint8_t byte_mask = 0;
  uint8_t byte_offset = 0;
  uint8_t byte_count = 0;
  uint8_t representative_lane = 0;
  bool valid = false;
  uint8_t access_byte_count = 0;
  uint8_t relative_cell_index = 0;
};

[[nodiscard]] constexpr bool consan_moi_exact_byte_cell_mask_is_contiguous(uint32_t byte_mask) {
  switch (byte_mask) {
  case 0x1:
  case 0x2:
  case 0x3:
  case 0x4:
  case 0x6:
  case 0x7:
  case 0x8:
  case 0xc:
  case 0xe:
  case 0xf:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] constexpr ConSanMoiExactByteCellProvenance
consan_moi_exact_byte_cell_provenance_from_mask(uint32_t byte_mask, uint32_t representative_lane) {
  if (!consan_moi_exact_byte_cell_mask_is_contiguous(byte_mask) ||
      representative_lane > consan_moi_exact_byte_cell::maximum_lane) {
    return {};
  }
  uint32_t byte_offset = 0;
  while ((byte_mask & (1u << byte_offset)) == 0u)
    ++byte_offset;
  uint32_t byte_count = 0;
  while ((byte_mask & (1u << (byte_offset + byte_count))) != 0u)
    ++byte_count;
  return {
      static_cast<uint8_t>(byte_mask),
      static_cast<uint8_t>(byte_offset),
      static_cast<uint8_t>(byte_count),
      static_cast<uint8_t>(representative_lane),
      true,
  };
}

namespace consan_moi_exact_byte_cell {

inline constexpr uint32_t byte_offset_by_mask_lookup = [] {
  uint32_t lookup = 0;
  for (uint32_t mask = 0; mask <= 0xfu; ++mask) {
    const ConSanMoiExactByteCellProvenance provenance =
        consan_moi_exact_byte_cell_provenance_from_mask(mask, 0u);
    if (provenance.valid)
      lookup |= static_cast<uint32_t>(provenance.byte_offset) << (2u * mask);
  }
  return lookup;
}();

inline constexpr uint32_t byte_end_minus_one_by_mask_lookup = [] {
  uint32_t lookup = 0;
  for (uint32_t mask = 0; mask <= 0xfu; ++mask) {
    const ConSanMoiExactByteCellProvenance provenance =
        consan_moi_exact_byte_cell_provenance_from_mask(mask, 0u);
    if (provenance.valid) {
      lookup |= static_cast<uint32_t>(provenance.byte_offset + provenance.byte_count - 1u)
                << (2u * mask);
    }
  }
  return lookup;
}();

} // namespace consan_moi_exact_byte_cell

[[nodiscard]] constexpr ConSanMoiExactByteCellProvenance
decode_consan_moi_exact_byte_cell_provenance(uint32_t packed) {
  const uint32_t byte_mask = (packed & consan_moi_exact_byte_cell::byte_mask_mask) >>
                             consan_moi_exact_byte_cell::byte_mask_shift;
  const uint32_t lane_plus_one = (packed & consan_moi_exact_byte_cell::lane_plus_one_mask) >>
                                 consan_moi_exact_byte_cell::lane_plus_one_shift;
  const uint32_t byte_offset = (packed & consan_moi_exact_byte_cell::byte_offset_mask) >>
                               consan_moi_exact_byte_cell::byte_offset_shift;
  const uint32_t byte_end = ((packed & consan_moi_exact_byte_cell::byte_end_minus_one_mask) >>
                             consan_moi_exact_byte_cell::byte_end_minus_one_shift) +
                            1u;
  const uint32_t access_byte_count =
      ((packed & consan_moi_exact_byte_cell::access_byte_count_minus_one_mask) >>
       consan_moi_exact_byte_cell::access_byte_count_minus_one_shift) +
      1u;
  const uint32_t relative_cell_index =
      (packed & consan_moi_exact_byte_cell::relative_cell_index_mask) >>
      consan_moi_exact_byte_cell::relative_cell_index_shift;
  const bool valid = (packed & ~consan_moi_exact_byte_cell::packed_mask) == 0u &&
                     consan_moi_exact_byte_cell_mask_is_contiguous(byte_mask) &&
                     byte_offset < byte_end && byte_end <= consan_moi_shadow_cell::granule_bytes &&
                     byte_mask == (((1u << (byte_end - byte_offset)) - 1u) << byte_offset) &&
                     lane_plus_one >= 1u &&
                     lane_plus_one <= consan_moi_exact_byte_cell::maximum_lane + 1u;
  return {
      static_cast<uint8_t>(byte_mask),
      static_cast<uint8_t>(valid ? byte_offset : 0u),
      static_cast<uint8_t>(valid ? byte_end - byte_offset : 0u),
      static_cast<uint8_t>(valid ? lane_plus_one - 1u : 0u),
      valid,
      static_cast<uint8_t>(valid ? access_byte_count : 0u),
      static_cast<uint8_t>(valid ? relative_cell_index : 0u),
  };
}

[[nodiscard]] constexpr uint32_t
consan_moi_maximum_cell_count_for_unaligned_bytes(uint32_t access_byte_count) {
  if (access_byte_count == 0u)
    return 0u;
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(access_byte_count) + 2u * consan_moi_shadow_cell::granule_bytes - 2u) /
      consan_moi_shadow_cell::granule_bytes);
}

struct ConSanMoiExactShadowEntry {
  ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t generation = 0;
  uint32_t instruction_offset = 0;
};

[[nodiscard]] constexpr uint64_t pack_consan_moi_exact_shadow_entry(ConSanMoiShadowAccessKind kind,
                                                                    uint32_t owner_id,
                                                                    uint32_t epoch,
                                                                    uint32_t generation,
                                                                    uint64_t instruction_offset) {
  return ((static_cast<uint64_t>(kind) &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::access_kind_bits))
          << consan_moi_exact_shadow::access_kind_shift) |
         ((static_cast<uint64_t>(owner_id) &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::owner_bits))
          << consan_moi_exact_shadow::owner_shift) |
         ((static_cast<uint64_t>(epoch) &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::epoch_bits))
          << consan_moi_exact_shadow::epoch_shift) |
         ((static_cast<uint64_t>(generation) &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::generation_bits))
          << consan_moi_exact_shadow::generation_shift) |
         ((instruction_offset &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::instruction_offset_bits))
          << consan_moi_exact_shadow::instruction_offset_shift);
}

[[nodiscard]] constexpr ConSanMoiExactShadowEntry
decode_consan_moi_exact_shadow_entry(uint64_t value) {
  return ConSanMoiExactShadowEntry{
      static_cast<ConSanMoiShadowAccessKind>((value & consan_moi_exact_shadow::access_kind_mask) >>
                                             consan_moi_exact_shadow::access_kind_shift),
      static_cast<uint32_t>((value & consan_moi_exact_shadow::owner_mask) >>
                            consan_moi_exact_shadow::owner_shift),
      static_cast<uint32_t>((value & consan_moi_exact_shadow::epoch_mask) >>
                            consan_moi_exact_shadow::epoch_shift),
      static_cast<uint32_t>((value & consan_moi_exact_shadow::generation_mask) >>
                            consan_moi_exact_shadow::generation_shift),
      static_cast<uint32_t>((value & consan_moi_exact_shadow::instruction_offset_mask) >>
                            consan_moi_exact_shadow::instruction_offset_shift),
  };
}

struct ConSanMoiExactByteOverlap {
  uint32_t byte_offset = 0;
  uint32_t byte_count = 0;

  [[nodiscard]] constexpr bool empty() const { return byte_count == 0; }
};

[[nodiscard]] constexpr ConSanMoiExactByteOverlap
consan_moi_exact_byte_overlap(const ConSanMoiExactByteAccess &first,
                              const ConSanMoiExactByteAccess &second) {
  const uint64_t first_end = static_cast<uint64_t>(first.lds_byte_offset) + first.lds_byte_count;
  const uint64_t second_end = static_cast<uint64_t>(second.lds_byte_offset) + second.lds_byte_count;
  const uint64_t overlap_begin = std::max(first.lds_byte_offset, second.lds_byte_offset);
  const uint64_t overlap_end = std::min(first_end, second_end);
  if (overlap_begin >= overlap_end)
    return {};
  return {
      static_cast<uint32_t>(overlap_begin),
      static_cast<uint32_t>(overlap_end - overlap_begin),
  };
}

/// Target-neutral conflict contract for exact-byte provenance.
/// Different sites in one wave are program ordered. Distinct exact address
/// groups from one site are conservatively treated as concurrent lane groups
/// and conflict on overlap. event_index is publication order, not a dynamic
/// instruction-instance identity: automatic capture assigns a separate index
/// while publishing each group from one address-group loop.
[[nodiscard]] constexpr bool
consan_moi_exact_byte_accesses_conflict(const ConSanMoiExactByteAccess &current,
                                        const ConSanMoiExactByteAccess &prior) {
  if (consan_moi_exact_byte_overlap(current, prior).empty() || current.epoch != prior.epoch ||
      current.generation != prior.generation ||
      !consan_moi_shadow_kind_conflicts(current.kind, prior.kind)) {
    return false;
  }
  if (current.owner_id != prior.owner_id)
    return true;
  return current.exact_address_group && prior.exact_address_group &&
         current.instruction_offset == prior.instruction_offset &&
         current.lane_mask != prior.lane_mask;
}

/// Bounded sparse byte-range provenance for host-side Record/Replay detection.
/// Cross-owner state is partitioned by generation, epoch, access kind, and
/// owner. Same-owner exact-group state additionally retains the instruction
/// site and lane group. These keys contain every non-spatial field read by
/// consan_moi_exact_byte_accesses_conflict; the cross-owner key also contains
/// every prior-access field read by consan_moi_acquired_epoch_orders. An access
/// replaces byte intervals only within a class that is equivalent for every
/// future conflict and acquired-order decision. Storage stays linear in
/// admitted accesses and interval boundaries rather than the LDS extent.
/// Lookup visits the conflicting access-kind classes and the owners or lane
/// groups in the current workgroup that share the relevant class; it does not
/// scale with the LDS extent or the total retained site inventory.
/// maximum_access_count is a structural input bound, not a second runtime
/// policy budget.
class ConSanMoiSparseExactByteShadow {
public:
  explicit ConSanMoiSparseExactByteShadow(uint64_t byte_capacity, uint32_t maximum_access_count);

  [[nodiscard]] ConSanMoiExactByteAccessResult
  access(const ConSanMoiExactByteAccess &current,
         std::span<const ConSanMoiAcquiredEpochToken> acquired_epoch_tokens = {});

  /// Retires epochs strictly before first_live_epoch for one generation and
  /// compacts provenance that is no longer referenced by any byte interval.
  /// The caller must prove that no later access can name a retired epoch; the
  /// model deliberately does not infer that proof from record arrival order.
  void retire_before_epoch(uint64_t generation, uint32_t first_live_epoch);

private:
  struct Interval {
    uint64_t end = 0;
    uint32_t provenance_index = 0;
  };

  struct EpochKindClass {
    uint64_t generation = 0;
    uint32_t epoch = 0;
    ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;

    auto operator<=>(const EpochKindClass &) const = default;
  };

  struct SameSiteClass {
    EpochKindClass epoch_kind;
    uint32_t owner_id = 0;
    uint32_t instruction_offset = 0;

    auto operator<=>(const SameSiteClass &) const = default;
  };

  using IntervalMap = std::map<uint64_t, Interval>;
  using OwnerIntervalMaps = std::map<uint32_t, IntervalMap>;
  using LaneIntervalMaps = std::map<uint64_t, IntervalMap>;

  uint64_t byte_capacity_ = 0;
  uint32_t maximum_access_count_ = 0;
  std::vector<ConSanMoiExactByteAccess> provenance_;
  std::map<EpochKindClass, OwnerIntervalMaps> cross_owner_intervals_;
  std::map<SameSiteClass, LaneIntervalMaps> same_site_intervals_;
};

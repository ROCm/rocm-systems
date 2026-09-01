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

[[nodiscard]] constexpr uint32_t
pack_consan_moi_exact_byte_cell_provenance(uint32_t byte_mask, uint32_t representative_lane) {
  const ConSanMoiExactByteCellProvenance provenance =
      consan_moi_exact_byte_cell_provenance_from_mask(byte_mask, representative_lane);
  if (!provenance.valid)
    return 0;
  return byte_mask |
         (static_cast<uint32_t>(provenance.byte_offset)
          << consan_moi_exact_byte_cell::byte_offset_shift) |
         (static_cast<uint32_t>(provenance.byte_offset + provenance.byte_count - 1u)
          << consan_moi_exact_byte_cell::byte_end_minus_one_shift) |
         ((representative_lane + 1u) << consan_moi_exact_byte_cell::lane_plus_one_shift);
}

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

[[nodiscard]] constexpr uint32_t consan_moi_exact_byte_mask_for_cell(uint32_t access_byte_offset,
                                                                     uint32_t access_byte_count,
                                                                     uint32_t relative_cell_index) {
  if (access_byte_count == 0u)
    return 0u;
  const uint32_t start = access_byte_offset & (consan_moi_shadow_cell::granule_bytes - 1u);
  const uint64_t relative_cell_start =
      static_cast<uint64_t>(relative_cell_index) * consan_moi_shadow_cell::granule_bytes;
  const uint64_t relative_cell_end = relative_cell_start + consan_moi_shadow_cell::granule_bytes;
  const uint64_t access_end = static_cast<uint64_t>(start) + access_byte_count;
  const uint64_t overlap_start = std::max<uint64_t>(start, relative_cell_start);
  const uint64_t overlap_end = std::min(access_end, relative_cell_end);
  if (overlap_start >= overlap_end)
    return 0u;
  const uint32_t cell_offset = static_cast<uint32_t>(overlap_start - relative_cell_start);
  const uint32_t cell_count = static_cast<uint32_t>(overlap_end - overlap_start);
  return ((1u << cell_count) - 1u) << cell_offset;
}

[[nodiscard]] constexpr uint32_t
consan_moi_maximum_cell_count_for_unaligned_bytes(uint32_t access_byte_count) {
  if (access_byte_count == 0u)
    return 0u;
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(access_byte_count) + 2u * consan_moi_shadow_cell::granule_bytes - 2u) /
      consan_moi_shadow_cell::granule_bytes);
}

enum class ConSanMoiInlineWorkgroupKeyShape : uint8_t {
  OneDimensional,
  TwoDimensional,
  ThreeDimensional,
};

/// AMDHSA initializes at most 16 user SGPRs. Dispatch-ID insertion is allowed
/// only when all original preloads, the new two-SGPR ID, and the following
/// system SGPRs can be mapped without truncation or register overflow.
enum class ConSanMoiDispatchIdPreloadSupport : uint8_t {
  SupportedAlreadyEnabled,
  SupportedInsert,
  InvalidDispatchPosition,
  UserSgprInitializationLimit,
  SgprAllocationLimit,
};

struct ConSanMoiDispatchIdPreloadPlan {
  ConSanMoiDispatchIdPreloadSupport support =
      ConSanMoiDispatchIdPreloadSupport::InvalidDispatchPosition;
  uint16_t dispatch_id_sgpr = 0;
  uint16_t original_user_sgpr_count = 0;
  uint16_t expanded_user_sgpr_count = 0;
  uint16_t system_sgpr_count = 0;
  uint16_t first_shifted_guest_sgpr = 0;
  uint16_t shifted_guest_sgpr_count = 0;
  /// Full user-SGPR windows can make room for dispatch ID by shortening the
  /// hardware kernarg preload and reloading its final two dwords in software.
  uint16_t kernarg_reload_sgpr = 0;
  uint16_t kernarg_reload_base_sgpr = 0;
  uint16_t kernarg_reload_offset_dwords = 0;
  uint16_t kernarg_reload_count = 0;
  uint16_t shifted_system_sgpr_count = 0;
  uint16_t system_sgpr_shift = 0;
  uint16_t original_kernarg_preload_length = 0;
  uint16_t replacement_kernarg_preload_length = 0;
  uint16_t required_sgpr_count = 0;

  [[nodiscard]] constexpr bool supported() const {
    return support == ConSanMoiDispatchIdPreloadSupport::SupportedAlreadyEnabled ||
           support == ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
  }
  [[nodiscard]] constexpr bool descriptor_change_required() const {
    return support == ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
  }
  [[nodiscard]] constexpr bool requires_kernarg_reload() const { return kernarg_reload_count != 0; }
};

[[nodiscard]] constexpr uint16_t
consan_moi_amdhsa_dispatch_id_prefix_sgpr_count(bool private_segment_buffer, bool dispatch_ptr,
                                                bool queue_ptr, bool kernarg_segment_ptr) {
  return static_cast<uint16_t>((private_segment_buffer ? 4u : 0u) + (dispatch_ptr ? 2u : 0u) +
                               (queue_ptr ? 2u : 0u) + (kernarg_segment_ptr ? 2u : 0u));
}

/// Reference contract for adding AMDHSA's 64-bit dispatch-ID preload without
/// changing the SGPR view seen by guest instructions. The generated prologue
/// first captures dispatch_id_sgpr:{+1} into persistent instrumentation state,
/// then copies every shifted guest preload from high layout index `dest + 2`
/// back to original index `dest`, in increasing destination order.
[[nodiscard]] constexpr ConSanMoiDispatchIdPreloadPlan
consan_moi_plan_dispatch_id_preload(uint16_t original_user_sgpr_count, uint16_t system_sgpr_count,
                                    uint16_t dispatch_id_prefix_sgpr_count,
                                    bool dispatch_id_already_enabled, uint16_t sgpr_limit = 106,
                                    uint16_t user_sgpr_initialization_limit = 16) {
  ConSanMoiDispatchIdPreloadPlan plan;
  plan.dispatch_id_sgpr = dispatch_id_prefix_sgpr_count;
  plan.original_user_sgpr_count = original_user_sgpr_count;
  plan.expanded_user_sgpr_count = original_user_sgpr_count;
  plan.system_sgpr_count = system_sgpr_count;
  const uint32_t original_required_count =
      static_cast<uint32_t>(original_user_sgpr_count) + system_sgpr_count;

  const uint32_t dispatch_end = static_cast<uint32_t>(dispatch_id_prefix_sgpr_count) + 2u;
  if (dispatch_id_prefix_sgpr_count > original_user_sgpr_count ||
      (dispatch_id_already_enabled && dispatch_end > original_user_sgpr_count)) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::InvalidDispatchPosition;
    return plan;
  }
  if (original_user_sgpr_count > user_sgpr_initialization_limit) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::UserSgprInitializationLimit;
    return plan;
  }
  if (dispatch_id_already_enabled) {
    if (original_required_count > sgpr_limit) {
      plan.support = ConSanMoiDispatchIdPreloadSupport::SgprAllocationLimit;
      return plan;
    }
    plan.required_sgpr_count = static_cast<uint16_t>(original_required_count);
    plan.support = ConSanMoiDispatchIdPreloadSupport::SupportedAlreadyEnabled;
    return plan;
  }

  const uint32_t expanded_user_count = static_cast<uint32_t>(original_user_sgpr_count) + 2u;
  const uint32_t required_count = expanded_user_count + system_sgpr_count;
  if (expanded_user_count > user_sgpr_initialization_limit) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::UserSgprInitializationLimit;
    return plan;
  }
  if (required_count > sgpr_limit) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::SgprAllocationLimit;
    return plan;
  }
  plan.expanded_user_sgpr_count = static_cast<uint16_t>(expanded_user_count);
  plan.first_shifted_guest_sgpr = dispatch_id_prefix_sgpr_count;
  plan.shifted_guest_sgpr_count = static_cast<uint16_t>(
      original_user_sgpr_count + system_sgpr_count - dispatch_id_prefix_sgpr_count);
  plan.required_sgpr_count = static_cast<uint16_t>(required_count);
  plan.support = ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
  return plan;
}

[[nodiscard]] constexpr std::optional<uint16_t>
consan_moi_dispatch_id_restore_source(const ConSanMoiDispatchIdPreloadPlan &plan,
                                      uint16_t guest_destination_sgpr) {
  if (!plan.supported() || !plan.descriptor_change_required())
    return std::nullopt;
  const uint32_t shifted_end =
      static_cast<uint32_t>(plan.first_shifted_guest_sgpr) + plan.shifted_guest_sgpr_count;
  if (guest_destination_sgpr < plan.first_shifted_guest_sgpr ||
      guest_destination_sgpr >= shifted_end)
    return std::nullopt;
  return static_cast<uint16_t>(guest_destination_sgpr + 2u);
}

struct ConSanMoiInlineWorkgroupKey {
  bool valid = false;
  uint32_t value = 0;
};

/// Encodes an exact nonzero workgroup identity in the exact-shadow generation
/// field. Each shape is injective inside its declared bounds; coordinates that
/// do not fit are unsupported rather than truncated or hashed.
[[nodiscard]] constexpr ConSanMoiInlineWorkgroupKey
consan_moi_inline_workgroup_key(uint32_t x, uint32_t y, uint32_t z,
                                ConSanMoiInlineWorkgroupKeyShape shape) {
  uint32_t packed = 0;
  switch (shape) {
  case ConSanMoiInlineWorkgroupKeyShape::OneDimensional:
    if (y != 0 || z != 0 || x >= consan_moi_exact_shadow::max_generation)
      return {};
    packed = x;
    break;
  case ConSanMoiInlineWorkgroupKeyShape::TwoDimensional:
    if (z != 0 || x >= (1u << 10u) || y >= (1u << 10u))
      return {};
    packed = x | (y << 10u);
    if (packed == consan_moi_exact_shadow::max_generation)
      return {};
    break;
  case ConSanMoiInlineWorkgroupKeyShape::ThreeDimensional:
    if (x >= (1u << 8u) || y >= (1u << 6u) || z >= (1u << 6u))
      return {};
    packed = x | (y << 8u) | (z << 14u);
    if (packed == consan_moi_exact_shadow::max_generation)
      return {};
    break;
  }
  return {/*valid=*/true, /*value=*/packed + 1u};
}

struct ConSanMoiExactShadowEntry {
  ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t generation = 0;
  uint32_t instruction_offset = 0;
};

[[nodiscard]] constexpr bool
consan_moi_exact_byte_cells_conflict(const ConSanMoiExactShadowEntry &current_access,
                                     const ConSanMoiExactByteCellProvenance &current_byte,
                                     const ConSanMoiExactShadowEntry &prior_access,
                                     const ConSanMoiExactByteCellProvenance &prior_byte) {
  // Specification model for the emitted Inline predicate. Exact address
  // grouping and byte-mask partitioning happen before publication, so they do
  // not need separate flags here. The owner is a normalized resident-wave
  // identity that is unique within the workgroup's execution domain, while
  // representative_lane is the wave-relative workitem discriminator. Their
  // pair is independent of the launch dimensionality.
  if (!current_byte.valid || !prior_byte.valid ||
      (current_byte.byte_mask & prior_byte.byte_mask) == 0u ||
      current_access.epoch != prior_access.epoch ||
      current_access.generation != prior_access.generation ||
      !consan_moi_shadow_kind_conflicts(current_access.kind, prior_access.kind)) {
    return false;
  }
  if (current_access.owner_id != prior_access.owner_id)
    return true;
  return current_access.instruction_offset == prior_access.instruction_offset &&
         current_byte.representative_lane != prior_byte.representative_lane;
}

inline constexpr uint32_t kConSanMoiInlineExactInitialVersion = 0;
inline constexpr uint32_t kConSanMoiInlineExactMaxReadyVersion =
    std::numeric_limits<uint32_t>::max() - 1u;

enum class ConSanMoiInlineExactSnapshotState : uint8_t {
  Empty,
  Stable,
  Publishing,
  ChangedDuringRead,
  Malformed,
};

/// Words observed by a reader in acquire order. The version is deliberately
/// sampled around every payload field: accepting a sidecar or one version read
/// would permit a packed access from one publisher to be paired with another
/// dispatch.
struct ConSanMoiInlineExactSnapshotWords {
  uint32_t version_before = 0;
  uint64_t packed_access = 0;
  uint64_t dispatch_id = 0;
  uint32_t byte_provenance = 0;
  uint32_t version_after = 0;
};

struct ConSanMoiInlineExactSnapshot {
  ConSanMoiInlineExactSnapshotState state = ConSanMoiInlineExactSnapshotState::Empty;
  ConSanMoiExactShadowEntry entry;
  ConSanMoiExactByteCellProvenance byte_provenance;
  uint64_t dispatch_id = 0;
  uint32_t version = 0;
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

[[nodiscard]] constexpr ConSanMoiInlineExactSnapshot
classify_consan_moi_inline_exact_snapshot(const ConSanMoiInlineExactSnapshotWords &words) {
  if (words.version_before != words.version_after) {
    return {.state = ConSanMoiInlineExactSnapshotState::ChangedDuringRead,
            .entry = {},
            .byte_provenance = {},
            .dispatch_id = 0,
            .version = 0};
  }
  if ((words.version_after & 1u) != 0) {
    return {.state = ConSanMoiInlineExactSnapshotState::Publishing,
            .entry = {},
            .byte_provenance = {},
            .dispatch_id = 0,
            .version = words.version_after};
  }
  if (words.version_after == kConSanMoiInlineExactInitialVersion) {
    if (words.packed_access == 0 && words.dispatch_id == 0 && words.byte_provenance == 0)
      return {.state = ConSanMoiInlineExactSnapshotState::Empty,
              .entry = {},
              .byte_provenance = {},
              .dispatch_id = 0,
              .version = 0};
    return {.state = ConSanMoiInlineExactSnapshotState::Malformed,
            .entry = {},
            .byte_provenance = {},
            .dispatch_id = 0,
            .version = 0};
  }
  const ConSanMoiExactShadowEntry entry = decode_consan_moi_exact_shadow_entry(words.packed_access);
  const ConSanMoiExactByteCellProvenance byte_provenance =
      decode_consan_moi_exact_byte_cell_provenance(words.byte_provenance);
  if (!byte_provenance.valid || words.packed_access == 0 || words.dispatch_id == 0 ||
      entry.kind < ConSanMoiShadowAccessKind::Read ||
      entry.kind > ConSanMoiShadowAccessKind::Atomic || entry.generation == 0) {
    return {.state = ConSanMoiInlineExactSnapshotState::Malformed,
            .entry = {},
            .byte_provenance = {},
            .dispatch_id = 0,
            .version = words.version_after};
  }
  return {.state = ConSanMoiInlineExactSnapshotState::Stable,
          .entry = entry,
          .byte_provenance = byte_provenance,
          .dispatch_id = words.dispatch_id,
          .version = words.version_after};
}

/// Host reference for the packed-cell conflict checks emitted by Inline
/// Shadow. The device path performs the equivalent field comparisons.
[[nodiscard]] constexpr bool
consan_moi_exact_shadow_entries_conflict(const ConSanMoiExactShadowEntry &current,
                                         const ConSanMoiExactShadowEntry &prior) {
  return !consan_moi_shadow_kind_is_empty(prior.kind) && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         consan_moi_shadow_kind_conflicts(current.kind, prior.kind);
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

[[nodiscard]] constexpr uint32_t consan_moi_inline_acquired_epoch_token_value(uint32_t epoch) {
  return epoch >= consan_moi_exact_shadow::max_epoch ? consan_moi_exact_shadow::max_epoch
                                                     : epoch + 1u;
}

/// Qualify one directed edge in a deferred diagnostic. The diagnostic retains
/// both access epochs and the persistent workgroup key, so a stable final token
/// can safely prove that this consumer observed this producer segment even
/// when the token transaction completed after the device-side lookup.
[[nodiscard]] constexpr bool consan_moi_inline_stable_token_orders_deferred(
    const ConSanMoiInlineAcquiredEpochTokenSlot &token, uint64_t dispatch_id,
    uint32_t workgroup_key, uint32_t consumer_owner_id, uint32_t producer_owner_id,
    uint32_t consumer_epoch, uint32_t producer_epoch) {
  if (dispatch_id == 0 || workgroup_key == 0 || consumer_owner_id == 0 || producer_owner_id == 0 ||
      consumer_owner_id == producer_owner_id ||
      consumer_epoch >= consan_moi_exact_shadow::max_epoch ||
      producer_epoch > consan_moi_exact_shadow::max_epoch || token.version == 0 ||
      (token.version & 1u) != 0 || token.source_release_address == 0 ||
      !consan_moi_inline_release_version_is_ready(token.source_release_version) ||
      !consan_moi_inline_causal_epoch_is_valid(token.consumer_epoch_plus_one) ||
      !consan_moi_inline_causal_epoch_is_valid(token.producer_epoch_plus_one) ||
      token.reservation_version != 0)
    return false;
  if (token.dispatch_id != dispatch_id || token.workgroup_key != workgroup_key ||
      token.consumer_owner_id != consumer_owner_id ||
      token.producer_owner_id != producer_owner_id ||
      token.consumer_epoch_plus_one >
          consan_moi_inline_acquired_epoch_token_value(consumer_epoch) ||
      token.producer_epoch_plus_one < consan_moi_inline_acquired_epoch_token_value(producer_epoch))
    return false;
  return token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct) ||
         token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited);
}

struct ConSanMoiDeferredDiagnosticFilterResult {
  std::vector<uint32_t> visible_indices;
  uint32_t qualified_count = 0;
  uint32_t effective_diagnostic_count = 0;
};

/// Filters only diagnostics whose access pair is ordered in either direction
/// by a stable acquired token. The total count includes dropped records, so
/// the effective count retains overflow evidence while subtracting qualified
/// visible records. Incomplete token evidence fails closed.
[[nodiscard]] inline ConSanMoiDeferredDiagnosticFilterResult
consan_moi_filter_deferred_inline_diagnostics(
    std::span<const ConSanMoiDiagnosticRecord> visible_diagnostics, uint32_t total_diagnostic_count,
    std::span<const ConSanMoiInlineAcquiredEpochTokenSlot> stable_tokens,
    bool token_evidence_complete) {
  ConSanMoiDeferredDiagnosticFilterResult result;
  result.visible_indices.reserve(visible_diagnostics.size());
  for (uint32_t i = 0; i < visible_diagnostics.size(); ++i) {
    const auto &diagnostic = visible_diagnostics[i];
    const bool ordered =
        token_evidence_complete &&
        diagnostic.kind == static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict) &&
        diagnostic.backend == static_cast<uint32_t>(ConSanMoiEngine::InlineShadow) &&
        diagnostic.reserved != 0 && std::ranges::any_of(stable_tokens, [&](const auto &token) {
          return consan_moi_inline_stable_token_orders_deferred(
                     token, diagnostic.generation, diagnostic.reserved, diagnostic.second_owner_id,
                     diagnostic.first_owner_id, diagnostic.epoch, diagnostic.first_epoch) ||
                 consan_moi_inline_stable_token_orders_deferred(
                     token, diagnostic.generation, diagnostic.reserved, diagnostic.first_owner_id,
                     diagnostic.second_owner_id, diagnostic.first_epoch, diagnostic.epoch);
        });
    if (ordered)
      ++result.qualified_count;
    else
      result.visible_indices.push_back(i);
  }
  result.effective_diagnostic_count = total_diagnostic_count >= result.qualified_count
                                          ? total_diagnostic_count - result.qualified_count
                                          : 0;
  return result;
}

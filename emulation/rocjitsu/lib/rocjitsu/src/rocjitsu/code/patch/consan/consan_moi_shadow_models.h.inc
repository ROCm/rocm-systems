struct ConSanMoiLdsCellRange {
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
};

[[nodiscard]] constexpr uint64_t consan_moi_low_bit_mask(uint32_t bits) {
  return bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1u);
}

[[nodiscard]] constexpr uint64_t consan_moi_bit_field_mask(uint32_t shift, uint32_t bits) {
  return consan_moi_low_bit_mask(bits) << shift;
}

[[nodiscard]] constexpr ConSanMoiShadowAccessKind
consan_moi_shadow_kind_from_access_kind(ConSanLdsAccessKind kind) {
  switch (kind) {
  case ConSanLdsAccessKind::Read:
    return ConSanMoiShadowAccessKind::Read;
  case ConSanLdsAccessKind::Write:
    return ConSanMoiShadowAccessKind::Write;
  case ConSanLdsAccessKind::Atomic:
    return ConSanMoiShadowAccessKind::Atomic;
  case ConSanLdsAccessKind::Other:
    return ConSanMoiShadowAccessKind::Empty;
  }
  return ConSanMoiShadowAccessKind::Empty;
}

[[nodiscard]] constexpr bool consan_moi_shadow_kind_is_empty(ConSanMoiShadowAccessKind kind) {
  return kind == ConSanMoiShadowAccessKind::Empty;
}

[[nodiscard]] constexpr bool consan_moi_shadow_kind_conflicts(ConSanMoiShadowAccessKind current,
                                                              ConSanMoiShadowAccessKind prior) {
  if (consan_moi_shadow_kind_is_empty(current) || consan_moi_shadow_kind_is_empty(prior))
    return false;
  if (current == ConSanMoiShadowAccessKind::Read)
    return prior != ConSanMoiShadowAccessKind::Read;
  if (current == ConSanMoiShadowAccessKind::Atomic)
    return prior != ConSanMoiShadowAccessKind::Atomic;
  return true;
}

namespace consan_moi_exact_shadow {

inline constexpr uint32_t granule_shift = 2;
inline constexpr uint32_t granule_bytes = 1u << granule_shift;

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
  uint32_t reserved = 0;
  uint32_t version_after = 0;
};

struct ConSanMoiInlineExactSnapshot {
  ConSanMoiInlineExactSnapshotState state = ConSanMoiInlineExactSnapshotState::Empty;
  ConSanMoiExactShadowEntry entry;
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
            .dispatch_id = 0,
            .version = 0};
  }
  if ((words.version_after & 1u) != 0) {
    return {.state = ConSanMoiInlineExactSnapshotState::Publishing,
            .entry = {},
            .dispatch_id = 0,
            .version = words.version_after};
  }
  if (words.version_after == kConSanMoiInlineExactInitialVersion) {
    if (words.packed_access == 0 && words.dispatch_id == 0 && words.reserved == 0)
      return {.state = ConSanMoiInlineExactSnapshotState::Empty,
              .entry = {},
              .dispatch_id = 0,
              .version = 0};
    return {.state = ConSanMoiInlineExactSnapshotState::Malformed,
            .entry = {},
            .dispatch_id = 0,
            .version = 0};
  }
  const ConSanMoiExactShadowEntry entry = decode_consan_moi_exact_shadow_entry(words.packed_access);
  if (words.reserved != 0 || words.packed_access == 0 || words.dispatch_id == 0 ||
      entry.kind < ConSanMoiShadowAccessKind::Read ||
      entry.kind > ConSanMoiShadowAccessKind::Atomic || entry.generation == 0) {
    return {.state = ConSanMoiInlineExactSnapshotState::Malformed,
            .entry = {},
            .dispatch_id = 0,
            .version = words.version_after};
  }
  return {.state = ConSanMoiInlineExactSnapshotState::Stable,
          .entry = entry,
          .dispatch_id = words.dispatch_id,
          .version = words.version_after};
}

[[nodiscard]] constexpr bool
consan_moi_exact_shadow_entries_conflict(const ConSanMoiExactShadowEntry &current,
                                         const ConSanMoiExactShadowEntry &prior) {
  return !consan_moi_shadow_kind_is_empty(prior.kind) && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         consan_moi_shadow_kind_conflicts(current.kind, prior.kind);
}

[[nodiscard]] constexpr uint32_t consan_moi_inline_acquired_epoch_token_value(uint32_t epoch) {
  return epoch >= consan_moi_exact_shadow::max_epoch ? consan_moi_exact_shadow::max_epoch
                                                     : epoch + 1u;
}

struct ConSanMoiInlineAcquiredEpochTokenPublishResult {
  bool updated = false;
  bool collision = false;
  bool invalid_identity = false;
};

/// Reference publication contract for one already-indexed direct-mapped slot.
/// Exact updates are monotonic; a collision or invalid identity leaves the
/// slot untouched so it can never manufacture acquired order.
[[nodiscard]] constexpr ConSanMoiInlineAcquiredEpochTokenPublishResult
consan_moi_inline_publish_acquired_epoch_token(ConSanMoiInlineAcquiredEpochTokenSlot &slot,
                                               uint32_t workgroup_key, uint32_t consumer_owner_id,
                                               uint32_t producer_owner_id, uint32_t producer_epoch,
                                               uint32_t consumer_epoch, uint64_t dispatch_id,
                                               ConSanMoiInlineTokenEvidenceKind kind,
                                               uint64_t source_release_address,
                                               uint32_t source_release_version) {
  ConSanMoiInlineAcquiredEpochTokenPublishResult result;
  if (workgroup_key == 0 || consumer_owner_id == 0 || producer_owner_id == 0 ||
      consumer_owner_id == producer_owner_id || dispatch_id == 0 || source_release_address == 0 ||
      !consan_moi_inline_release_version_is_ready(source_release_version) ||
      consumer_epoch >= consan_moi_exact_shadow::max_epoch ||
      (kind != ConSanMoiInlineTokenEvidenceKind::Direct &&
       kind != ConSanMoiInlineTokenEvidenceKind::Inherited &&
       kind != ConSanMoiInlineTokenEvidenceKind::ReleaseSequence)) {
    result.invalid_identity = true;
    return result;
  }
  const auto lookup = consan_moi_inline_acquired_epoch_token_lookup(
      slot, workgroup_key, consumer_owner_id, producer_owner_id);
  if (lookup == ConSanMoiInlineAcquiredEpochTokenLookup::Collision) {
    result.collision = true;
    return result;
  }
  const uint32_t token_value = consan_moi_inline_acquired_epoch_token_value(producer_epoch);
  const uint32_t consumer_token_value =
      consan_moi_inline_acquired_epoch_token_value(consumer_epoch);
  if (lookup == ConSanMoiInlineAcquiredEpochTokenLookup::Exact &&
      slot.producer_epoch_plus_one >= token_value &&
      slot.consumer_epoch_plus_one >= consumer_token_value)
    return result;
  slot = {.version = 2,
          .consumer_owner_id = consumer_owner_id,
          .producer_owner_id = producer_owner_id,
          .producer_epoch_plus_one = token_value,
          .workgroup_key = workgroup_key,
          .kind = static_cast<uint32_t>(kind),
          .dispatch_id = dispatch_id,
          .source_release_address = source_release_address,
          .source_release_version = source_release_version,
          .consumer_epoch_plus_one = consumer_token_value};
  result.updated = true;
  return result;
}

[[nodiscard]] constexpr bool
consan_moi_inline_acquired_epoch_orders(const ConSanMoiInlineAcquiredEpochTokenSlot &slot,
                                        const ConSanMoiExactShadowEntry &current,
                                        const ConSanMoiExactShadowEntry &prior) {
  if (current.generation == 0 || current.generation != prior.generation)
    return false;
  if (consan_moi_inline_acquired_epoch_token_lookup(slot, current.generation, current.owner_id,
                                                    prior.owner_id) !=
      ConSanMoiInlineAcquiredEpochTokenLookup::Exact)
    return false;
  return current.epoch < consan_moi_exact_shadow::max_epoch &&
         slot.consumer_epoch_plus_one <=
             consan_moi_inline_acquired_epoch_token_value(current.epoch) &&
         slot.producer_epoch_plus_one >= consan_moi_inline_acquired_epoch_token_value(prior.epoch);
}

/// Exact-shadow replacement order is not itself a happens-before clock.
/// Accept either directed token when it proves that the two observed accesses
/// are causally ordered.
[[nodiscard]] constexpr bool consan_moi_inline_acquired_epoch_orders_pair(
    const ConSanMoiInlineAcquiredEpochTokenSlot &current_after_prior,
    const ConSanMoiInlineAcquiredEpochTokenSlot &prior_after_current,
    const ConSanMoiExactShadowEntry &current, const ConSanMoiExactShadowEntry &prior) {
  return consan_moi_inline_acquired_epoch_orders(current_after_prior, current, prior) ||
         consan_moi_inline_acquired_epoch_orders(prior_after_current, prior, current);
}

/// A stable full token records causal evidence from one source-release
/// version. Direct and inherited tokens commit only after the source evidence
/// is stable and the guest RMW has completed, so later publication to the
/// direct-mapped source slot cannot revoke either fact. Release-sequence tokens
/// carry ancestry between release operations and never authorize an ordinary
/// access directly.
[[nodiscard]] constexpr bool consan_moi_inline_stable_token_orders(
    const ConSanMoiInlineAcquiredTokenSnapshot &token_words,
    const ConSanMoiInlineReleaseSnapshotWords & /*source_words*/, uint64_t dispatch_id,
    const ConSanMoiExactShadowEntry &current, const ConSanMoiExactShadowEntry &prior) {
  const auto classified_token = consan_moi_inline_classify_acquired_token(token_words);
  if (classified_token.state != ConSanMoiInlineAcquiredTokenState::Stable || dispatch_id == 0 ||
      current.generation == 0 || current.generation != prior.generation)
    return false;
  const auto &token = classified_token.token;
  if (token.dispatch_id != dispatch_id || token.workgroup_key != current.generation ||
      token.consumer_owner_id != current.owner_id || token.producer_owner_id != prior.owner_id ||
      current.epoch >= consan_moi_exact_shadow::max_epoch ||
      token.consumer_epoch_plus_one > consan_moi_inline_acquired_epoch_token_value(current.epoch) ||
      token.producer_epoch_plus_one < consan_moi_inline_acquired_epoch_token_value(prior.epoch))
    return false;
  return token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct) ||
         token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited);
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

[[nodiscard]] constexpr uint32_t consan_moi_acquired_epoch_token_value(uint32_t epoch) {
  return epoch == std::numeric_limits<uint32_t>::max() ? epoch : epoch + 1u;
}

namespace consan_moi_sampled_watchpoint {

inline constexpr uint32_t granule_shift = 2;
inline constexpr uint32_t granule_bytes = 1u << granule_shift;

inline constexpr uint32_t valid_shift = 0;
inline constexpr uint32_t consumed_shift = 1;
inline constexpr uint32_t access_kind_shift = 2;
inline constexpr uint32_t access_kind_bits = 3;
inline constexpr uint32_t owner_shift = 5;
inline constexpr uint32_t owner_bits = 10;
inline constexpr uint32_t epoch_shift = 15;
inline constexpr uint32_t epoch_bits = 10;
inline constexpr uint32_t generation_shift = 25;
inline constexpr uint32_t generation_bits = 20;
inline constexpr uint32_t start_shift = 45;
inline constexpr uint32_t start_bits = 14;
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
inline constexpr uint64_t start_mask = consan_moi_bit_field_mask(start_shift, start_bits);
inline constexpr uint64_t count_mask = consan_moi_bit_field_mask(count_shift, count_bits);
inline constexpr uint64_t epoch_generation_mask = epoch_mask | generation_mask;

inline constexpr uint32_t max_owner = (1u << owner_bits) - 1u;
inline constexpr uint32_t max_epoch = (1u << epoch_bits) - 1u;
inline constexpr uint32_t max_generation = (1u << generation_bits) - 1u;
inline constexpr uint32_t max_start = (1u << start_bits) - 1u;
inline constexpr uint32_t max_count = 1u << count_bits;

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
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
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

[[nodiscard]] constexpr uint32_t encode_consan_moi_sampled_cell_count(uint32_t cell_count) {
  return cell_count == 0 ? 0 : cell_count - 1u;
}

[[nodiscard]] constexpr uint64_t
pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind kind, uint32_t owner_id,
                                         uint32_t epoch, uint32_t generation, uint32_t start_cell,
                                         uint32_t cell_count, bool consumed = false) {
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
         ((static_cast<uint64_t>(start_cell) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::start_bits))
          << consan_moi_sampled_watchpoint::start_shift) |
         ((static_cast<uint64_t>(encode_consan_moi_sampled_cell_count(cell_count)) &
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
      static_cast<uint32_t>((value & consan_moi_sampled_watchpoint::start_mask) >>
                            consan_moi_sampled_watchpoint::start_shift),
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
      entry.kind != ConSanMoiShadowAccessKind::Write)
    return {ConSanMoiSampledSnapshotState::Malformed, entry};
  return {ConSanMoiSampledSnapshotState::Stable, entry};
}

[[nodiscard]] constexpr uint32_t consan_moi_range_end_cell(uint32_t start_cell,
                                                           uint32_t cell_count) {
  return start_cell + cell_count;
}

[[nodiscard]] constexpr bool consan_moi_cell_ranges_overlap(uint32_t first_start,
                                                            uint32_t first_count,
                                                            uint32_t second_start,
                                                            uint32_t second_count) {
  return first_start < consan_moi_range_end_cell(second_start, second_count) &&
         second_start < consan_moi_range_end_cell(first_start, first_count);
}

[[nodiscard]] constexpr bool consan_moi_cell_ranges_overlap(ConSanMoiLdsCellRange first,
                                                            ConSanMoiLdsCellRange second) {
  return consan_moi_cell_ranges_overlap(first.start_cell, first.cell_count, second.start_cell,
                                        second.cell_count);
}

[[nodiscard]] constexpr ConSanMoiLdsCellRange
consan_moi_lds_cell_range_for_bytes(uint32_t byte_offset, uint32_t byte_count) {
  const uint32_t start_cell = byte_offset >> consan_moi_exact_shadow::granule_shift;
  const uint32_t end_cell =
      (byte_offset + byte_count + consan_moi_exact_shadow::granule_bytes - 1u) >>
      consan_moi_exact_shadow::granule_shift;
  return ConSanMoiLdsCellRange{start_cell, end_cell - start_cell};
}

[[nodiscard]] constexpr bool
consan_moi_sampled_watchpoints_conflict(const ConSanMoiSampledWatchpointEntry &current,
                                        const ConSanMoiSampledWatchpointEntry &prior) {
  return current.valid && prior.valid && !prior.consumed && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         consan_moi_shadow_kind_conflicts(current.kind, prior.kind) &&
         consan_moi_cell_ranges_overlap(current.start_cell, current.cell_count, prior.start_cell,
                                        prior.cell_count);
}

[[nodiscard]] ConSanMoiRecordReplayAccessResult
consan_moi_record_replay_access(std::span<uint64_t> exact_shadow_entries,
                                const ConSanMoiRecordReplayAccess &access);

[[nodiscard]] ConSanMoiRecordReplayAccessResult
consan_moi_record_replay_access(std::span<uint64_t> exact_shadow_entries,
                                const ConSanMoiRecordReplayAccess &access,
                                std::span<const ConSanMoiAcquiredEpochToken> acquired_epoch_tokens);

[[nodiscard]] bool
consan_moi_acquired_epoch_orders(std::span<const ConSanMoiAcquiredEpochToken> tokens,
                                 const ConSanMoiExactShadowEntry &current,
                                 const ConSanMoiExactShadowEntry &prior);

[[nodiscard]] ConSanMoiAtomicSyncResult
consan_moi_record_replay_atomic_release(std::span<ConSanMoiAtomicReleaseRecord> release_records,
                                        uint64_t generation, uint64_t atomic_address,
                                        uint32_t producer_owner_id, uint32_t producer_epoch,
                                        uint32_t release_instruction_offset);

[[nodiscard]] ConSanMoiAtomicSyncResult consan_moi_record_replay_atomic_acquire(
    std::span<const ConSanMoiAtomicReleaseRecord> release_records,
    std::span<ConSanMoiAcquiredEpochToken> acquired_epoch_tokens, uint64_t generation,
    uint64_t atomic_address, uint32_t consumer_owner_id, uint32_t acquire_instruction_offset);

[[nodiscard]] ConSanMoiRecordReplayResult
consan_moi_record_replay_access_records(ConSanMoiReportHeader &header,
                                        std::span<const ConSanMoiAccessRecord> access_records,
                                        std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
                                        std::span<uint64_t> exact_shadow_entries);

[[nodiscard]] ConSanMoiRecordReplayResult
consan_moi_record_replay_access_records(ConSanMoiReportHeader &header,
                                        std::span<const ConSanMoiAccessRecord> access_records,
                                        std::span<const ConSanMoiBarrierRecord> barrier_records,
                                        std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
                                        std::span<uint64_t> exact_shadow_entries);

[[nodiscard]] ConSanMoiRecordReplayResult consan_moi_record_replay_access_records(
    ConSanMoiReportHeader &header, std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries);

[[nodiscard]] ConSanMoiRecordReplayResult consan_moi_record_replay_access_records(
    ConSanMoiReportHeader &header, std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<const ConSanMoiRecordReplayFenceEvent> fence_events,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries);

[[nodiscard]] ConSanMoiSampledPublishResult
consan_moi_sampled_publish_access_records(const ConSanMoiReportHeader &header,
                                          std::span<const ConSanMoiAccessRecord> access_records,
                                          std::span<uint64_t> sampled_watchpoint_entries);

/// Classifies the semantic host representation without truncating invalid
/// enum or range values into the packed descriptor.
[[nodiscard]] ConSanMoiSampledSyncClassification
classify_consan_moi_sampled_sync_metadata(const ConSanMoiSampledSyncMetadata &metadata);

[[nodiscard]] ConSanMoiSampledSyncEncodeResult
encode_consan_moi_sampled_sync_metadata(const ConSanMoiSampledSyncMetadata &metadata);

[[nodiscard]] ConSanMoiSampledSyncDecodeResult
decode_consan_moi_sampled_sync_metadata(const ConSanMoiSampledSyncMetadataPacked &packed);

/// Publishes at an already selected metadata index. It never probes or grows
/// the table: an occupied nonidentical slot is a collision, and an out-of-range
/// selected index is capacity exhaustion. This keeps Sampled bounded rather
/// than turning it into a Record/Replay stream.
[[nodiscard]] ConSanMoiSampledSyncPublishResult
consan_moi_sampled_publish_sync_metadata(std::span<ConSanMoiSampledSyncMetadataPacked> slots,
                                         uint32_t selected_slot,
                                         const ConSanMoiSampledSyncMetadata &metadata);

/// Returns true only for an exact, complete workgroup barrier sequence that
/// Sampled may use to advance its selected causal epoch. This deliberately
/// excludes individual signal/wait halves, dynamic IDs, cyclic forms, scalar
/// clauses, unresolved/multiple kernel owners, and ambiguous membership.
[[nodiscard]] bool
consan_moi_sampled_qualifies_barrier_sequence(const ConSanSyncSequence &sequence);

[[nodiscard]] ConSanMoiSampledPublishResult
consan_moi_sampled_publish_causal_windows(const ConSanMoiReportHeader &header,
                                          std::span<const ConSanMoiAccessRecord> access_records,
                                          uint32_t selection_stride, uint32_t selection_offset,
                                          std::span<uint64_t> sampled_watchpoint_entries,
                                          std::span<ConSanMoiSampledCausalWindow> causal_windows);

[[nodiscard]] ConSanMoiSampledReplayResult
consan_moi_sampled_replay_entries(ConSanMoiReportHeader &header,
                                  std::span<const uint64_t> sampled_watchpoint_entries,
                                  std::span<ConSanMoiDiagnosticRecord> diagnostic_records);

[[nodiscard]] ConSanMoiSampledReplayResult
consan_moi_sampled_replay_snapshots(ConSanMoiReportHeader &header,
                                    std::span<const ConSanMoiSampledSnapshotWords> snapshots,
                                    std::span<ConSanMoiDiagnosticRecord> diagnostic_records);

[[nodiscard]] ConSanMoiSampledReplayResult consan_moi_sampled_replay_causal_windows(
    ConSanMoiReportHeader &header, std::span<const uint64_t> sampled_watchpoint_entries,
    std::span<const ConSanMoiSampledCausalWindow> causal_windows,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records);

/// Begins a race-safe bounded claim. A Claimed result owns a Publishing slot
/// which must be committed or aborted; Busy asks the caller to drop/retry the
/// whole window without reading partially published key fields.
[[nodiscard]] ConSanMoiSampledClaimResult
consan_moi_sampled_begin_causal_claim(std::span<ConSanMoiSampledCausalWindow> windows,
                                      const ConSanMoiSampledCausalKey &key);

[[nodiscard]] bool consan_moi_sampled_commit_causal_claim(ConSanMoiSampledCausalWindow &window,
                                                          const ConSanMoiSampledCausalKey &key,
                                                          uint32_t first_entry,
                                                          uint32_t entry_count);

[[nodiscard]] bool consan_moi_sampled_abort_causal_claim(ConSanMoiSampledCausalWindow &window);

[[nodiscard]] bool consan_moi_supports_native_lds_record_replay_mnemonic(std::string_view mnemonic);

/// Return whether every MOI engine can lower this decoded FLAT LDS access.
[[nodiscard]] bool consan_moi_supports_flat_access_mnemonic(std::string_view mnemonic);

[[nodiscard]] ConSanResult try_patch_consan_moi(ConSanResult result, const ConSanOptions &options,
                                                std::span<const uint8_t> code_object_bytes,
                                                rj_code_arch_t arch);

/// Re-run only MOI planning, lowering, and validation from a semantic
/// inventory produced for the same bytes and engine. This is used after a
/// runtime-sized report buffer becomes available.
[[nodiscard]] ConSanResult retry_patch_consan_moi_from_inventory(
    ConSanResult inventory, const ConSanMoiReportRetryConfig &report,
    std::span<const uint8_t> code_object_bytes);

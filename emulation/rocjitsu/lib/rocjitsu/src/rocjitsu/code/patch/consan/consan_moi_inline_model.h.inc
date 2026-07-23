struct alignas(8) ConSanMoiInlineExactShadowSlot {
  uint64_t packed_access = 0;
  uint64_t dispatch_id = 0;
  uint32_t version = 0;
  uint32_t reserved = 0;
};

static_assert(sizeof(ConSanMoiInlineExactShadowSlot) == 24);
static_assert(alignof(ConSanMoiInlineExactShadowSlot) == 8);

/// Select the largest power-of-two dispatch fanout whose full-LDS exact shadow
/// consumes at most half of the automatic report ceiling. The remaining half
/// is available to the report header, diagnostics, atomic-release state, and
/// causal metadata.
[[nodiscard]] constexpr uint32_t
consan_moi_inline_exact_dispatch_bank_count_for_lds(uint64_t max_lds_bytes) {
  constexpr uint32_t kMaximumDispatchBankCount = 256u;
  constexpr uint32_t kGuestLdsGranuleBytes = 4u;
  const uint64_t lds_cell_count =
      (static_cast<uint64_t>(max_lds_bytes) + kGuestLdsGranuleBytes - 1u) / kGuestLdsGranuleBytes;
  const uint64_t exact_bytes_per_bank = lds_cell_count * sizeof(ConSanMoiInlineExactShadowSlot);
  const uint64_t exact_shadow_budget = kConSanMoiAutoReportBufferCeilingBytes / 2u;
  if (exact_bytes_per_bank > exact_shadow_budget)
    return 0u;
  uint32_t bank_count = 1u;
  while (bank_count < kMaximumDispatchBankCount &&
         exact_bytes_per_bank <= exact_shadow_budget / (bank_count * 2u)) {
    bank_count *= 2u;
  }
  return bank_count;
}

// A code object owns one report buffer while several hardware dispatches can
// execute concurrently. Hash the dispatch and workgroup identities into
// independent exact-shadow banks so correct, unrelated workgroups do not
// serialize every LDS cell through one versioned slot. The planner selects the
// largest fitting fanout for each report's LDS inventory and carries it in the
// layout; explicit legacy buffers retain the original maximum fanout.
inline constexpr uint32_t kConSanMoiInlineMaximumDispatchBankCount = 256u;
static_assert(consan_moi_inline_exact_dispatch_bank_count_for_lds(
                  consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_GFX1250)) != 0u,
              "the configured LDS aperture cannot fit one InlineShadow dispatch bank");

struct alignas(8) ConSanMoiInlineAtomicReleaseSlot {
  uint32_t version = 0;
  uint32_t owner_id = 0;
  uint32_t epoch_plus_one = 0;
  uint32_t workgroup_key = 0;
  uint64_t atomic_address = 0;
  uint64_t dispatch_id = 0;
};

static_assert(sizeof(ConSanMoiInlineAtomicReleaseSlot) == 32);

enum class ConSanMoiInlineReleaseClaim : uint8_t {
  Empty,
  ExactReady,
  InvalidIdentity,
  Publishing,
  UnstableRead,
  Collision,
  VersionExhausted,
};

struct ConSanMoiInlineVersionedReleaseIdentity {
  uint64_t dispatch_id = 0;
  uint64_t atomic_address = 0;
  uint32_t workgroup_key = 0;

  [[nodiscard]] constexpr bool valid() const {
    return dispatch_id != 0 && atomic_address != 0 && workgroup_key != 0;
  }
  constexpr auto operator<=>(const ConSanMoiInlineVersionedReleaseIdentity &) const = default;
};

struct ConSanMoiInlineVersionedReleaseState {
  uint32_t version = 0;
  ConSanMoiInlineVersionedReleaseIdentity identity;
};

struct ConSanMoiInlineReleaseClaimResult {
  ConSanMoiInlineReleaseClaim claim = ConSanMoiInlineReleaseClaim::Collision;
  uint32_t expected_version = 0;
  uint32_t publishing_version = 0;

  [[nodiscard]] constexpr bool can_publish() const {
    return claim == ConSanMoiInlineReleaseClaim::Empty ||
           claim == ConSanMoiInlineReleaseClaim::ExactReady;
  }
};

[[nodiscard]] constexpr bool consan_moi_inline_release_version_is_ready(uint32_t version) {
  return version != 0 && (version & 1u) == 0;
}

[[nodiscard]] constexpr bool consan_moi_inline_release_version_is_publishing(uint32_t version) {
  return (version & 1u) != 0;
}

[[nodiscard]] constexpr ConSanMoiInlineReleaseClaimResult
consan_moi_inline_plan_release_claim(const ConSanMoiInlineVersionedReleaseState &state,
                                     const ConSanMoiInlineVersionedReleaseIdentity &identity,
                                     uint32_t version_after_identity_read) {
  if (!identity.valid())
    return {ConSanMoiInlineReleaseClaim::InvalidIdentity, state.version, 0};
  if (state.version == 0)
    return {ConSanMoiInlineReleaseClaim::Empty, /*expected_version=*/0,
            /*publishing_version=*/1};
  if (consan_moi_inline_release_version_is_publishing(state.version))
    return {ConSanMoiInlineReleaseClaim::Publishing, state.version, 0};
  if (version_after_identity_read != state.version)
    return {ConSanMoiInlineReleaseClaim::UnstableRead, state.version, 0};
  if (state.identity != identity)
    return {ConSanMoiInlineReleaseClaim::Collision, state.version, 0};
  if (state.version >= std::numeric_limits<uint32_t>::max() - 1u)
    return {ConSanMoiInlineReleaseClaim::VersionExhausted, state.version, 0};
  return {ConSanMoiInlineReleaseClaim::ExactReady, state.version, state.version + 1u};
}

[[nodiscard]] constexpr bool
consan_moi_inline_release_claim_is_well_formed(const ConSanMoiInlineReleaseClaimResult &claim) {
  if (!claim.can_publish() ||
      !consan_moi_inline_release_version_is_publishing(claim.publishing_version) ||
      claim.publishing_version == std::numeric_limits<uint32_t>::max() ||
      claim.publishing_version != claim.expected_version + 1u)
    return false;
  if (claim.claim == ConSanMoiInlineReleaseClaim::Empty)
    return claim.expected_version == 0;
  return claim.claim == ConSanMoiInlineReleaseClaim::ExactReady &&
         consan_moi_inline_release_version_is_ready(claim.expected_version);
}

[[nodiscard]] constexpr bool
consan_moi_inline_release_claim_cas_succeeded(const ConSanMoiInlineReleaseClaimResult &claim,
                                              uint32_t returned_version) {
  return consan_moi_inline_release_claim_is_well_formed(claim) &&
         returned_version == claim.expected_version;
}

[[nodiscard]] constexpr uint32_t
consan_moi_inline_commit_release_version(const ConSanMoiInlineReleaseClaimResult &claim) {
  return consan_moi_inline_release_claim_is_well_formed(claim) ? claim.publishing_version + 1u : 0u;
}

[[nodiscard]] constexpr uint32_t
consan_moi_inline_restore_release_version(const ConSanMoiInlineReleaseClaimResult &claim) {
  return consan_moi_inline_release_claim_is_well_formed(claim) ? claim.expected_version : 0u;
}

/// This validates only the version envelope around a field/snapshot read. The
/// reader must separately validate every field and the complete snapshot.
[[nodiscard]] constexpr bool consan_moi_inline_release_snapshot_is_stable(uint32_t version_before,
                                                                          uint32_t version_after) {
  return version_before == version_after &&
         consan_moi_inline_release_version_is_ready(version_before);
}

enum class ConSanMoiInlineReleaseTransactionEvent : uint8_t {
  PriorSnapshot,
  Reserve,
  GuestAtomic,
  AcquireImport,
  Metadata,
  CausalSnapshot,
  CommitReady,
  RestorePrior,
  PoisonCoverage,
};

/// Reference ordering contract for the versioned release critical section.
/// A failed reservation poisons suppression before the guest proceeds. A
/// successful releasing outcome stays odd across the guest atomic and any
/// acquire import, then commits only after metadata and the causal snapshot.
/// A non-releasing CAS outcome restores the exact prior version instead.
[[nodiscard]] constexpr bool consan_moi_inline_release_transaction_is_sound(
    std::span<const ConSanMoiInlineReleaseTransactionEvent> events, bool claim_succeeded,
    bool dynamic_acquire_semantics, bool release_outcome, bool outcome_dependent_release) {
  const auto position = [&](ConSanMoiInlineReleaseTransactionEvent event) -> size_t {
    const auto found = std::find(events.begin(), events.end(), event);
    return found == events.end() ? events.size() : static_cast<size_t>(found - events.begin());
  };
  const auto exactly_once = [&](ConSanMoiInlineReleaseTransactionEvent event) {
    return std::count(events.begin(), events.end(), event) == 1;
  };
  if (!exactly_once(ConSanMoiInlineReleaseTransactionEvent::GuestAtomic))
    return false;
  const size_t guest = position(ConSanMoiInlineReleaseTransactionEvent::GuestAtomic);
  if (!claim_succeeded) {
    const size_t poison = position(ConSanMoiInlineReleaseTransactionEvent::PoisonCoverage);
    const size_t prior = position(ConSanMoiInlineReleaseTransactionEvent::PriorSnapshot);
    return exactly_once(ConSanMoiInlineReleaseTransactionEvent::PoisonCoverage) && poison < guest &&
           (prior == events.size() ||
            (exactly_once(ConSanMoiInlineReleaseTransactionEvent::PriorSnapshot) &&
             prior < poison)) &&
           position(ConSanMoiInlineReleaseTransactionEvent::Reserve) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::AcquireImport) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::Metadata) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::CausalSnapshot) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::CommitReady) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::RestorePrior) == events.size();
  }
  if (!exactly_once(ConSanMoiInlineReleaseTransactionEvent::Reserve) ||
      position(ConSanMoiInlineReleaseTransactionEvent::Reserve) >= guest ||
      position(ConSanMoiInlineReleaseTransactionEvent::PoisonCoverage) != events.size())
    return false;
  const size_t prior = position(ConSanMoiInlineReleaseTransactionEvent::PriorSnapshot);
  const size_t acquire = position(ConSanMoiInlineReleaseTransactionEvent::AcquireImport);
  if (dynamic_acquire_semantics != (acquire != events.size()) ||
      dynamic_acquire_semantics != (prior != events.size()) ||
      (dynamic_acquire_semantics &&
       (!exactly_once(ConSanMoiInlineReleaseTransactionEvent::PriorSnapshot) ||
        !exactly_once(ConSanMoiInlineReleaseTransactionEvent::AcquireImport) ||
        prior >= position(ConSanMoiInlineReleaseTransactionEvent::Reserve) || acquire <= guest)))
    return false;
  if (!release_outcome) {
    const size_t restore = position(ConSanMoiInlineReleaseTransactionEvent::RestorePrior);
    return outcome_dependent_release &&
           exactly_once(ConSanMoiInlineReleaseTransactionEvent::RestorePrior) && restore > guest &&
           (!dynamic_acquire_semantics || restore > acquire) &&
           position(ConSanMoiInlineReleaseTransactionEvent::Metadata) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::CausalSnapshot) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::CommitReady) == events.size();
  }
  const size_t metadata = position(ConSanMoiInlineReleaseTransactionEvent::Metadata);
  const size_t snapshot = position(ConSanMoiInlineReleaseTransactionEvent::CausalSnapshot);
  const size_t commit = position(ConSanMoiInlineReleaseTransactionEvent::CommitReady);
  const bool metadata_ordered = outcome_dependent_release || dynamic_acquire_semantics
                                    ? metadata > guest && snapshot > guest
                                    : metadata < guest && snapshot < guest;
  return exactly_once(ConSanMoiInlineReleaseTransactionEvent::Metadata) &&
         exactly_once(ConSanMoiInlineReleaseTransactionEvent::CausalSnapshot) &&
         exactly_once(ConSanMoiInlineReleaseTransactionEvent::CommitReady) && metadata_ordered &&
         (!dynamic_acquire_semantics || (metadata > acquire && snapshot > acquire)) &&
         commit > guest && commit > metadata && commit > snapshot &&
         position(ConSanMoiInlineReleaseTransactionEvent::RestorePrior) == events.size();
}

enum class ConSanMoiInlineTokenEvidenceKind : uint8_t {
  Direct,
  Inherited,
  // Carries an RMW release sequence into a later release snapshot without
  // granting acquire semantics to the intermediate release-only owner.
  ReleaseSequence,
};

struct alignas(8) ConSanMoiInlineAcquiredEpochTokenSlot {
  uint32_t version = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
  uint32_t workgroup_key = 0;
  uint32_t kind = 0;
  uint64_t dispatch_id = 0;
  uint64_t source_release_address = 0;
  uint32_t source_release_version = 0;
  uint32_t reserved = 0;

  constexpr auto operator<=>(const ConSanMoiInlineAcquiredEpochTokenSlot &) const = default;
};

static_assert(sizeof(ConSanMoiInlineAcquiredEpochTokenSlot) == 48);

inline constexpr uint32_t kConSanMoiInlineCausalSnapshotEntryCapacity = 4;

enum class ConSanMoiInlineCausalSnapshotFlag : uint32_t {
  None = 0,
  CapacityOverflow = 1u << 0u,
  SourceIncomplete = 1u << 1u,
  Malformed = 1u << 2u,
};

struct ConSanMoiInlineCausalSnapshotEntry {
  uint32_t ancestor_owner_id = 0;
  uint32_t ancestor_epoch_plus_one = 0;

  constexpr auto operator<=>(const ConSanMoiInlineCausalSnapshotEntry &) const = default;
};

/// Fixed-size immutable ancestry captured while the corresponding release slot
/// is odd. Entries are canonicalized by strictly increasing ancestor owner.
struct alignas(8) ConSanMoiInlineCausalSnapshot {
  uint32_t entry_count = 0;
  uint32_t flags = 0;
  std::array<ConSanMoiInlineCausalSnapshotEntry, kConSanMoiInlineCausalSnapshotEntryCapacity>
      entries{};
};

static_assert(sizeof(ConSanMoiInlineCausalSnapshotEntry) == 8);
static_assert(sizeof(ConSanMoiInlineCausalSnapshot) == 40);
static_assert(alignof(ConSanMoiInlineCausalSnapshot) == 8);

/// Stable-read token-table view used by release-time reference capture. The
/// same version must bracket every field; only nonzero even versions are ready.
struct alignas(8) ConSanMoiInlineCausalTokenView {
  uint32_t version_before = 0;
  uint32_t version_after = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_key = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
  ConSanMoiInlineTokenEvidenceKind kind = ConSanMoiInlineTokenEvidenceKind::Direct;
  uint64_t source_release_address = 0;
  uint32_t source_release_version = 0;
  uint32_t reserved = 0;
};

[[nodiscard]] constexpr uint32_t
consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag flag) {
  return static_cast<uint32_t>(flag);
}

[[nodiscard]] constexpr bool consan_moi_inline_causal_epoch_is_valid(uint32_t epoch_plus_one) {
  return epoch_plus_one != 0 && epoch_plus_one <= 1023u;
}

/// Captures the exact causal frontier owned by `releaser_owner_id`. Unrelated
/// ready tokens are ignored. Any malformed table entry, duplicate/cyclic
/// ancestry, incomplete scan, or fifth matching ancestor marks the complete
/// snapshot nonauthorizing.
[[nodiscard]] constexpr ConSanMoiInlineCausalSnapshot
consan_moi_inline_capture_causal_snapshot(std::span<const ConSanMoiInlineCausalTokenView> tokens,
                                          uint64_t dispatch_id, uint32_t workgroup_key,
                                          uint32_t releaser_owner_id, bool source_complete = true) {
  ConSanMoiInlineCausalSnapshot snapshot;
  if (!source_complete)
    snapshot.flags |=
        consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::SourceIncomplete);
  if (dispatch_id == 0 || workgroup_key == 0 || releaser_owner_id == 0)
    snapshot.flags |=
        consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);

  for (const auto &token : tokens) {
    if (token.version_before == 0 && token.version_after == 0 && token.dispatch_id == 0 &&
        token.workgroup_key == 0 && token.consumer_owner_id == 0 && token.producer_owner_id == 0 &&
        token.producer_epoch_plus_one == 0 && token.source_release_address == 0 &&
        token.source_release_version == 0 &&
        token.kind == ConSanMoiInlineTokenEvidenceKind::Direct && token.reserved == 0)
      continue;
    if (token.version_before != token.version_after || token.version_before == 0 ||
        (token.version_before & 1u) != 0 || token.dispatch_id == 0 || token.workgroup_key == 0 ||
        token.consumer_owner_id == 0 || token.producer_owner_id == 0 ||
        !consan_moi_inline_causal_epoch_is_valid(token.producer_epoch_plus_one) ||
        (token.kind != ConSanMoiInlineTokenEvidenceKind::Direct &&
         token.kind != ConSanMoiInlineTokenEvidenceKind::Inherited &&
         token.kind != ConSanMoiInlineTokenEvidenceKind::ReleaseSequence) ||
        token.source_release_address == 0 ||
        !consan_moi_inline_release_version_is_ready(token.source_release_version) ||
        token.reserved != 0) {
      snapshot.flags |=
          consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);
      continue;
    }
    if (token.dispatch_id != dispatch_id || token.workgroup_key != workgroup_key ||
        token.consumer_owner_id != releaser_owner_id)
      continue;
    if (token.producer_owner_id == releaser_owner_id) {
      snapshot.flags |=
          consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);
      continue;
    }

    uint32_t insertion = 0;
    while (insertion < snapshot.entry_count &&
           insertion < kConSanMoiInlineCausalSnapshotEntryCapacity &&
           snapshot.entries[insertion].ancestor_owner_id < token.producer_owner_id)
      ++insertion;
    if (insertion < snapshot.entry_count &&
        snapshot.entries[insertion].ancestor_owner_id == token.producer_owner_id) {
      snapshot.flags |=
          consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);
      continue;
    }
    if (snapshot.entry_count >= kConSanMoiInlineCausalSnapshotEntryCapacity) {
      snapshot.flags |= consan_moi_inline_causal_snapshot_flag(
          ConSanMoiInlineCausalSnapshotFlag::CapacityOverflow);
      continue;
    }
    for (uint32_t move = snapshot.entry_count; move > insertion; --move)
      snapshot.entries[move] = snapshot.entries[move - 1u];
    snapshot.entries[insertion] = {token.producer_owner_id, token.producer_epoch_plus_one};
    ++snapshot.entry_count;
  }
  return snapshot;
}

enum class ConSanMoiInlineCausalSnapshotStatus : uint8_t {
  Usable,
  CapacityOverflow,
  SourceIncomplete,
  Malformed,
};

[[nodiscard]] constexpr ConSanMoiInlineCausalSnapshotStatus
consan_moi_inline_validate_causal_snapshot(const ConSanMoiInlineCausalSnapshot &snapshot,
                                           uint32_t releaser_owner_id) {
  constexpr uint32_t known_flags =
      consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::CapacityOverflow) |
      consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::SourceIncomplete) |
      consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);
  if (releaser_owner_id == 0 ||
      snapshot.entry_count > kConSanMoiInlineCausalSnapshotEntryCapacity ||
      (snapshot.flags & ~known_flags) != 0 ||
      (snapshot.flags &
       consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed)) != 0)
    return ConSanMoiInlineCausalSnapshotStatus::Malformed;
  uint32_t prior_owner = 0;
  for (uint32_t i = 0; i < snapshot.entry_count; ++i) {
    const auto &entry = snapshot.entries[i];
    if (entry.ancestor_owner_id == 0 || entry.ancestor_owner_id == releaser_owner_id ||
        entry.ancestor_owner_id <= prior_owner ||
        !consan_moi_inline_causal_epoch_is_valid(entry.ancestor_epoch_plus_one))
      return ConSanMoiInlineCausalSnapshotStatus::Malformed;
    prior_owner = entry.ancestor_owner_id;
  }
  for (uint32_t i = snapshot.entry_count; i < kConSanMoiInlineCausalSnapshotEntryCapacity; ++i) {
    if (snapshot.entries[i] != ConSanMoiInlineCausalSnapshotEntry{})
      return ConSanMoiInlineCausalSnapshotStatus::Malformed;
  }
  if ((snapshot.flags & consan_moi_inline_causal_snapshot_flag(
                            ConSanMoiInlineCausalSnapshotFlag::CapacityOverflow)) != 0)
    return ConSanMoiInlineCausalSnapshotStatus::CapacityOverflow;
  if ((snapshot.flags & consan_moi_inline_causal_snapshot_flag(
                            ConSanMoiInlineCausalSnapshotFlag::SourceIncomplete)) != 0)
    return ConSanMoiInlineCausalSnapshotStatus::SourceIncomplete;
  return ConSanMoiInlineCausalSnapshotStatus::Usable;
}

struct ConSanMoiInlineStableReleaseSnapshot {
  uint32_t version_before = 0;
  uint32_t version_after = 0;
  ConSanMoiInlineVersionedReleaseIdentity identity;
  uint32_t releaser_owner_id = 0;
  uint32_t releaser_epoch_plus_one = 0;
  ConSanMoiInlineCausalSnapshot snapshot;
};

enum class ConSanMoiInlineReleaseSnapshotState : uint8_t {
  Empty,
  Stable,
  Publishing,
  ChangedDuringRead,
  CapacityOverflow,
  SourceIncomplete,
  Malformed,
};

struct ConSanMoiInlineReleaseSnapshotWords {
  uint32_t version_before = 0;
  ConSanMoiInlineAtomicReleaseSlot slot;
  ConSanMoiInlineCausalSnapshot snapshot;
  uint32_t version_after = 0;
};

struct ConSanMoiInlineClassifiedReleaseSnapshot {
  ConSanMoiInlineReleaseSnapshotState state = ConSanMoiInlineReleaseSnapshotState::Empty;
  ConSanMoiInlineStableReleaseSnapshot release;
};

[[nodiscard]] constexpr bool
consan_moi_inline_causal_snapshot_is_zero(const ConSanMoiInlineCausalSnapshot &snapshot) {
  if (snapshot.entry_count != 0 || snapshot.flags != 0)
    return false;
  for (const auto &entry : snapshot.entries) {
    if (entry != ConSanMoiInlineCausalSnapshotEntry{})
      return false;
  }
  return true;
}

/// Classifies a release and its parallel causal snapshot only after one
/// version has bracketed every field. No odd, changed, malformed, or partial
/// publication can become ordering authority.
[[nodiscard]] constexpr ConSanMoiInlineClassifiedReleaseSnapshot
classify_consan_moi_inline_release_snapshot(const ConSanMoiInlineReleaseSnapshotWords &words) {
  if (words.version_before != words.version_after || words.slot.version != words.version_before)
    return {.state = ConSanMoiInlineReleaseSnapshotState::ChangedDuringRead, .release = {}};
  if (consan_moi_inline_release_version_is_publishing(words.version_after))
    return {.state = ConSanMoiInlineReleaseSnapshotState::Publishing, .release = {}};
  if (words.version_after == 0) {
    if (words.slot.owner_id == 0 && words.slot.epoch_plus_one == 0 &&
        words.slot.workgroup_key == 0 && words.slot.atomic_address == 0 &&
        words.slot.dispatch_id == 0 && consan_moi_inline_causal_snapshot_is_zero(words.snapshot))
      return {.state = ConSanMoiInlineReleaseSnapshotState::Empty, .release = {}};
    return {.state = ConSanMoiInlineReleaseSnapshotState::Malformed, .release = {}};
  }
  const ConSanMoiInlineVersionedReleaseIdentity identity{
      words.slot.dispatch_id, words.slot.atomic_address, words.slot.workgroup_key};
  if (!identity.valid() || words.slot.owner_id == 0 ||
      !consan_moi_inline_causal_epoch_is_valid(words.slot.epoch_plus_one))
    return {.state = ConSanMoiInlineReleaseSnapshotState::Malformed, .release = {}};
  switch (consan_moi_inline_validate_causal_snapshot(words.snapshot, words.slot.owner_id)) {
  case ConSanMoiInlineCausalSnapshotStatus::CapacityOverflow:
    return {.state = ConSanMoiInlineReleaseSnapshotState::CapacityOverflow, .release = {}};
  case ConSanMoiInlineCausalSnapshotStatus::SourceIncomplete:
    return {.state = ConSanMoiInlineReleaseSnapshotState::SourceIncomplete, .release = {}};
  case ConSanMoiInlineCausalSnapshotStatus::Malformed:
    return {.state = ConSanMoiInlineReleaseSnapshotState::Malformed, .release = {}};
  case ConSanMoiInlineCausalSnapshotStatus::Usable:
    return {.state = ConSanMoiInlineReleaseSnapshotState::Stable,
            .release = {words.version_before, words.version_after, identity, words.slot.owner_id,
                        words.slot.epoch_plus_one, words.snapshot}};
  }
  return {.state = ConSanMoiInlineReleaseSnapshotState::Malformed, .release = {}};
}

struct ConSanMoiInlineCausalImportEntry {
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
};

enum class ConSanMoiInlineCausalImportStatus : uint8_t {
  Usable,
  InvalidConsumer,
  UnstableRelease,
  IdentityMismatch,
  CapacityOverflow,
  SourceIncomplete,
  MalformedSnapshot,
  DestinationCollision,
  DestinationCapacityExhausted,
};

struct ConSanMoiInlineCausalImportPlan {
  ConSanMoiInlineCausalImportStatus status = ConSanMoiInlineCausalImportStatus::MalformedSnapshot;
  uint32_t entry_count = 0;
  std::array<ConSanMoiInlineCausalImportEntry, kConSanMoiInlineCausalSnapshotEntryCapacity + 1u>
      entries{};

  [[nodiscard]] constexpr bool authoritative() const {
    return status == ConSanMoiInlineCausalImportStatus::Usable;
  }
};

/// Produces direct `(consumer,releaser)` first, then immutable inherited
/// `(consumer,ancestor)` tokens. The caller must preflight every destination
/// and pass collision/capacity state here before publishing any token; any
/// failure returns zero authorizing entries and poisons suppression for the
/// dispatch.
[[nodiscard]] constexpr ConSanMoiInlineCausalImportPlan consan_moi_inline_plan_causal_import(
    const ConSanMoiInlineStableReleaseSnapshot &release,
    const ConSanMoiInlineVersionedReleaseIdentity &expected_identity, uint32_t consumer_owner_id,
    bool destination_collision = false, bool destination_capacity_exhausted = false) {
  ConSanMoiInlineCausalImportPlan result;
  if (consumer_owner_id == 0 || consumer_owner_id == release.releaser_owner_id) {
    result.status = ConSanMoiInlineCausalImportStatus::InvalidConsumer;
    return result;
  }
  if (!consan_moi_inline_release_snapshot_is_stable(release.version_before,
                                                    release.version_after)) {
    result.status = ConSanMoiInlineCausalImportStatus::UnstableRelease;
    return result;
  }
  if (!expected_identity.valid() || release.identity != expected_identity) {
    result.status = ConSanMoiInlineCausalImportStatus::IdentityMismatch;
    return result;
  }
  if (release.releaser_owner_id == 0 ||
      !consan_moi_inline_causal_epoch_is_valid(release.releaser_epoch_plus_one)) {
    result.status = ConSanMoiInlineCausalImportStatus::MalformedSnapshot;
    return result;
  }
  switch (consan_moi_inline_validate_causal_snapshot(release.snapshot, release.releaser_owner_id)) {
  case ConSanMoiInlineCausalSnapshotStatus::CapacityOverflow:
    result.status = ConSanMoiInlineCausalImportStatus::CapacityOverflow;
    return result;
  case ConSanMoiInlineCausalSnapshotStatus::SourceIncomplete:
    result.status = ConSanMoiInlineCausalImportStatus::SourceIncomplete;
    return result;
  case ConSanMoiInlineCausalSnapshotStatus::Malformed:
    result.status = ConSanMoiInlineCausalImportStatus::MalformedSnapshot;
    return result;
  case ConSanMoiInlineCausalSnapshotStatus::Usable:
    break;
  }
  for (uint32_t i = 0; i < release.snapshot.entry_count; ++i) {
    if (release.snapshot.entries[i].ancestor_owner_id == consumer_owner_id) {
      result.status = ConSanMoiInlineCausalImportStatus::MalformedSnapshot;
      return result;
    }
  }
  if (destination_capacity_exhausted) {
    result.status = ConSanMoiInlineCausalImportStatus::DestinationCapacityExhausted;
    return result;
  }
  if (destination_collision) {
    result.status = ConSanMoiInlineCausalImportStatus::DestinationCollision;
    return result;
  }
  result.status = ConSanMoiInlineCausalImportStatus::Usable;
  result.entries[0] = {release.releaser_owner_id, release.releaser_epoch_plus_one};
  result.entry_count = release.snapshot.entry_count + 1u;
  for (uint32_t i = 0; i < release.snapshot.entry_count; ++i) {
    result.entries[i + 1u] = {release.snapshot.entries[i].ancestor_owner_id,
                              release.snapshot.entries[i].ancestor_epoch_plus_one};
  }
  return result;
}

struct ConSanMoiInlineStableReleaseEvidence {
  ConSanMoiInlineVersionedReleaseIdentity identity;
  uint32_t version_before = 0;
  uint32_t version_after = 0;
  uint32_t releaser_owner_id = 0;
  uint32_t releaser_epoch_plus_one = 0;
  ConSanMoiInlineCausalSnapshot snapshot;
};

struct ConSanMoiInlineTokenEvidence {
  uint64_t dispatch_id = 0;
  uint32_t workgroup_key = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
  ConSanMoiInlineTokenEvidenceKind kind = ConSanMoiInlineTokenEvidenceKind::Direct;
  ConSanMoiInlineVersionedReleaseIdentity source_release;
  uint32_t source_release_version = 0;
};

struct ConSanMoiInlineAccessEvidence {
  uint64_t dispatch_id = 0;
  uint32_t workgroup_key = 0;
  uint32_t owner_id = 0;
  uint32_t access_count = 0;
};

struct ConSanMoiInlineEvidenceCounters {
  uint64_t dispatch_id = 0;
  uint32_t dispatch_coverage_count = 0;
  uint32_t undercoverage_count = 0;
  uint32_t overflow_count = 0;
  uint32_t unsupported_count = 0;
  uint32_t consan_diagnostic_count = 0;
  uint32_t matched_source_diagnostic_count = 0;
};

enum class ConSanMoiInlineQualificationSemantics : uint8_t {
  Ordered,
  NativeRelaxed,
};

struct ConSanMoiInlineQualificationExpectation {
  ConSanMoiInlineQualificationSemantics semantics = ConSanMoiInlineQualificationSemantics::Ordered;
  ConSanMoiInlineVersionedReleaseIdentity release_identity;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t required_ancestor_owner_id = 0;
  uint32_t required_ancestor_epoch_plus_one = 0;
};

enum class ConSanMoiInlineQualificationFailure : uint32_t {
  None = 0,
  InvalidExpectation = 1u << 0u,
  DispatchCoverage = 1u << 1u,
  Undercoverage = 1u << 2u,
  Overflow = 1u << 3u,
  Unsupported = 1u << 4u,
  ProducerAccess = 1u << 5u,
  ConsumerAccess = 1u << 6u,
  StableRelease = 1u << 7u,
  DirectToken = 1u << 8u,
  InheritedToken = 1u << 9u,
  UnexpectedAuthorization = 1u << 10u,
  MissingSourceDiagnostic = 1u << 11u,
  UnexpectedDiagnostic = 1u << 12u,
  MalformedEvidence = 1u << 13u,
};

struct ConSanMoiInlineQualificationResult {
  uint32_t failures = 0;

  [[nodiscard]] constexpr bool accepted() const { return failures == 0; }
  [[nodiscard]] constexpr bool has(ConSanMoiInlineQualificationFailure failure) const {
    return (failures & static_cast<uint32_t>(failure)) != 0;
  }
};

[[nodiscard]] constexpr ConSanMoiInlineQualificationResult consan_moi_inline_qualify_token_evidence(
    const ConSanMoiInlineQualificationExpectation &expected,
    std::span<const ConSanMoiInlineStableReleaseEvidence> releases,
    std::span<const ConSanMoiInlineTokenEvidence> tokens,
    std::span<const ConSanMoiInlineAccessEvidence> accesses,
    const ConSanMoiInlineEvidenceCounters &counters) {
  ConSanMoiInlineQualificationResult result;
  const auto fail = [&](ConSanMoiInlineQualificationFailure failure) {
    result.failures |= static_cast<uint32_t>(failure);
  };
  const bool requires_ancestor = expected.required_ancestor_owner_id != 0;
  if (!expected.release_identity.valid() || expected.producer_owner_id == 0 ||
      !consan_moi_inline_causal_epoch_is_valid(expected.producer_epoch_plus_one) ||
      expected.consumer_owner_id == 0 || expected.consumer_owner_id == expected.producer_owner_id ||
      (requires_ancestor != (expected.required_ancestor_epoch_plus_one != 0)) ||
      (requires_ancestor &&
       (expected.required_ancestor_owner_id == expected.producer_owner_id ||
        expected.required_ancestor_owner_id == expected.consumer_owner_id ||
        !consan_moi_inline_causal_epoch_is_valid(expected.required_ancestor_epoch_plus_one))))
    fail(ConSanMoiInlineQualificationFailure::InvalidExpectation);

  if (counters.dispatch_id != expected.release_identity.dispatch_id ||
      counters.dispatch_coverage_count == 0)
    fail(ConSanMoiInlineQualificationFailure::DispatchCoverage);
  if (counters.undercoverage_count != 0)
    fail(ConSanMoiInlineQualificationFailure::Undercoverage);
  if (counters.overflow_count != 0)
    fail(ConSanMoiInlineQualificationFailure::Overflow);
  if (counters.unsupported_count != 0)
    fail(ConSanMoiInlineQualificationFailure::Unsupported);

  for (const auto &access : accesses) {
    if (access.dispatch_id == expected.release_identity.dispatch_id &&
        access.workgroup_key == expected.release_identity.workgroup_key &&
        (access.owner_id == 0 || access.access_count == 0))
      fail(ConSanMoiInlineQualificationFailure::MalformedEvidence);
  }
  const auto has_access = [&](uint32_t owner_id) {
    return std::any_of(accesses.begin(), accesses.end(), [&](const auto &access) {
      return access.dispatch_id == expected.release_identity.dispatch_id &&
             access.workgroup_key == expected.release_identity.workgroup_key &&
             access.owner_id == owner_id && access.access_count != 0;
    });
  };
  if (!has_access(expected.producer_owner_id))
    fail(ConSanMoiInlineQualificationFailure::ProducerAccess);
  if (!has_access(expected.consumer_owner_id))
    fail(ConSanMoiInlineQualificationFailure::ConsumerAccess);

  const ConSanMoiInlineStableReleaseEvidence *matching_release = nullptr;
  uint32_t relevant_release_count = 0;
  for (const auto &release : releases) {
    if (release.identity.dispatch_id != expected.release_identity.dispatch_id ||
        release.identity.workgroup_key != expected.release_identity.workgroup_key)
      continue;
    ++relevant_release_count;
    const bool structurally_valid =
        release.identity.valid() &&
        consan_moi_inline_release_snapshot_is_stable(release.version_before,
                                                     release.version_after) &&
        release.releaser_owner_id != 0 &&
        consan_moi_inline_causal_epoch_is_valid(release.releaser_epoch_plus_one) &&
        consan_moi_inline_validate_causal_snapshot(release.snapshot, release.releaser_owner_id) ==
            ConSanMoiInlineCausalSnapshotStatus::Usable;
    if (!structurally_valid) {
      fail(ConSanMoiInlineQualificationFailure::MalformedEvidence);
      continue;
    }
    if (release.identity == expected.release_identity &&
        release.releaser_owner_id == expected.producer_owner_id &&
        release.releaser_epoch_plus_one == expected.producer_epoch_plus_one) {
      if (matching_release != nullptr)
        fail(ConSanMoiInlineQualificationFailure::MalformedEvidence);
      matching_release = &release;
    }
  }

  uint32_t relevant_token_count = 0;
  uint32_t direct_token_count = 0;
  uint32_t inherited_token_count = 0;
  for (const auto &token : tokens) {
    if (token.dispatch_id != expected.release_identity.dispatch_id ||
        token.workgroup_key != expected.release_identity.workgroup_key)
      continue;
    ++relevant_token_count;
    const bool structurally_valid =
        token.consumer_owner_id != 0 && token.producer_owner_id != 0 &&
        token.consumer_owner_id != token.producer_owner_id &&
        consan_moi_inline_causal_epoch_is_valid(token.producer_epoch_plus_one) &&
        (token.kind == ConSanMoiInlineTokenEvidenceKind::Direct ||
         token.kind == ConSanMoiInlineTokenEvidenceKind::Inherited ||
         token.kind == ConSanMoiInlineTokenEvidenceKind::ReleaseSequence) &&
        token.source_release.valid() && token.source_release.dispatch_id == token.dispatch_id &&
        token.source_release.workgroup_key == token.workgroup_key &&
        consan_moi_inline_release_version_is_ready(token.source_release_version);
    if (!structurally_valid) {
      fail(ConSanMoiInlineQualificationFailure::MalformedEvidence);
      continue;
    }
    const bool expected_source = matching_release != nullptr &&
                                 token.source_release == matching_release->identity &&
                                 token.source_release_version == matching_release->version_after;
    if (expected_source && token.kind == ConSanMoiInlineTokenEvidenceKind::Direct &&
        token.consumer_owner_id == expected.consumer_owner_id &&
        token.producer_owner_id == expected.producer_owner_id &&
        token.producer_epoch_plus_one == expected.producer_epoch_plus_one)
      ++direct_token_count;
    if (expected_source && token.kind == ConSanMoiInlineTokenEvidenceKind::Inherited &&
        token.consumer_owner_id == expected.consumer_owner_id && requires_ancestor &&
        token.producer_owner_id == expected.required_ancestor_owner_id &&
        token.producer_epoch_plus_one == expected.required_ancestor_epoch_plus_one)
      ++inherited_token_count;
  }

  if (expected.semantics == ConSanMoiInlineQualificationSemantics::NativeRelaxed) {
    if (relevant_release_count != 0 || relevant_token_count != 0)
      fail(ConSanMoiInlineQualificationFailure::UnexpectedAuthorization);
    if (counters.matched_source_diagnostic_count == 0)
      fail(ConSanMoiInlineQualificationFailure::MissingSourceDiagnostic);
    if (counters.consan_diagnostic_count != counters.matched_source_diagnostic_count)
      fail(ConSanMoiInlineQualificationFailure::UnexpectedDiagnostic);
    return result;
  }

  if (matching_release == nullptr)
    fail(ConSanMoiInlineQualificationFailure::StableRelease);
  if (direct_token_count != 1)
    fail(ConSanMoiInlineQualificationFailure::DirectToken);
  if (requires_ancestor) {
    bool snapshot_contains_ancestor = false;
    if (matching_release != nullptr) {
      snapshot_contains_ancestor = std::any_of(
          matching_release->snapshot.entries.begin(),
          matching_release->snapshot.entries.begin() + matching_release->snapshot.entry_count,
          [&](const auto &entry) {
            return entry.ancestor_owner_id == expected.required_ancestor_owner_id &&
                   entry.ancestor_epoch_plus_one == expected.required_ancestor_epoch_plus_one;
          });
    }
    if (!snapshot_contains_ancestor || inherited_token_count != 1)
      fail(ConSanMoiInlineQualificationFailure::InheritedToken);
  }
  if (counters.consan_diagnostic_count != 0 || counters.matched_source_diagnostic_count != 0)
    fail(ConSanMoiInlineQualificationFailure::UnexpectedDiagnostic);
  return result;
}
// Token qualification uses EXEC-mask pairs through +21, local publication
// retains its validated subset through +23, and dynamic-stack/visible-evidence
// state can occupy +24:+25. Keep the application access mask disjoint from all
// of them for the complete probe lifetime. Dense call anchors retain their
// scalar return address in the final pair while the appended body runs.
inline constexpr uint16_t kConSanMoiInlineOriginalExecSaveOffset = 26u;
inline constexpr uint16_t kConSanMoiInlineExecSaveSgprCount = 30u;

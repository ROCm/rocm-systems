struct alignas(8) ConSanMoiInlineExactShadowSlot {
  uint64_t packed_access = 0;
  uint64_t dispatch_id = 0;
  uint32_t version = 0;
  uint32_t byte_provenance = 0;
};

static_assert(sizeof(ConSanMoiInlineExactShadowSlot) == 24);
static_assert(alignof(ConSanMoiInlineExactShadowSlot) == 8);

/// Select the largest power-of-two dispatch fanout whose byte-addressable
/// full-LDS exact shadow consumes at most half of the automatic report
/// ceiling. Subword objects need one independent slot per LDS byte so an
/// access to one byte cannot erase provenance for an adjacent byte. Ordinary
/// four-byte-cell objects use the same conservative allocation and simply
/// leave the additional slots unused. The remaining half is available to the
/// report header, diagnostics, atomic-release state, and causal metadata.
[[nodiscard]] constexpr uint32_t
consan_moi_inline_exact_dispatch_bank_count_for_lds(uint64_t max_lds_bytes) {
  constexpr uint32_t kMaximumDispatchBankCount = 256u;
  const uint64_t exact_bytes_per_bank =
      static_cast<uint64_t>(max_lds_bytes) * sizeof(ConSanMoiInlineExactShadowSlot);
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

struct alignas(8) ConSanMoiInlineAtomicReleaseSlot {
  uint32_t version = 0;
  uint32_t owner_id = 0;
  uint32_t epoch_plus_one = 0;
  uint32_t workgroup_key = 0;
  uint64_t atomic_address = 0;
  uint64_t dispatch_id = 0;
};

static_assert(sizeof(ConSanMoiInlineAtomicReleaseSlot) == 32);

struct ConSanMoiInlineVersionedReleaseIdentity {
  uint64_t dispatch_id = 0;
  uint64_t atomic_address = 0;
  uint32_t workgroup_key = 0;

  [[nodiscard]] constexpr bool valid() const {
    return dispatch_id != 0 && atomic_address != 0 && workgroup_key != 0;
  }
  constexpr auto operator<=>(const ConSanMoiInlineVersionedReleaseIdentity &) const = default;
};

[[nodiscard]] constexpr bool consan_moi_inline_release_version_is_ready(uint32_t version) {
  return version != 0 && (version & 1u) == 0;
}

[[nodiscard]] constexpr bool consan_moi_inline_release_version_is_publishing(uint32_t version) {
  return (version & 1u) != 0;
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
  // Segment epoch established by the acquire that published this token.
  // Access authorization also checks this bound so a later acquire cannot
  // retroactively order an earlier consumer access.
  uint32_t consumer_epoch_plus_one = 0;
  // Odd-version publication journal. Stable tokens always keep this zero;
  // publishers temporarily retain the replaced version here so a failed
  // multi-token transaction can roll back without losing payload state.
  uint32_t reservation_version = 0;

  constexpr auto operator<=>(const ConSanMoiInlineAcquiredEpochTokenSlot &) const = default;
};

static_assert(sizeof(ConSanMoiInlineAcquiredEpochTokenSlot) == 56);

/// Every 32-bit payload word whose zero state distinguishes an empty acquired
/// token slot from a partially published or stale token. The seqlock `version`
/// word is intentionally excluded because readers load it before and after the
/// payload; structure tail padding is likewise not evidence. Publication,
/// reuse, and final validation must consume this one ABI-derived list.
inline constexpr std::array<size_t, 12> kConSanMoiInlineAcquiredEpochTokenPayloadOffsets = {
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) + sizeof(uint32_t),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address) + sizeof(uint32_t),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_epoch_plus_one),
    offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version),
};

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

[[nodiscard]] constexpr uint32_t
consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag flag) {
  return static_cast<uint32_t>(flag);
}

[[nodiscard]] constexpr bool consan_moi_inline_causal_epoch_is_valid(uint32_t epoch_plus_one) {
  return epoch_plus_one != 0 && epoch_plus_one <= 1023u;
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

// Token qualification uses EXEC-mask pairs through +21, local publication
// retains its validated subset through +23, and dynamic-stack/visible-evidence
// state can occupy +24:+25. Keep the application access mask disjoint from all
// of them for the complete probe lifetime. Dense call anchors retain their
// scalar return address in the final pair while the appended body runs.
inline constexpr uint16_t kConSanMoiInlineOriginalExecSaveOffset = 26u;
inline constexpr uint16_t kConSanMoiInlineExecSaveSgprCount = 30u;

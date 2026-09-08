/// Ordinary automatic Record/Replay uses a cheap report-wide dispatch
/// directory plus one report-wide access-identity table. The two access-bank
/// dimensions are sizing factors: the planner reserves that many anticipated
/// identities per static range and distinct address group, applies standard
/// open-addressing headroom, rounds the total table to a power of two, and the
/// device probes by static range + dispatch + workgroup + wave + dynamic LDS
/// address. Hot sites can therefore borrow unused capacity from cold sites
/// without workload-specific sizing. Large code objects halve the dispatch,
/// owner, and address-group factors under the existing per-buffer ceiling;
/// direct layouts keep their historical single-identity contract.
inline constexpr uint32_t kConSanMoiRecordReplayMaximumDispatchBankCount = 16u;
inline constexpr uint32_t kConSanMoiRecordReplayMaximumOwnerBankCount = 16u;
inline constexpr uint32_t kConSanMoiRecordReplayMaximumAddressGroupsPerWave = 64u;
inline constexpr uint32_t kConSanMoiRecordReplayHashTableHeadroom = 2u;
inline constexpr uint32_t kConSanMoiRecordReplayAnticipatedDispatchCount = 1024u;
inline constexpr uint32_t kConSanMoiRecordReplayMaximumDispatchTokenCount =
    kConSanMoiRecordReplayAnticipatedDispatchCount * kConSanMoiRecordReplayHashTableHeadroom;
inline constexpr uint32_t kConSanMoiRecordReplayIdentityHashMultiplier = 0x9e3779b1u;
/// Device-side open-addressing must have a practical bound independent of the
/// report allocation size. Exhaustion sets the typed saturation flag; later
/// probes observe that flag and bypass capture instead of repeatedly walking a
/// full table. Large admitted inventories can legitimately drive the bounded
/// table above half occupancy, so retain a finite 1,024-step collision budget.
inline constexpr uint32_t kConSanMoiRecordReplayProbeLimit = 1024u;
[[nodiscard]] constexpr uint32_t consan_moi_record_replay_mix_identity(uint32_t hash,
                                                                       uint32_t coordinate) {
  return (hash * kConSanMoiRecordReplayIdentityHashMultiplier) ^ coordinate;
}
/// Advances an open-addressing probe after @p probe_count has been incremented.
/// Repeated calls therefore visit home + 1, home + 3, home + 6, ... . This
/// triangular sequence covers every slot in a power-of-two table.
/// @pre @p table_capacity is a nonzero power of two.
[[nodiscard]] constexpr uint32_t consan_moi_record_replay_advance_probe(uint32_t bank,
                                                                        uint32_t probe_count,
                                                                        uint32_t table_capacity) {
  return (bank + probe_count) & (table_capacity - 1u);
}
static_assert((kConSanMoiRecordReplayMaximumDispatchTokenCount &
               (kConSanMoiRecordReplayMaximumDispatchTokenCount - 1u)) == 0u);
static_assert(kConSanMoiRecordReplayMaximumDispatchTokenCount ==
              kConSanMoiRecordReplayAnticipatedDispatchCount *
                  kConSanMoiRecordReplayHashTableHeadroom);
static_assert((kConSanMoiRecordReplayMaximumDispatchBankCount &
               (kConSanMoiRecordReplayMaximumDispatchBankCount - 1u)) == 0u);
static_assert((kConSanMoiRecordReplayMaximumOwnerBankCount &
               (kConSanMoiRecordReplayMaximumOwnerBankCount - 1u)) == 0u);
static_assert((kConSanMoiRecordReplayMaximumAddressGroupsPerWave &
               (kConSanMoiRecordReplayMaximumAddressGroupsPerWave - 1u)) == 0u);
static_assert(kConSanMoiRecordReplayHashTableHeadroom >= 2u);
static_assert((kConSanMoiRecordReplayProbeLimit & (kConSanMoiRecordReplayProbeLimit - 1u)) == 0u);

// Record/Replay synchronization probes can execute many times per static
// site, and a code-object-scoped report remains live across repeated
// dispatches. Barriers publish once per wave, while atomic and fence probes
// publish once per participating lane.
inline constexpr uint64_t kConSanMoiRecordReplayDynamicEventHeadroom = 1024u;
inline constexpr uint64_t kConSanMoiRecordReplayDynamicLaneEventHeadroom =
    64u * kConSanMoiRecordReplayDynamicEventHeadroom;

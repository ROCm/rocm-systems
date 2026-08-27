struct alignas(8) ConSanMoiDiagnosticRecord {
  uint32_t kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::None);
  uint32_t backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
  // Report generation for ordinary records; native dispatch identity for
  // InlineShadow atomic-tracking diagnostics.
  uint64_t generation = 0;
  uint32_t epoch = 0;
  // Epoch of the prior exact-shadow access. InlineShadow atomic-tracking
  // diagnostics use this with the token's consumer segment epoch to prove
  // that an acquire preceded that specific access.
  uint32_t first_epoch = 0;
  uint32_t first_owner_id = 0;
  uint32_t second_owner_id = 0;
  // Record/Replay host diagnostics retain the current event index here so
  // byte-range provenance can be joined to the exact dynamic record.
  // InlineShadow atomic-tracking diagnostics instead retain the persistent
  // workgroup key so teardown can qualify a token publication that completed
  // after the device-side diagnostic snapshot.
  uint32_t reserved = 0;
  uint64_t first_lane_mask = 0;
  uint64_t second_lane_mask = 0;
  uint32_t first_instruction_offset = 0;
  uint32_t second_instruction_offset = 0;
  uint32_t first_lds_byte_offset = 0;
  uint32_t first_lds_byte_count = 0;
  uint32_t second_lds_byte_offset = 0;
  uint32_t second_lds_byte_count = 0;
  uint32_t first_access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
  uint32_t second_access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
};

/// Host-populated lookup record for the compact workgroup-local shadow.
/// Compact cells retain a per-kernel token instead of a full instruction
/// offset. Keeping one descriptor-qualified record per patched site in the
/// report allocation lets teardown resolve that token without an unbounded
/// process-heap side table.
struct alignas(8) ConSanMoiCompactDiagnosticTokenMapping {
  uint64_t owner_descriptor_file_offset = 0;
  uint32_t instruction_offset = 0;
  uint16_t token = 0;
  uint16_t reserved = 0;
};
static_assert(sizeof(ConSanMoiCompactDiagnosticTokenMapping) == 16);

/// Concrete byte geometry of one legacy MOI report allocation.
///
/// Capacities state how many ABI records each region can hold; offsets name
/// their aligned locations relative to the beginning of the allocation.
/// Unused regions alias the end of the allocation and have zero capacity.
/// This value contains no device address, allocation handle, generation, or
/// lifetime. It is the current ABI-layout implementation detail of typed
/// evidence requirements and remains independently revalidated at late bind.
struct ConSanMoiReportBufferLayout {
  uint32_t access_record_capacity = 0;
  uint32_t record_replay_logical_access_range_count = 0;
  uint32_t record_replay_dispatch_token_capacity = 0;
  uint32_t record_replay_access_dispatch_bank_count = 1;
  uint32_t record_replay_access_owner_bank_count = 1;
  uint32_t record_replay_address_group_headroom = 1;
  uint32_t barrier_record_capacity = 0;
  uint32_t atomic_record_capacity = 0;
  uint32_t fence_record_capacity = 0;
  uint32_t diagnostic_capacity = 0;
  uint32_t exact_shadow_entry_capacity = 0;
  uint32_t inline_exact_dispatch_bank_count = 0;
  uint32_t inline_atomic_release_capacity = 0;
  uint32_t inline_acquired_epoch_token_capacity = 0;
  uint32_t inline_causal_snapshot_capacity = 0;
  uint32_t inline_compact_token_mapping_capacity = 0;
  uint32_t sampled_watchpoint_capacity = 0;
  uint32_t sampled_causal_window_capacity = 0;
  uint32_t sampled_sync_metadata_capacity = 0;
  uint32_t sampled_pending_acquire_capacity = 0;
  size_t record_replay_dispatch_tokens_offset = sizeof(ConSanMoiReportHeader);
  size_t access_records_offset = sizeof(ConSanMoiReportHeader);
  size_t barrier_records_offset = sizeof(ConSanMoiReportHeader);
  size_t atomic_records_offset = sizeof(ConSanMoiReportHeader);
  size_t fence_records_offset = sizeof(ConSanMoiReportHeader);
  size_t diagnostic_records_offset = sizeof(ConSanMoiReportHeader);
  size_t exact_shadow_entries_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_atomic_release_slots_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_acquired_epoch_token_slots_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_causal_snapshots_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_compact_token_mappings_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_watchpoints_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_causal_windows_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_sync_metadata_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_pending_acquires_offset = sizeof(ConSanMoiReportHeader);
  size_t required_bytes = sizeof(ConSanMoiReportHeader);
  bool valid = false;

  bool operator==(const ConSanMoiReportBufferLayout &) const = default;
};

/// Creates a header from the exact heterogeneous layout. The legacy
/// make_consan_moi_report_header helper intentionally keeps its historical
/// single Sampled capacity contract for explicit size-derived buffers.
[[nodiscard]] constexpr ConSanMoiReportHeader
make_consan_moi_report_header_for_layout(uint64_t generation, uint64_t dispatch_id,
                                         const ConSanMoiReportBufferLayout &layout,
                                         ConSanMoiEngine engine) {
  ConSanMoiReportHeader header;
  header.header_size = sizeof(ConSanMoiReportHeader);
  header.generation = generation;
  header.dispatch_id = dispatch_id;
  header.engine = static_cast<uint32_t>(engine);
  header.record_replay_dispatch_token_capacity = layout.record_replay_dispatch_token_capacity;
  header.access_record_capacity = layout.access_record_capacity;
  header.barrier_record_capacity = layout.barrier_record_capacity;
  header.atomic_record_capacity = layout.atomic_record_capacity;
  header.fence_record_capacity = layout.fence_record_capacity;
  header.diagnostic_capacity = layout.diagnostic_capacity;
  header.exact_shadow_entry_capacity = layout.exact_shadow_entry_capacity;
  header.inline_atomic_release_capacity = layout.inline_atomic_release_capacity;
  header.inline_acquired_epoch_token_capacity = layout.inline_acquired_epoch_token_capacity;
  header.inline_causal_snapshot_capacity = layout.inline_causal_snapshot_capacity;
  header.sampled_watchpoint_capacity = layout.sampled_watchpoint_capacity;
  header.sampled_causal_window_capacity = layout.sampled_causal_window_capacity;
  header.sampled_sync_metadata_capacity = layout.sampled_sync_metadata_capacity;
  header.sampled_pending_acquire_capacity = layout.sampled_pending_acquire_capacity;
  if (engine == ConSanMoiEngine::InlineShadow)
    header.layout_flags = kConSanMoiReportKnownLayoutFlags;
  return header;
}

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
[[nodiscard]] constexpr uint32_t consan_moi_record_replay_fold_dispatch_token(uint64_t token) {
  return static_cast<uint32_t>(token) ^ static_cast<uint32_t>(token >> 32u);
}
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

/// Final-code inventory reduced to the quantities needed to size one report
/// ABI. Counts are deliberately independent: candidate extraction must not
/// assume that one sampled causal window holds only one watchpoint, or that
/// all Inline ordering tables have the same capacity.
struct ConSanMoiAutoReportInventory {
  ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay;
  uint64_t access_range_count = 0;
  uint64_t barrier_event_count = 0;
  uint64_t atomic_event_count = 0;
  uint64_t fence_event_count = 0;
  uint64_t diagnostic_count = 0;

  /// Automatic static capture starts with bounded dispatch, owner, and
  /// address-group diversity factors used only to size the report-wide
  /// access-identity table. The fitter first reduces dispatch and owner
  /// together, then address-group headroom when a very large object would
  /// otherwise exceed the ordinary report ceiling. Its separate dispatch
  /// directory retains the full lifetime bound while access storage adapts.
  /// Explicit plans remain exact.
  uint64_t record_replay_dispatch_token_capacity = kConSanMoiRecordReplayMaximumDispatchTokenCount;
  uint64_t record_replay_access_dispatch_bank_count =
      kConSanMoiRecordReplayMaximumDispatchBankCount;
  uint64_t record_replay_access_owner_bank_count = kConSanMoiRecordReplayMaximumOwnerBankCount;
  uint64_t record_replay_address_group_headroom = kConSanMoiRecordReplayMaximumAddressGroupsPerWave;
  bool record_replay_bank_count_adaptive = false;

  /// Total immutable banks across admitted Sampled logical ranges. Atomic
  /// synchronization lowering can reserve additional tail slots in the
  /// causal, synchronization-metadata, and pending-acquire arrays. Multi-cell
  /// ranges can independently require more watchpoints.
  uint64_t sampled_range_bank_count = 0;
  uint64_t sampled_sync_slot_count = 0;
  uint64_t sampled_watchpoint_count = 0;
  /// Automatic runtime sampling may reduce the number of independent banks,
  /// while retaining at least one bank for every logical access range, to fit
  /// the frozen per-object report ceiling. Explicit inventories remain exact.
  bool sampled_bank_count_adaptive = false;

  /// Maximum LDS span covered by an admitted Inline site. Native sites use the
  /// owning kernel's declared segment; generic-flat sites require the full
  /// architectural aperture because their runtime offset is not statically
  /// bounded by that declaration. The planner reserves one external exact
  /// shadow slot per byte so mixed-width objects can use byte-granular
  /// provenance without a second report layout.
  uint64_t inline_lds_bytes = 0;
  uint64_t inline_atomic_release_count = 0;
  uint64_t inline_causal_snapshot_count = 0;
  uint64_t inline_acquired_epoch_token_count = 0;
  uint64_t inline_compact_token_mapping_count = 0;
  /// Automatic inventories request one wave of diagnostic headroom per
  /// logical range in every exact-shadow dispatch bank. Large objects may
  /// reduce only that headroom to the largest capacity fitting the frozen
  /// report ceiling. Runtime saturation remains visible and fail-closed.
  bool inline_diagnostic_count_adaptive = false;

  /// Return whether the inventory describes any admitted access or ordering
  /// observation. ABI-only header storage and Record/Replay hash-table
  /// headroom do not by themselves require a runtime report binding.
  [[nodiscard]] constexpr bool has_semantic_observations() const {
    return access_range_count != 0u || barrier_event_count != 0u || atomic_event_count != 0u ||
           fence_event_count != 0u;
  }

  bool operator==(const ConSanMoiAutoReportInventory &) const = default;
};

using ConSanMoiAutoReportCapacityMember = uint64_t ConSanMoiAutoReportInventory::*;

/// Canonical projection of every numeric report-capacity dimension. Keep
/// capacity comparisons expressed through this list so a dimension cannot
/// acquire a separate, silently incomplete comparison path.
inline constexpr std::array<ConSanMoiAutoReportCapacityMember, 17>
    kConSanMoiAutoReportCapacityMembers = {
        &ConSanMoiAutoReportInventory::access_range_count,
        &ConSanMoiAutoReportInventory::barrier_event_count,
        &ConSanMoiAutoReportInventory::atomic_event_count,
        &ConSanMoiAutoReportInventory::fence_event_count,
        &ConSanMoiAutoReportInventory::diagnostic_count,
        &ConSanMoiAutoReportInventory::record_replay_dispatch_token_capacity,
        &ConSanMoiAutoReportInventory::record_replay_access_dispatch_bank_count,
        &ConSanMoiAutoReportInventory::record_replay_access_owner_bank_count,
        &ConSanMoiAutoReportInventory::record_replay_address_group_headroom,
        &ConSanMoiAutoReportInventory::sampled_range_bank_count,
        &ConSanMoiAutoReportInventory::sampled_sync_slot_count,
        &ConSanMoiAutoReportInventory::sampled_watchpoint_count,
        &ConSanMoiAutoReportInventory::inline_lds_bytes,
        &ConSanMoiAutoReportInventory::inline_atomic_release_count,
        &ConSanMoiAutoReportInventory::inline_causal_snapshot_count,
        &ConSanMoiAutoReportInventory::inline_acquired_epoch_token_count,
        &ConSanMoiAutoReportInventory::inline_compact_token_mapping_count,
};

/// Return whether a report sized for @p capacity can represent every quantity
/// required by @p required.
[[nodiscard]] constexpr bool
consan_moi_auto_report_inventory_covers(const ConSanMoiAutoReportInventory &capacity,
                                        const ConSanMoiAutoReportInventory &required) {
  return capacity.engine == required.engine &&
         std::ranges::all_of(kConSanMoiAutoReportCapacityMembers,
                             [&](ConSanMoiAutoReportCapacityMember member) {
                               return capacity.*member >= required.*member;
                             });
}

/// High-level outcome of converting address-free capacities into ABI bytes.
///
/// The outcome separates a representable request that exceeds a configured
/// ceiling from arithmetic or ABI-width overflow. Only `Complete` permits a
/// runtime coordinator to allocate and bind the associated layout.
enum class ConSanMoiAutoReportPlanOutcome : uint8_t {
  /// Every required region is representable and fits the active ceiling.
  Complete,
  /// The geometry is representable but needs more bytes than the ceiling.
  InsufficientReportCapacity,
  /// A count, multiplication, alignment, or ABI field cannot be represented.
  Overflow,
  /// One-past-the-end iteration bound; never a planner outcome.
  Count,
};

/// Machine-readable explanation for an automatic report-plan outcome.
///
/// `None` accompanies only a complete plan. The remaining values distinguish
/// policy ceilings from fixed ABI field widths, Record/Replay hash-table
/// geometry, and host byte-address arithmetic so callers never parse log text.
enum class ConSanMoiAutoReportPlanReason : uint8_t {
  /// Planning completed without a limitation.
  None,
  /// The representable allocation exceeds the selected per-buffer ceiling.
  PerBufferCeiling,
  /// A logical capacity cannot fit its fixed-width ABI field.
  AbiCapacityOverflow,
  /// Record/Replay identity-table geometry exceeds its supported ABI domain.
  AbiGeometryCapacityOverflow,
  /// Byte multiplication, alignment, or the host size type overflowed.
  ByteSizeOverflow,
  /// One-past-the-end iteration bound; never a planner reason.
  Count,
};

/// Address-free result of laying out one legacy MOI evidence buffer.
///
/// A complete value owns exact region capacities, offsets, and total bytes.
/// Incomplete values retain the typed outcome and, where representable, the
/// byte requirement that exceeded policy. The value never owns an allocation
/// and may therefore be computed before runtime capability discovery.
struct ConSanMoiAutoReportPlan {
  ConSanMoiAutoReportPlanOutcome outcome = ConSanMoiAutoReportPlanOutcome::Overflow;
  ConSanMoiAutoReportPlanReason reason = ConSanMoiAutoReportPlanReason::ByteSizeOverflow;
  ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay;
  uint64_t required_bytes = 0;
  uint64_t ceiling_bytes = kConSanMoiAutoReportBufferCeilingBytes;
  ConSanMoiReportBufferLayout layout;

  [[nodiscard]] constexpr bool complete() const {
    return outcome == ConSanMoiAutoReportPlanOutcome::Complete;
  }

  bool operator==(const ConSanMoiAutoReportPlan &) const = default;
};

/// Typed failure to derive evidence requirements from semantic policy.
///
/// Capacity insufficiency is deliberately not a construction failure: it is a
/// valid ABI-plan outcome retained inside the requirements. These reasons are
/// reserved for an invalid input contract or an incompatible engine.
enum class ConSanEvidenceRequirementReason : uint8_t {
  /// Requirements were derived from a valid matching observation plan.
  None,
  /// The observation plan violates its own identity or relationship rules.
  InvalidObservationPlan,
  /// The planner was invoked for a different engine's observation plan.
  WrongEngine,
  /// A valid plan contains an intent belonging to another evidence schema.
  UnexpectedIntentKind,
  /// An intent covers semantic-site domains incompatible with its record kind.
  InvalidIntentPayload,
  /// A semantic intent could not be joined to the immutable program fact
  /// needed to size or interpret its evidence.
  MissingInventoryFact,
  /// One-past-the-end iteration bound; never a construction reason.
  Count,
};

/// Complete iterable set of evidence-requirement construction reasons.
inline constexpr std::array<ConSanEvidenceRequirementReason, 6> kConSanEvidenceRequirementReasons =
    {
        ConSanEvidenceRequirementReason::None,
        ConSanEvidenceRequirementReason::InvalidObservationPlan,
        ConSanEvidenceRequirementReason::WrongEngine,
        ConSanEvidenceRequirementReason::UnexpectedIntentKind,
        ConSanEvidenceRequirementReason::InvalidIntentPayload,
        ConSanEvidenceRequirementReason::MissingInventoryFact,
};

/// Return the stable diagnostic spelling of a requirement-construction reason.
[[nodiscard]] constexpr std::string_view
consan_evidence_requirement_reason_name(ConSanEvidenceRequirementReason reason) {
  switch (reason) {
  case ConSanEvidenceRequirementReason::None:
    return "none";
  case ConSanEvidenceRequirementReason::InvalidObservationPlan:
    return "invalid-observation-plan";
  case ConSanEvidenceRequirementReason::WrongEngine:
    return "wrong-engine";
  case ConSanEvidenceRequirementReason::UnexpectedIntentKind:
    return "unexpected-intent-kind";
  case ConSanEvidenceRequirementReason::InvalidIntentPayload:
    return "invalid-intent-payload";
  case ConSanEvidenceRequirementReason::MissingInventoryFact:
    return "missing-inventory-fact";
  case ConSanEvidenceRequirementReason::Count:
    break;
  }
  return "invalid-evidence-requirement-reason";
}

static_assert(kConSanEvidenceRequirementReasons.size() ==
              static_cast<size_t>(ConSanEvidenceRequirementReason::Count));

/// Caller policy controlling bounded Record/Replay evidence capacity.
///
/// A zero byte ceiling means the engine safety ceiling. The optional physical
/// access-probe limit models only the explicit expert patch limit; absence
/// means that every admitted access intent contributes requirements. It does
/// not cap barrier, atomic, or fence intents and contains no report address.
struct ConSanRecordReplayCapacityPolicy {
  /// Optional caller cap, bounded internally by the engine safety ceiling.
  uint64_t caller_ceiling_bytes = 0;
  /// Maximum admitted physical access intents, or no limit when absent.
  std::optional<uint64_t> maximum_access_probe_count;

  bool operator==(const ConSanRecordReplayCapacityPolicy &) const = default;
};

/// Address-free Record/Replay evidence contract for one code object.
///
/// The immutable observation plan determines logical record counts; capacity
/// policy supplies only capacity choices. This alternative intrinsically
/// denotes bounded first-light evidence whose loss invalidates completeness
/// and whose concrete binding has executable lifetime. `sizing_inventory` is the exact
/// input to the current ABI layout implementation and `abi_plan` is its output.
/// Runtime requirements state allocation properties without an address or
/// handle. A capacity-limited or overflowing ABI plan remains a well-formed
/// requirement and can be reported without attempting allocation.
struct ConSanRecordReplayEvidenceRequirements {
  /// Runtime facts required before allocation or executable binding.
  RuntimeCapabilityRequirements runtime_requirements;
  /// Exact address-free capacities consumed by the current ABI layout planner.
  ConSanMoiAutoReportInventory sizing_inventory;
  /// Exact byte geometry or typed capacity/overflow outcome.
  ConSanMoiAutoReportPlan abi_plan;
  /// Construction result, distinct from `abi_plan` capacity outcomes.
  ConSanEvidenceRequirementReason reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;

  /// Return whether all cross-type relationships form a usable requirement.
  [[nodiscard]] bool well_formed() const;

  /// Return whether a well-formed requirement has an allocatable ABI layout.
  [[nodiscard]] bool complete() const { return well_formed() && abi_plan.complete(); }
  /// Return whether a complete report must be bound before lowering can use
  /// this contract. An empty plan's header-only ABI is not a binding request.
  [[nodiscard]] bool requires_binding() const {
    return complete() && sizing_inventory.has_semantic_observations();
  }

  bool operator==(const ConSanRecordReplayEvidenceRequirements &) const = default;
};

/// Derive the complete address-free Record/Replay evidence contract from an
/// immutable semantic plan. The function performs no allocation, reads no
/// code-object bytes, and does not inspect patches or resource assignments.
[[nodiscard]] ConSanRecordReplayEvidenceRequirements
plan_consan_record_replay_evidence(const ConSanObservationPlan &observation_plan,
                                   const ConSanRecordReplayCapacityPolicy &capacity_policy = {});

/// Caller policy controlling bounded Sampled evidence capacity.
///
/// The optional physical-probe limit carries the explicit expert admission
/// bound; it does not truncate logical ranges within an admitted instruction
/// or synchronization intents. The byte ceiling affects only ABI fitting.
/// Neither member contains a report address or mutable lowering state.
struct ConSanSampledCapacityPolicy {
  /// Optional caller cap, bounded internally by the engine safety ceiling.
  uint64_t caller_ceiling_bytes = 0;
  /// Maximum admitted physical access intents, or no limit when absent.
  std::optional<uint64_t> maximum_access_probe_count;

  bool operator==(const ConSanSampledCapacityPolicy &) const = default;
};

/// Address-free Sampled evidence contract for one code object.
///
/// The observation plan supplies logical access windows and ordering events;
/// the capacity policy supplies only explicit bounds. The sizing inventory and
/// ABI plan preserve the current report representation without making it the
/// semantic interface. This alternative intrinsically denotes bounded sampled
/// evidence whose loss invalidates completeness and whose concrete binding has
/// executable lifetime. Runtime requirements describe that binding but never
/// own it.
struct ConSanSampledEvidenceRequirements {
  /// Runtime facts required before allocation or executable binding.
  RuntimeCapabilityRequirements runtime_requirements;
  /// Exact address-free capacities consumed by the current ABI planner.
  ConSanMoiAutoReportInventory sizing_inventory;
  /// Exact byte geometry or typed capacity/overflow outcome.
  ConSanMoiAutoReportPlan abi_plan;
  /// Construction result, distinct from `abi_plan` capacity outcomes.
  ConSanEvidenceRequirementReason reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;

  /// Return whether all cross-type relationships form a usable requirement.
  [[nodiscard]] bool well_formed() const;
  /// Return whether a well-formed requirement has an allocatable ABI layout.
  [[nodiscard]] bool complete() const { return well_formed() && abi_plan.complete(); }
  /// Return whether a complete report must be bound before lowering can use
  /// this contract. An empty plan's header-only ABI is not a binding request.
  [[nodiscard]] bool requires_binding() const {
    return complete() && sizing_inventory.has_semantic_observations();
  }

  bool operator==(const ConSanSampledEvidenceRequirements &) const = default;
};

/// Caller policy controlling bounded InlineShadow evidence capacity.
///
/// `maximum_workgroup_lds_bytes` is the runtime- or caller-supplied aperture
/// used when an admitted group-FLAT or dynamic-LDS access cannot be bounded by
/// a kernel's fixed descriptor declaration. Its absence selects the immutable
/// target profile. The policy owns neither original ELF bytes nor a report.
struct ConSanInlineShadowCapacityPolicy {
  /// Optional caller cap, bounded internally by the engine safety ceiling.
  uint64_t caller_ceiling_bytes = 0;
  /// Maximum admitted physical access intents, or no limit when absent.
  std::optional<uint64_t> maximum_access_probe_count;
  /// Explicit full LDS aperture used for address-free shadow sizing.
  std::optional<uint32_t> maximum_workgroup_lds_bytes;

  bool operator==(const ConSanInlineShadowCapacityPolicy &) const = default;
};

/// Address-free InlineShadow evidence contract for one code object.
///
/// Exact shadow geometry is derived by joining admitted intents to immutable
/// access provenance, execution owners, and pristine kernel-descriptor LDS
/// declarations. `required_lds_aperture_bytes` makes that semantic derivation
/// explicit beside the current ABI plan. The type contains no ELF view,
/// transformed descriptor, report address, or device allocation. This
/// alternative intrinsically denotes exact on-device conflict decisions with
/// bounded diagnostics; any evidence loss invalidates completeness and the
/// concrete binding has executable lifetime.
struct ConSanInlineShadowEvidenceRequirements {
  /// Runtime facts required before allocation or executable binding.
  RuntimeCapabilityRequirements runtime_requirements;
  /// Largest byte aperture that any admitted exact-shadow access can address.
  uint64_t required_lds_aperture_bytes = 0;
  /// Exact address-free capacities consumed by the current ABI planner.
  ConSanMoiAutoReportInventory sizing_inventory;
  /// Exact byte geometry or typed capacity/overflow outcome.
  ConSanMoiAutoReportPlan abi_plan;
  /// Construction result, distinct from `abi_plan` capacity outcomes.
  ConSanEvidenceRequirementReason reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;

  /// Return whether all cross-type relationships form a usable requirement.
  [[nodiscard]] bool well_formed() const;
  /// Return whether a well-formed requirement has an allocatable ABI layout.
  [[nodiscard]] bool complete() const { return well_formed() && abi_plan.complete(); }
  /// Return whether a complete report must be bound before lowering can use
  /// this contract. An empty plan's header-only ABI is not a binding request.
  [[nodiscard]] bool requires_binding() const {
    return complete() && sizing_inventory.has_semantic_observations();
  }

  bool operator==(const ConSanInlineShadowEvidenceRequirements &) const = default;
};

/// Address-free SuperCollider sticky-marker contract for one code object.
///
/// SuperCollider does not share an MOI record union: redundant execution
/// publishes a fixed sticky mismatch marker. `marker_bytes` is zero when the
/// valid observation plan contains no admitted access and otherwise states the
/// complete marker requirement. Runtime properties and executable lifetime
/// remain explicit even though the current registry implements the allocation.
/// This alternative intrinsically denotes a sticky marker whose loss
/// invalidates completeness and whose concrete binding has executable lifetime.
struct ConSanSuperColliderEvidenceRequirements {
  /// Runtime facts required before marker allocation or executable binding.
  RuntimeCapabilityRequirements runtime_requirements;
  /// Required sticky-marker storage; zero means no admitted observation.
  uint64_t marker_bytes = 0;
  /// Construction result for the marker contract.
  ConSanEvidenceRequirementReason reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;

  /// Return whether all cross-type relationships form a usable requirement.
  [[nodiscard]] bool well_formed() const;
  /// Return whether the marker requirement is internally usable.
  [[nodiscard]] bool complete() const { return well_formed(); }
  /// Return whether a concrete marker allocation must be bound.
  [[nodiscard]] bool requires_binding() const { return complete() && marker_bytes != 0u; }

  bool operator==(const ConSanSuperColliderEvidenceRequirements &) const = default;
};

/// Closed set of engine-specific evidence contracts passed from semantic
/// policy to runtime coordination.
///
/// Each alternative is its own schema and retains its schema-specific
/// capacities; this variant
/// is an interface over distinct contracts, not a largest-union report layout.
/// Visitors can ask common questions without assuming that an MOI record array
/// and a SuperCollider marker have compatible bytes.
using ConSanEvidenceRequirements =
    std::variant<ConSanRecordReplayEvidenceRequirements, ConSanSampledEvidenceRequirements,
                 ConSanInlineShadowEvidenceRequirements, ConSanSuperColliderEvidenceRequirements>;

/// Return whether an engine-specific evidence requirement is internally
/// consistent and can safely cross the runtime-coordination boundary.
[[nodiscard]] bool
consan_evidence_requirements_well_formed(const ConSanEvidenceRequirements &requirements);

/// Derive Sampled evidence solely from a typed observation plan and capacity
/// policy. The function reads no code bytes, patches, or resource assignments.
[[nodiscard]] ConSanSampledEvidenceRequirements
plan_consan_sampled_evidence(const ConSanObservationPlan &observation_plan,
                             const ConSanSampledCapacityPolicy &capacity_policy = {});

/// Derive InlineShadow evidence from typed policy plus immutable original-
/// program facts. The function reads no code bytes, emitted patches, or
/// mutable resource assignments.
[[nodiscard]] ConSanInlineShadowEvidenceRequirements
plan_consan_inline_shadow_evidence(const ProgramInventory &inventory,
                                   const ConSanObservationPlan &observation_plan,
                                   const ConSanInlineShadowCapacityPolicy &capacity_policy = {});

/// Derive the fixed SuperCollider marker contract from a typed observation
/// plan. No report-layout or allocation mechanism participates.
[[nodiscard]] ConSanSuperColliderEvidenceRequirements
plan_consan_supercollider_evidence(const ConSanObservationPlan &observation_plan);

/// Converts a complete planner result into the caller-carried exact layout.
/// An incomplete plan produces std::nullopt.
[[nodiscard]] std::optional<ConSanMoiReportLayoutOverride>
consan_moi_auto_report_layout_override(const ConSanMoiAutoReportPlan &plan);

/// Revalidates a caller-carried layout against the canonical planner and the
/// registered allocation size. A malformed, noncanonical, wrong-engine, or
/// undersized override returns an invalid layout.
[[nodiscard]] ConSanMoiReportBufferLayout
consan_moi_report_layout_from_override(const ConSanMoiReportLayoutOverride &override_layout,
                                       ConSanMoiEngine engine, uint64_t report_buffer_size);

[[nodiscard]] constexpr std::string_view
consan_moi_auto_report_plan_outcome_name(ConSanMoiAutoReportPlanOutcome outcome) {
  switch (outcome) {
  case ConSanMoiAutoReportPlanOutcome::Complete:
    return "complete";
  case ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity:
    return "insufficient_report_capacity";
  case ConSanMoiAutoReportPlanOutcome::Overflow:
    return "overflow";
  case ConSanMoiAutoReportPlanOutcome::Count:
    break;
  }
  return "invalid_auto_report_plan_outcome";
}

[[nodiscard]] constexpr std::string_view
consan_moi_auto_report_plan_reason_name(ConSanMoiAutoReportPlanReason reason) {
  switch (reason) {
  case ConSanMoiAutoReportPlanReason::None:
    return "none";
  case ConSanMoiAutoReportPlanReason::PerBufferCeiling:
    return "per_buffer_ceiling";
  case ConSanMoiAutoReportPlanReason::AbiCapacityOverflow:
    return "abi_capacity_overflow";
  case ConSanMoiAutoReportPlanReason::AbiGeometryCapacityOverflow:
    return "abi_geometry_capacity_overflow";
  case ConSanMoiAutoReportPlanReason::ByteSizeOverflow:
    return "byte_size_overflow";
  case ConSanMoiAutoReportPlanReason::Count:
    break;
  }
  return "invalid_auto_report_plan_reason";
}

/// Computes an exact ABI layout from explicit inventory without allocating,
/// reading code-object bytes, or consulting environment state. Counts that do
/// not fit the ABI produce `overflow`; a representable plan above the ordinary
/// per-buffer ceiling produces `insufficient_report_capacity` while retaining
/// required bytes and capacities for telemetry. A nonzero caller ceiling is
/// bounded by the engine safety ceiling and lets an explicit runtime cap
/// participate in adaptive planning.
[[nodiscard]] ConSanMoiAutoReportPlan
plan_consan_moi_auto_report(const ConSanMoiAutoReportInventory &inventory,
                            uint64_t caller_ceiling_bytes = 0);

/// Expands static Record/Replay synchronization counts by the largest
/// power-of-two dynamic headroom that fits the ordinary per-buffer ceiling.
/// Small objects retain the standard headroom; large objects retain every
/// static site with smaller dispatch, owner, address-group, and event
/// headroom. Runtime saturation remains explicit and fail-closed.
[[nodiscard]] ConSanMoiAutoReportInventory
fit_consan_moi_record_replay_auto_report_inventory(ConSanMoiAutoReportInventory inventory,
                                                   uint64_t caller_ceiling_bytes = 0);

/// Reduces only an explicitly adaptive Sampled inventory by powers of two
/// until its complete logical-range population fits the report ceiling or one
/// bank per range remains. All other inventory dimensions remain exact.
[[nodiscard]] ConSanMoiAutoReportInventory
fit_consan_moi_sampled_auto_report_inventory(ConSanMoiAutoReportInventory inventory,
                                             uint64_t caller_ceiling_bytes = 0);

/// Reduces only an explicitly adaptive Inline Shadow diagnostic inventory to
/// the largest capacity that fits the ordinary report ceiling. Exact-shadow
/// and causal-metadata capacities remain unchanged.
[[nodiscard]] ConSanMoiAutoReportInventory
fit_consan_moi_inline_auto_report_inventory(ConSanMoiAutoReportInventory inventory,
                                            uint64_t caller_ceiling_bytes = 0);

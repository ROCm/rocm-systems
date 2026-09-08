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
inline constexpr std::array<ConSanMoiAutoReportCapacityMember, 16>
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
  uint64_t required_bytes = 0;
  uint64_t ceiling_bytes = 0;
  ConSanMoiReportBufferLayout layout;

  [[nodiscard]] constexpr bool complete() const {
    return outcome == ConSanMoiAutoReportPlanOutcome::Complete;
  }
  /// Return the canonical layout only when planning completed consistently.
  [[nodiscard]] std::optional<ConSanMoiReportBufferLayout> complete_layout() const {
    if (!complete() || !layout.valid || required_bytes != layout.required_bytes)
      return std::nullopt;
    return layout;
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

/// Buffer-independent role played by one admitted observation in the
/// evidence protocol.
///
/// This vocabulary is deliberately smaller than `ConSanProbeIntentKind`.
/// Probe kinds retain flavor policy (for example, sampled versus exact access),
/// while these roles state only what evidence must survive for report sizing
/// and interpretation. The layout planner therefore consumes roles without
/// reinterpreting flavor policy or examining instruction bytes.
enum class ConSanEvidenceIntentKind : uint8_t {
  /// Retain evidence for every logical memory range covered by the probe.
  Access,
  /// Retain a barrier event or completed barrier-sequence epoch.
  Barrier,
  /// Retain ordering evidence for one admitted atomic communication.
  Atomic,
  /// Retain one Record/Replay fence-association event.
  Fence,
  /// Preserve an effective address for a later synchronization observation;
  /// this operation consumes lowering state but no independent report slot.
  AddressCapture,
  /// Publish SuperCollider's fixed sticky mismatch marker.
  StickyMarker,
  /// One-past-the-end iteration bound; never an evidence intent.
  Count,
};

/// Evidence planning reads the authoritative observation intents directly.
/// The evidence role and element count are deterministic views of an intent;
/// retaining a second copied intent sequence would create another identity
/// and synchronization boundary without owning any new fact.
[[nodiscard]] std::optional<ConSanEvidenceIntentKind>
consan_evidence_intent_kind(ConSanCapabilityEngine engine, ConSanProbeIntentKind kind);

[[nodiscard]] std::optional<uint64_t>
consan_evidence_element_count(ConSanCapabilityEngine engine, const ConSanProbeIntent &intent);

/// Report-backed evidence contract shared by the three MOI engines.
///
/// This base owns only the representation-independent runtime and ABI product.
/// Each mode retains its semantic validation and any additional requirements
/// in the corresponding derived contract.
struct ConSanMoiEvidenceRequirements {
  RuntimeCapabilityRequirements runtime_requirements;
  ConSanMoiAutoReportInventory sizing_inventory;
  ConSanMoiAutoReportPlan abi_plan;
  ConSanEvidenceRequirementReason reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;

  [[nodiscard]] bool common_well_formed(ConSanMoiEngine expected_engine) const;

  bool operator==(const ConSanMoiEvidenceRequirements &) const = default;
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
struct ConSanRecordReplayEvidenceRequirements : ConSanMoiEvidenceRequirements {
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

/// Address-free Sampled evidence contract for one code object.
///
/// The observation plan supplies logical access windows and ordering events;
/// the capacity policy supplies only explicit bounds. The sizing inventory and
/// ABI plan preserve the current report representation without making it the
/// semantic interface. This alternative intrinsically denotes bounded sampled
/// evidence whose loss invalidates completeness and whose concrete binding has
/// executable lifetime. Runtime requirements describe that binding but never
/// own it.
struct ConSanSampledEvidenceRequirements : ConSanMoiEvidenceRequirements {
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
struct ConSanInlineShadowEvidenceRequirements : ConSanMoiEvidenceRequirements {
  /// Largest byte aperture that any admitted exact-shadow access can address.
  uint64_t required_lds_aperture_bytes = 0;

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

/// Address-free SuperCollider mismatch-evidence contract for one code object.
///
/// SuperCollider does not share an MOI record union. Trap-only evidence owns no
/// runtime storage; sticky-marker evidence publishes one fixed mismatch word.
/// `marker_bytes` is zero for trap-only evidence or an empty marker plan and
/// otherwise states the complete allocation requirement.
struct ConSanSuperColliderEvidenceRequirements {
  /// Requested mismatch-evidence protocol.
  ConSanSuperColliderEvidenceMode mode = ConSanSuperColliderEvidenceMode::StickyMarker;
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

/// Derive the fixed SuperCollider marker contract from a typed observation
/// plan. No report-layout or allocation mechanism participates.
[[nodiscard]] ConSanSuperColliderEvidenceRequirements plan_consan_supercollider_evidence(
    const ConSanObservationPlan &observation_plan,
    ConSanSuperColliderEvidenceMode mode = ConSanSuperColliderEvidenceMode::StickyMarker);

/// Revalidates a caller-carried layout against the canonical planner and the
/// registered allocation size. A malformed, noncanonical, wrong-engine, or
/// undersized candidate returns an invalid layout.
[[nodiscard]] ConSanMoiReportBufferLayout
revalidate_consan_moi_report_layout(const ConSanMoiReportBufferLayout &candidate,
                                    ConSanMoiEngine engine, uint64_t report_buffer_size);

[[nodiscard]] constexpr std::string_view
consan_moi_auto_report_plan_outcome_name(ConSanMoiAutoReportPlanOutcome outcome) {
  using E = ConSanMoiAutoReportPlanOutcome;
  constexpr auto vocabulary = make_consan_enum_vocabulary(
      "invalid_auto_report_plan_outcome", consan_enum(E::Complete, "complete"),
      consan_enum(E::InsufficientReportCapacity, "insufficient_report_capacity"),
      consan_enum(E::Overflow, "overflow"));
  return vocabulary.name(outcome);
}

[[nodiscard]] constexpr std::string_view
consan_moi_auto_report_plan_reason_name(ConSanMoiAutoReportPlanReason reason) {
  using E = ConSanMoiAutoReportPlanReason;
  constexpr auto vocabulary = make_consan_enum_vocabulary(
      "invalid_auto_report_plan_reason", consan_enum(E::None, "none"),
      consan_enum(E::PerBufferCeiling, "per_buffer_ceiling"),
      consan_enum(E::AbiCapacityOverflow, "abi_capacity_overflow"),
      consan_enum(E::AbiGeometryCapacityOverflow, "abi_geometry_capacity_overflow"),
      consan_enum(E::ByteSizeOverflow, "byte_size_overflow"));
  return vocabulary.name(reason);
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

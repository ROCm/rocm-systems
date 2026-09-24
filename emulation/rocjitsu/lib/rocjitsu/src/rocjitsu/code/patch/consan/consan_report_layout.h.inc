/// Final-code inventory reduced to the quantities needed to size one report
/// ABI. Counts are deliberately independent: candidate extraction must not
/// assume that one sampled causal window holds only one watchpoint, or that
/// all synchronization tables have the same capacity.
struct AutoReportInventory {
  uint64_t access_range_count = 0;
  uint64_t barrier_event_count = 0;
  uint64_t atomic_event_count = 0;

  /// Total immutable banks across admitted ConSan logical ranges. Atomic
  /// synchronization lowering can reserve additional tail slots in the
  /// causal, synchronization-metadata, and pending-acquire arrays. Multi-cell
  /// ranges can independently require more watchpoints.
  uint64_t range_bank_count = 0;
  uint64_t sync_slot_count = 0;
  uint64_t watchpoint_count = 0;
  /// Automatic runtime sampling may reduce the number of independent banks,
  /// while retaining at least one bank for every logical access range, to fit
  /// the frozen per-object report ceiling. Explicit inventories remain exact.
  bool bank_count_adaptive = false;

  /// Return whether the inventory describes any admitted access or ordering
  /// observation. ABI-only header storage does not require a runtime binding.
  [[nodiscard]] constexpr bool has_semantic_observations() const {
    return access_range_count != 0u || barrier_event_count != 0u || atomic_event_count != 0u;
  }

  bool operator==(const AutoReportInventory &) const = default;
};

static_assert(sizeof(AutoReportInventory) <= 56);

using AutoReportCapacityMember = uint64_t AutoReportInventory::*;

/// Canonical projection of every numeric report-capacity dimension. Keep
/// capacity comparisons expressed through this list so a dimension cannot
/// acquire a separate, silently incomplete comparison path.
inline constexpr std::array<AutoReportCapacityMember, 6> kAutoReportCapacityMembers = {
    &AutoReportInventory::access_range_count, &AutoReportInventory::barrier_event_count,
    &AutoReportInventory::atomic_event_count, &AutoReportInventory::range_bank_count,
    &AutoReportInventory::sync_slot_count,    &AutoReportInventory::watchpoint_count,
};

/// Return whether a report sized for @p capacity can represent every quantity
/// required by @p required.
[[nodiscard]] constexpr bool auto_report_inventory_covers(const AutoReportInventory &capacity,
                                                          const AutoReportInventory &required) {
  return std::ranges::all_of(kAutoReportCapacityMembers, [&](AutoReportCapacityMember member) {
    return capacity.*member >= required.*member;
  });
}

/// High-level outcome of converting address-free capacities into ABI bytes.
///
/// The outcome separates a representable request that exceeds a configured
/// ceiling from arithmetic or ABI-width overflow. Only `Complete` permits a
/// runtime coordinator to allocate and bind the associated layout.
enum class AutoReportPlanOutcome : uint8_t {
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
/// policy ceilings from fixed ABI field widths
/// and host byte-address arithmetic so callers never parse log text.
enum class AutoReportPlanReason : uint8_t {
  /// Planning completed without a limitation.
  None,
  /// The representable allocation exceeds the selected per-buffer ceiling.
  PerBufferCeiling,
  /// A logical capacity cannot fit its fixed-width ABI field.
  AbiCapacityOverflow,
  /// Byte multiplication, alignment, or the host size type overflowed.
  ByteSizeOverflow,
  /// One-past-the-end iteration bound; never a planner reason.
  Count,
};

/// Address-free result of laying out one ConSan evidence buffer.
///
/// A complete value owns exact region capacities, offsets, and total bytes.
/// Incomplete values retain the typed outcome and, where representable, the
/// byte requirement that exceeded policy. The value never owns an allocation
/// and may therefore be computed before runtime capability discovery.
struct AutoReportPlan {
  AutoReportPlanOutcome outcome = AutoReportPlanOutcome::Overflow;
  AutoReportPlanReason reason = AutoReportPlanReason::ByteSizeOverflow;
  uint64_t required_bytes = 0;
  uint64_t ceiling_bytes = 0;
  ReportBufferLayout layout;

  [[nodiscard]] constexpr bool complete() const {
    return outcome == AutoReportPlanOutcome::Complete;
  }
  /// Return the canonical layout only when planning completed consistently.
  [[nodiscard]] std::optional<ReportBufferLayout> complete_layout() const {
    if (!complete() || !layout.valid || required_bytes != layout.required_bytes)
      return std::nullopt;
    return layout;
  }

  bool operator==(const AutoReportPlan &) const = default;
};

/// Typed failure to derive evidence requirements from semantic policy.
///
/// Capacity insufficiency is deliberately not a construction failure: it is a
/// valid ABI-plan outcome retained inside the requirements. These reasons are
/// reserved for an invalid input contract or an incompatible mode.
enum class EvidenceRequirementReason : uint8_t {
  /// Requirements were derived from a valid matching observation plan.
  None,
  /// The observation plan violates its own identity or relationship rules.
  InvalidObservationPlan,
  /// The planner was invoked for a different mode's observation plan.
  WrongMode,
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
/// This vocabulary is deliberately smaller than `ProbeIntentKind`.
/// Probe kinds retain mode policy (for example, sampled versus exact access),
/// while these roles state only what evidence must survive for report sizing
/// and interpretation. The layout planner therefore consumes roles without
/// reinterpreting mode policy or examining instruction bytes.
enum class EvidenceIntentKind : uint8_t {
  /// Retain evidence for every logical memory range covered by the probe.
  Access,
  /// Retain a barrier event or completed barrier-sequence epoch.
  Barrier,
  /// Retain ordering evidence for one admitted atomic communication.
  Atomic,
  /// Preserve an effective address for a later synchronization observation;
  /// this operation consumes lowering state but no independent report slot.
  AddressCapture,
  /// Retain a modification footprint independently of synchronization roles.
  PublicationModification,
  /// Preserve the address for a publication modification footprint.
  PublicationAddressCapture,
  /// Publish SuperCollider's fixed sticky mismatch marker.
  StickyMarker,
  /// One-past-the-end iteration bound; never an evidence intent.
  Count,
};

/// Evidence planning reads the authoritative observation intents directly.
/// The evidence role and element count are deterministic views of an intent;
/// retaining a second copied intent sequence would create another identity
/// and synchronization boundary without owning any new fact.
[[nodiscard]] std::optional<EvidenceIntentKind> evidence_intent_kind(Mode mode,
                                                                     ProbeIntentKind kind);

[[nodiscard]] std::optional<uint64_t> evidence_element_count(Mode mode, const ProbeIntent &intent);

/// Address-free ConSan evidence contract for one code object.
///
/// The observation plan supplies logical access windows and ordering events;
/// the capacity policy supplies only explicit bounds. The sizing inventory and
/// ABI plan preserve the current report representation without making it the
/// semantic interface. This alternative intrinsically denotes bounded sampled
/// evidence whose loss invalidates completeness and whose concrete binding has
/// executable lifetime. Runtime requirements describe that binding but never
/// own it.
struct ReportRequirements {
  RuntimeCapabilityRequirements runtime_requirements;
  AutoReportInventory sizing_inventory;
  AutoReportPlan abi_plan;
  EvidenceRequirementReason reason = EvidenceRequirementReason::InvalidObservationPlan;

  /// Return whether all cross-type relationships form a usable requirement.
  [[nodiscard]] bool well_formed() const;
  /// Return whether a well-formed requirement has an allocatable ABI layout.
  [[nodiscard]] bool complete() const { return well_formed() && abi_plan.complete(); }
  /// Return whether a complete report must be bound before lowering can use
  /// this contract. An empty plan's header-only ABI is not a binding request.
  [[nodiscard]] bool requires_binding() const {
    return complete() && sizing_inventory.has_semantic_observations();
  }

  bool operator==(const ReportRequirements &) const = default;
};

/// Address-free SuperCollider mismatch-evidence contract for one code object.
///
/// SuperCollider does not share a ConSan record union. Trap-only evidence owns no
/// runtime storage; sticky-marker evidence publishes one fixed mismatch word.
/// `marker_bytes` is zero for trap-only evidence or an empty marker plan and
/// otherwise states the complete allocation requirement.
struct SuperColliderEvidenceRequirements {
  /// Requested mismatch-evidence protocol.
  SuperColliderEvidenceMode mode = SuperColliderEvidenceMode::StickyMarker;
  /// Runtime facts required before marker allocation or executable binding.
  RuntimeCapabilityRequirements runtime_requirements;
  /// Required sticky-marker storage; zero means no admitted observation.
  uint64_t marker_bytes = 0;
  /// Construction result for the marker contract.
  EvidenceRequirementReason reason = EvidenceRequirementReason::InvalidObservationPlan;

  /// Return whether all cross-type relationships form a usable requirement.
  [[nodiscard]] bool well_formed() const;
  /// Return whether the marker requirement is internally usable.
  [[nodiscard]] bool complete() const { return well_formed(); }
  /// Return whether a concrete marker allocation must be bound.
  [[nodiscard]] bool requires_binding() const { return complete() && marker_bytes != 0u; }

  bool operator==(const SuperColliderEvidenceRequirements &) const = default;
};

/// Closed set of mode-specific evidence contracts passed from semantic
/// policy to runtime coordination.
///
/// Each alternative is its own schema and retains its schema-specific
/// capacities. This variant presents common queries over those contracts.
/// Visitors can ask common questions without assuming that a ConSan record array
/// and a SuperCollider marker have compatible bytes.
using EvidenceRequirements = std::variant<ReportRequirements, SuperColliderEvidenceRequirements>;

/// Return whether a mode-specific evidence requirement is internally
/// consistent and can safely cross the runtime-coordination boundary.
[[nodiscard]] bool evidence_requirements_well_formed(const EvidenceRequirements &requirements);

/// Derive the fixed SuperCollider marker contract from a typed observation
/// plan. No report-layout or allocation mechanism participates.
[[nodiscard]] SuperColliderEvidenceRequirements plan_supercollider_evidence(
    const ObservationPlan &observation_plan,
    SuperColliderEvidenceMode mode = SuperColliderEvidenceMode::StickyMarker);

/// Revalidates a caller-carried layout against the canonical planner and the
/// registered allocation size. A malformed, noncanonical, or undersized
/// candidate returns an invalid layout.
[[nodiscard]] ReportBufferLayout revalidate_report_layout(const ReportBufferLayout &candidate,
                                                          uint64_t report_buffer_size);

[[nodiscard]] constexpr std::string_view
auto_report_plan_outcome_name(AutoReportPlanOutcome outcome) {
  using E = AutoReportPlanOutcome;
  constexpr auto vocabulary = make_enum_vocabulary(
      "invalid_auto_report_plan_outcome", enum_entry(E::Complete, "complete"),
      enum_entry(E::InsufficientReportCapacity, "insufficient_report_capacity"),
      enum_entry(E::Overflow, "overflow"));
  return vocabulary.name(outcome);
}

[[nodiscard]] constexpr std::string_view auto_report_plan_reason_name(AutoReportPlanReason reason) {
  using E = AutoReportPlanReason;
  constexpr auto vocabulary =
      make_enum_vocabulary("invalid_auto_report_plan_reason", enum_entry(E::None, "none"),
                           enum_entry(E::PerBufferCeiling, "per_buffer_ceiling"),
                           enum_entry(E::AbiCapacityOverflow, "abi_capacity_overflow"),
                           enum_entry(E::ByteSizeOverflow, "byte_size_overflow"));
  return vocabulary.name(reason);
}

/// Computes an exact ABI layout from explicit inventory without allocating,
/// reading code-object bytes, or consulting environment state. Counts that do
/// not fit the ABI produce `overflow`; a representable plan above the ordinary
/// per-buffer ceiling produces `insufficient_report_capacity` while retaining
/// required bytes and capacities for telemetry. A nonzero caller ceiling is
/// bounded by the mode safety ceiling and lets an explicit runtime cap
/// participate in adaptive planning.
[[nodiscard]] AutoReportPlan plan_auto_report(const AutoReportInventory &inventory,
                                              uint64_t caller_ceiling_bytes = 0);

/// Reduces only an explicitly adaptive ConSan inventory by powers of two
/// until its complete logical-range population fits the report ceiling or one
/// bank per range remains. All other inventory dimensions remain exact.
[[nodiscard]] AutoReportInventory fit_auto_report_inventory(AutoReportInventory inventory,
                                                            uint64_t caller_ceiling_bytes = 0);

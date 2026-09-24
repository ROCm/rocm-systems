// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Stable identifier of one target-neutral instrumentation request in a
/// `ObservationPlan`.
///
/// The identifier is an index into the plan's immutable intent sequence, not a
/// machine-code address or patch ordinal. Several semantic range decisions can
/// intentionally cite the same intent when one physical probe observes them
/// together. The maximum 32-bit value is reserved as the invalid default so a
/// zero-initialized identifier cannot silently select the first intent.
struct ProbeIntentId {
  /// Reserved value used by the default constructor and rejected by plans.
  static constexpr uint32_t invalid_value = std::numeric_limits<uint32_t>::max();

  /// Zero-based position in the owning observation plan, or `invalid_value`.
  uint32_t value = invalid_value;

  /// Return whether this value can name an entry in some observation plan.
  /// The owning plan must still perform the upper-bound check.
  [[nodiscard]] constexpr bool valid() const { return value != invalid_value; }

  auto operator<=>(const ProbeIntentId &) const = default;
};

/// Policy answer for one semantic site before register or placement planning.
///
/// `NotApplicable` is an intentional exclusion from the selected mode's
/// contract. `Unsupported` belongs to that contract but lacks a required
/// decoded fact or qualified implementation. `Admitted` requests one or more
/// probe intents. Lowering is not permitted to rewrite any of these answers;
/// its independent outcome is recorded by `CoverageLedger`.
enum class SiteDecisionKind : uint8_t {
  /// The semantic site is outside the selected mode/request contract.
  NotApplicable,
  /// The semantic site is relevant but cannot currently be instrumented.
  Unsupported,
  /// Policy has requested one or more target-neutral probe intents.
  Admitted,
  /// One-past-the-end iteration bound; never a policy answer.
  Count,
};

/// Machine-readable reason for an access-policy decision.
///
/// These values describe only semantic applicability and support. Register
/// pressure, branch reachability, cave allocation, and emitted patch shape are
/// later lowering facts and deliberately cannot be represented here. `None`
/// is valid only for `Admitted`; every other decision must retain the reason
/// that excluded it.
enum class AccessPolicyReason : uint8_t {
  /// The access was admitted and therefore needs no rejection reason.
  None,
  /// The selected request disabled this access family.
  AccessFamilyDisabled,
  /// The decoded operation kind is not an access observed by this mode.
  OperationKindExcluded,
  /// A later synchronization policy owns this ordinary-memory operation.
  ReservedForSynchronizationPolicy,
  /// A diagnostic-only container filter excluded every alias of the site.
  ContainerFilterExcluded,
  /// The selected target/mode capability contract excludes this form.
  TargetCapabilityUnavailable,
  /// Inventory classified the instruction outside ConSan's access kinds.
  NonAccessInstruction,
  /// The decoded instruction has no usable nonzero dword-aligned size.
  InvalidInstructionSize,
  /// The decoded operation has no usable nonzero byte-aligned access width.
  InvalidAccessWidth,
  /// The operation requires an address operand that inventory could not find.
  MissingAddressOperand,
  /// Inventory could not recover every logical range encoded by the operation.
  RangeEncodingUnavailable,
  /// The decoded instruction range lies outside the pristine code object.
  InstructionOutOfBounds,
  /// The semantic operation is relevant but its exact mnemonic is unsupported.
  UnsupportedMnemonic,
  /// A group-FLAT operation uses an instruction layout this mode cannot lower.
  UnsupportedFlatEncoding,
  /// The selected target cannot preserve this encoded FLAT immediate offset.
  NonzeroFlatOffset,
  /// The FLAT address uses a reserved vector-register encoding.
  ReservedFlatAddressRegister,
  /// Inventory proved that the FLAT operation does not access group memory.
  NonGroupAddressSpace,
  /// Strict provenance policy excluded a conservatively inferred group access.
  FlatProvenancePolicyExcluded,
  /// Aliased containers decoded incompatible facts for one physical instruction.
  ConflictingPhysicalAliases,
  /// One-past-the-end iteration bound; never a policy reason.
  Count,
};

inline constexpr auto kAccessPolicyReasons = [] {
  using E = AccessPolicyReason;
  return make_enum_vocabulary(
      "invalid-access-policy-reason", enum_entry(E::None, "none"),
      enum_entry(E::AccessFamilyDisabled, "access-family-disabled"),
      enum_entry(E::OperationKindExcluded, "operation-kind-excluded"),
      enum_entry(E::ReservedForSynchronizationPolicy, "reserved-for-synchronization-policy"),
      enum_entry(E::ContainerFilterExcluded, "container-filter-excluded"),
      enum_entry(E::TargetCapabilityUnavailable, "target-capability-unavailable"),
      enum_entry(E::NonAccessInstruction, "non-access-instruction"),
      enum_entry(E::InvalidInstructionSize, "invalid-instruction-size"),
      enum_entry(E::InvalidAccessWidth, "invalid-access-width"),
      enum_entry(E::MissingAddressOperand, "missing-address-operand"),
      enum_entry(E::RangeEncodingUnavailable, "range-encoding-unavailable"),
      enum_entry(E::InstructionOutOfBounds, "instruction-out-of-bounds"),
      enum_entry(E::UnsupportedMnemonic, "unsupported-mnemonic"),
      enum_entry(E::UnsupportedFlatEncoding, "unsupported-flat-encoding"),
      enum_entry(E::NonzeroFlatOffset, "nonzero-flat-offset"),
      enum_entry(E::ReservedFlatAddressRegister, "reserved-flat-address-register"),
      enum_entry(E::NonGroupAddressSpace, "non-group-address-space"),
      enum_entry(E::FlatProvenancePolicyExcluded, "flat-provenance-policy-excluded"),
      enum_entry(E::ConflictingPhysicalAliases, "conflicting-physical-aliases"));
}();

[[nodiscard]] constexpr std::string_view access_policy_reason_name(AccessPolicyReason reason) {
  return kAccessPolicyReasons.name(reason);
}

/// Machine-readable reason for a barrier-policy decision.
///
/// Barrier policy owns only whether a decoded barrier event belongs to the
/// selected mode contract and whether inventory has enough semantic facts to
/// request an observation. Register ownership, report capacity, trampoline
/// reachability, and emitted patch shape are intentionally absent: those are
/// later mechanism outcomes recorded by `CoverageLedger`.
enum class BarrierPolicyReason : uint8_t {
  /// The event was admitted and therefore needs no exclusion reason.
  None,
  /// Barrier tracking was disabled for this request.
  TrackingDisabled,
  /// The selected mode mutates barriers but does not observe their epochs.
  ModeMutationOnly,
  /// A diagnostic-only container filter excluded every alias of the event.
  ContainerFilterExcluded,
  /// The event belongs only to a ROCclr runtime kernel excluded by policy.
  RuntimeKernelExcluded,
  /// The selected target/mode contract does not support this barrier scope.
  TargetCapabilityUnavailable,
  /// The barrier instruction is not the encoding width required by its mode.
  InvalidBarrierEncoding,
  /// Shared synchronization analysis did not produce a qualified sequence.
  UnqualifiedSyncSequence,
  /// One event was claimed by more than one incompatible logical sequence.
  AmbiguousSequenceMembership,
  /// A qualified sequence had no unique physical completion instruction.
  MissingCompletingEvent,
  /// This full barrier immediately follows an equivalent full barrier.
  RedundantAdjacentFullBarrier,
  /// Symbol aliases decoded incompatible semantics for one physical event.
  ConflictingPhysicalAliases,
  /// One-past-the-end iteration bound; never a policy reason.
  Count,
};

inline constexpr auto kBarrierPolicyReasons = [] {
  using E = BarrierPolicyReason;
  return make_enum_vocabulary(
      "invalid-barrier-policy-reason", enum_entry(E::None, "none"),
      enum_entry(E::TrackingDisabled, "tracking-disabled"),
      enum_entry(E::ModeMutationOnly, "mode-mutation-only"),
      enum_entry(E::ContainerFilterExcluded, "container-filter-excluded"),
      enum_entry(E::RuntimeKernelExcluded, "runtime-kernel-excluded"),
      enum_entry(E::TargetCapabilityUnavailable, "target-capability-unavailable"),
      enum_entry(E::InvalidBarrierEncoding, "invalid-barrier-encoding"),
      enum_entry(E::UnqualifiedSyncSequence, "unqualified-sync-sequence"),
      enum_entry(E::AmbiguousSequenceMembership, "ambiguous-sequence-membership"),
      enum_entry(E::MissingCompletingEvent, "missing-completing-event"),
      enum_entry(E::RedundantAdjacentFullBarrier, "redundant-adjacent-full-barrier"),
      enum_entry(E::ConflictingPhysicalAliases, "conflicting-physical-aliases"));
}();

[[nodiscard]] constexpr std::string_view barrier_policy_reason_name(BarrierPolicyReason reason) {
  return kBarrierPolicyReasons.name(reason);
}

/// Machine-readable reason for an atomic-policy decision.
///
/// Atomic policy owns semantic sequence qualification, target/mode
/// capability, the availability of address and dynamic-result facts, and
/// whether an ordinary-memory sequence is represented by an associated fence.
/// Register allocation, report capacity, instruction placement, and emitted
/// target words remain later lowering responsibilities. `None` is valid only
/// for an admitted atomic observation.
enum class AtomicPolicyReason : uint8_t {
  /// The atomic or ordered ordinary-memory operation was admitted.
  None,
  /// Atomic synchronization tracking was disabled by the request.
  TrackingDisabled,
  /// The selected mode can mutate this operation but does not observe it.
  ModeMutationOnly,
  /// A diagnostic-only container filter excluded the operation.
  ContainerFilterExcluded,
  /// No inventoried kernel can execute the operation.
  MissingExecutionOwner,
  /// The mode has no admitted directional access window that can consume ordering.
  MissingDirectionalAccessWindow,
  /// The target/mode capability contract excludes this atomic form.
  TargetCapabilityUnavailable,
  /// No unique qualified synchronization sequence owns this event.
  UnqualifiedSyncSequence,
  /// More than one incompatible sequence claims the same event.
  AmbiguousSequenceMembership,
  /// The qualified sequence has no supported acquire/release role.
  UnsupportedMemoryRole,
  /// The qualified event has no decoded semantic scope.
  MissingScope,
  /// The decoded scope cannot communicate between modeled owners.
  UnsupportedScope,
  /// The communication width is zero, non-byte-aligned, or mode-unsupported.
  InvalidAccessWidth,
  /// The sequence uses an address source outside the supported contract.
  UnsupportedAddressSource,
  /// Required address or value operands are absent from decoded inventory.
  MissingOperands,
  /// Compare-and-swap success cannot be determined after the guest operation.
  CompareExchangeOutcomeUnavailable,
  /// The RMW result form is unknown or unsupported by the evidence model.
  UnsupportedDynamicOutcome,
  /// The target-native instruction form cannot implement the admitted model.
  UnsupportedEncoding,
  /// Physical aliases expose incompatible semantics for one instruction.
  ConflictingPhysicalAliases,
  /// One-past-the-end iteration bound; never a policy reason.
  Count,
};

/// Complete iterable set of atomic-policy reasons, including `None`.
inline constexpr auto kAtomicPolicyReasons = [] {
  using E = AtomicPolicyReason;
  return make_enum_vocabulary(
      "invalid-atomic-policy-reason", enum_entry(E::None, "none"),
      enum_entry(E::TrackingDisabled, "tracking-disabled"),
      enum_entry(E::ModeMutationOnly, "mode-mutation-only"),
      enum_entry(E::ContainerFilterExcluded, "container-filter-excluded"),
      enum_entry(E::MissingExecutionOwner, "missing-execution-owner"),
      enum_entry(E::MissingDirectionalAccessWindow, "missing-directional-access-window"),
      enum_entry(E::TargetCapabilityUnavailable, "target-capability-unavailable"),
      enum_entry(E::UnqualifiedSyncSequence, "unqualified-sync-sequence"),
      enum_entry(E::AmbiguousSequenceMembership, "ambiguous-sequence-membership"),
      enum_entry(E::UnsupportedMemoryRole, "unsupported-memory-role"),
      enum_entry(E::MissingScope, "missing-scope"),
      enum_entry(E::UnsupportedScope, "unsupported-scope"),
      enum_entry(E::InvalidAccessWidth, "invalid-access-width"),
      enum_entry(E::UnsupportedAddressSource, "unsupported-address-source"),
      enum_entry(E::MissingOperands, "missing-operands"),
      enum_entry(E::CompareExchangeOutcomeUnavailable, "compare-exchange-outcome-unavailable"),
      enum_entry(E::UnsupportedDynamicOutcome, "unsupported-dynamic-outcome"),
      enum_entry(E::UnsupportedEncoding, "unsupported-encoding"),
      enum_entry(E::ConflictingPhysicalAliases, "conflicting-physical-aliases"));
}();

/// Return the stable diagnostic spelling of an atomic-policy reason.
[[nodiscard]] constexpr std::string_view atomic_policy_reason_name(AtomicPolicyReason reason) {
  return kAtomicPolicyReasons.name(reason);
}

static_assert(kAtomicPolicyReasons.size() == static_cast<size_t>(AtomicPolicyReason::Count));

/// Machine-readable reason for a fence-policy decision.
///
/// Shared synchronization analysis owns the detailed association outcome;
/// this enum records how that fact participates in the selected mode's
/// observation contract. A qualified fence contributes ordering metadata to
/// its associated atomic observation.
enum class FencePolicyReason : uint8_t {
  /// The fence was admitted or associated and needs no exclusion reason.
  None,
  /// Atomic/fence synchronization tracking was disabled by the request.
  TrackingDisabled,
  /// The selected mode can mutate fences but does not observe them.
  ModeMutationOnly,
  /// A diagnostic-only container filter excluded the fence.
  ContainerFilterExcluded,
  /// No inventoried kernel can execute the fence or communication event.
  MissingExecutionOwner,
  /// Shared analysis rejected the addressed communication association.
  AssociationUnavailable,
  /// ConSan excluded the associated communication as causally irrelevant.
  CommunicationNotApplicable,
  /// The associated communication identity no longer names a unique event.
  MissingCommunicationEvent,
  /// The target/mode capability contract excludes the addressed fence.
  TargetCapabilityUnavailable,
  /// Physical aliases expose incompatible semantics for one fence.
  ConflictingPhysicalAliases,
  /// One-past-the-end iteration bound; never a policy reason.
  Count,
};

inline constexpr auto kFencePolicyReasons = [] {
  using E = FencePolicyReason;
  return make_enum_vocabulary(
      "invalid-fence-policy-reason", enum_entry(E::None, "none"),
      enum_entry(E::TrackingDisabled, "tracking-disabled"),
      enum_entry(E::ModeMutationOnly, "mode-mutation-only"),
      enum_entry(E::ContainerFilterExcluded, "container-filter-excluded"),
      enum_entry(E::MissingExecutionOwner, "missing-execution-owner"),
      enum_entry(E::AssociationUnavailable, "association-unavailable"),
      enum_entry(E::CommunicationNotApplicable, "communication-not-applicable"),
      enum_entry(E::MissingCommunicationEvent, "missing-communication-event"),
      enum_entry(E::TargetCapabilityUnavailable, "target-capability-unavailable"),
      enum_entry(E::ConflictingPhysicalAliases, "conflicting-physical-aliases"));
}();

[[nodiscard]] constexpr std::string_view fence_policy_reason_name(FencePolicyReason reason) {
  return kFencePolicyReasons.name(reason);
}

/// Stable identity of one synchronization association inside a code object.
///
/// The value is copied from immutable synchronization inventory and names the
/// complete ordered sequence, not merely its anchor instruction. It lets a
/// before-instruction address-capture intent and an after-sequence evidence
/// intent state that they implement one logical operation. Empty values are
/// invalid and are used only as the default construction sentinel.
struct SynchronizationAssociationId {
  /// Stable inventory identity of the complete synchronization sequence.
  std::string value;

  /// Return whether this value can identify a published sequence.
  [[nodiscard]] bool valid() const { return !value.empty(); }

  auto operator<=>(const SynchronizationAssociationId &) const = default;
};

/// Dynamic guest result that an atomic observation must preserve or derive.
///
/// Address capture is represented by a separate before-instruction intent.
/// This value describes only information available after the guest operation:
/// no-return operations need none, return-form RMWs preserve the old value,
/// and compare-and-swap must derive whether the exchange succeeded. A lowerer
/// may not silently weaken a stronger requirement.
enum class DynamicResultRequirement : uint8_t {
  /// No post-guest dynamic result is needed.
  None,
  /// Preserve the atomic value returned by the guest instruction.
  ReturnedOldValue,
  /// Derive compare-and-swap success versus failure from guest operands.
  CompareExchangeSuccess,
  /// One-past-the-end iteration bound; never an intent requirement.
  Count,
};

/// Target-neutral kind of evidence requested from one physical probe.
///
/// The kinds deliberately name the semantic observation rather than an ISA
/// sequence, report offset, or patch kind. This lets all targets share policy
/// while their lowerers remain free to implement the same intent differently.
enum class ProbeIntentKind : uint8_t {
  /// SuperCollider observes whether a redundant access agrees with the guest.
  RedundantAccessObservation,
  /// ConSan conditionally records the access selected by its sampling policy.
  Access,
  /// ConSan closes one selected causal window at a logical barrier sequence.
  BarrierEpoch,
  /// Preserve an addressed synchronization operation's effective address
  /// before the guest instruction can overwrite any input register.
  AtomicAddressCapture,
  /// ConSan publishes atomic ordering into the selected causal window.
  AtomicOrdering,
  /// Preserve the exact address of a modification independently of ordering.
  PublicationAddressCapture,
  /// Record a modification that can invalidate an atomic publication chain.
  PublicationModification,
  /// One-past-the-end iteration bound; never an observation intent.
  Count,
};

/// One mode's target-neutral probe and report-facing evidence vocabulary.
///
/// Semantic policy publishes these probe kinds; report planning consumes the
/// same row rather than reconstructing a parallel mode matrix. `Count` marks
/// an operation that the mode does not support.
struct ModeProbeVocabulary {
  Mode mode;
  ProbeIntentKind access;
  ProbeIntentKind barrier;
  ProbeIntentKind atomic;
  bool sticky_access_evidence;
  bool address_capture;
  /// Ordered communication is meaningful only when one admitted access probe
  /// owned by the same kernel can consume the corresponding direction.
  bool ordering_requires_directional_access_window;
  bool barrier_elements_per_semantic_site;
};

inline constexpr std::array<ModeProbeVocabulary, kEnabledModes.size()> kModeProbeVocabularies = {{
    {Mode::SuperCollider, ProbeIntentKind::RedundantAccessObservation, ProbeIntentKind::Count,
     ProbeIntentKind::Count, true, false, false, false},
    {Mode::Default, ProbeIntentKind::Access, ProbeIntentKind::BarrierEpoch,
     ProbeIntentKind::AtomicOrdering, false, true, true, true},
}};

[[nodiscard]] constexpr const ModeProbeVocabulary *mode_probe_vocabulary(Mode mode) {
  for (const auto &vocabulary : kModeProbeVocabularies) {
    if (vocabulary.mode == mode)
      return &vocabulary;
  }
  return nullptr;
}

/// Required ordering of a probe relative to the pristine guest instruction.
///
/// This is a semantic placement constraint. It does not choose inline versus
/// trampoline lowering or prescribe how a target preserves clobbered operands.
enum class ProbePosition : uint8_t {
  /// Complete the observation before executing the guest instruction.
  Before,
  /// Complete the observation after executing the guest instruction.
  After,
  /// One-past-the-end iteration bound; never a placement constraint.
  Count,
};

/// One mode-policy decision for one normalized semantic access range.
///
/// Container aliases are attribution only and are retained as names for
/// diagnostics. The semantic identity is authoritative. Probe intents are the
/// independent, authoritative lowering contract.
struct SiteDecision {
  /// Stable identity of the logical access range being decided.
  SemanticSiteId semantic_site;
  /// Applicability/admission result that lowering cannot change.
  SiteDecisionKind kind = SiteDecisionKind::NotApplicable;
  /// Typed explanation; `None` is reserved for admitted decisions.
  AccessPolicyReason reason = AccessPolicyReason::AccessFamilyDisabled;
  bool operator==(const SiteDecision &) const = default;
};

/// One mode-policy decision for one decoded barrier event.
///
/// The decision is about original-program meaning, not a patch. A paired
/// signal and wait may therefore share one completion intent. `semantic_site`
/// remains the authoritative diagnostic identity; source aliases are derived
/// from the program inventory when diagnostics need them.
struct BarrierSiteDecision {
  /// Stable identity of the original synchronization event being decided.
  SemanticSiteId semantic_site;
  /// Applicability/admission result that lowering cannot change.
  SiteDecisionKind kind = SiteDecisionKind::NotApplicable;
  /// Typed explanation; `None` is reserved for admitted decisions.
  BarrierPolicyReason reason = BarrierPolicyReason::TrackingDisabled;
  bool operator==(const BarrierSiteDecision &) const = default;
};

/// One mode-policy decision for an ordered atomic communication event.
///
/// The event may be a target-native RMW or the address-bearing load/store in a
/// qualified language-level atomic sequence. `capability` preserves the public
/// target/mode contract, while `reason` records the more specific decoded
/// decision. The corresponding before-address and after-operation contracts
/// live only in the plan's probe intents.
struct AtomicSiteDecision {
  /// Stable identity of the address-bearing synchronization event.
  SemanticSiteId semantic_site;
  /// Applicability/admission result that lowering cannot change.
  SiteDecisionKind kind = SiteDecisionKind::NotApplicable;
  /// Public target/mode capability disposition for this operation form.
  CapabilityDisposition capability = CapabilityDisposition::OutOfContract;
  /// Typed explanation; `None` is reserved for admitted decisions.
  AtomicPolicyReason reason = AtomicPolicyReason::TrackingDisabled;
  bool operator==(const AtomicSiteDecision &) const = default;
};

/// One mode-policy decision for a decoded cache/fence event.
///
/// `inventory_association` is the mode-independent analysis result and is
/// retained even when policy excludes the event. `AssociatedOnly` makes modes
/// without an independent fence probe explicit without copying intent links
/// into diagnostic decisions.
struct FenceSiteDecision {
  /// Stable identity of the original cache/fence instruction.
  SemanticSiteId semantic_site;
  /// Applicability/admission result that lowering cannot change.
  SiteDecisionKind kind = SiteDecisionKind::NotApplicable;
  /// Public target/mode capability disposition for addressed fences.
  CapabilityDisposition capability = CapabilityDisposition::OutOfContract;
  /// Typed policy explanation; `None` is reserved for admitted decisions.
  FencePolicyReason reason = FencePolicyReason::TrackingDisabled;
  /// Detailed shared-analysis association outcome, never inferred from prose.
  FenceAssociation inventory_association = FenceAssociation::MissingOrAmbiguousSequence;
  bool operator==(const FenceSiteDecision &) const = default;
};

/// One target-neutral observation requested at a physical instruction.
///
/// `covered_semantic_sites` can contain multiple access ranges from one
/// instruction or every event in one logical barrier sequence. That coalescing
/// is explicit and keeps placement capacity at one insertion point. The intent
/// contains no chosen registers, report addresses, machine words, branch route,
/// or patch kind. Every intent observes every lane active at the guest
/// instruction; no mode may silently narrow that execution mask. Every
/// admitted intent is part of the requested mode contract, so losing any
/// intent during lowering is a coverage gap.
struct ProbeIntent {
  /// Stable plan-local identifier cited by semantic decisions.
  ProbeIntentId id;
  /// Mode whose evidence semantics this intent implements.
  Mode mode = Mode::None;
  /// Exact decoded site at which lowering must place this probe.
  ProgramSiteId source_site;
  /// Physical instruction at which lowering must place the coalesced probe.
  PhysicalSiteId physical_site;
  /// All logical access ranges or synchronization events observed by the probe.
  std::vector<SemanticSiteId> covered_semantic_sites;
  /// Semantic evidence operation requested from target lowering.
  ProbeIntentKind kind = ProbeIntentKind::Count;
  /// Required ordering relative to the guest instruction.
  ProbePosition position = ProbePosition::Count;
  /// Complete synchronization sequence implemented by this probe, if any.
  /// Access and barrier intents leave this absent; atomic/fence intents use it
  /// to join before- and after-operation work without relying on patch order.
  std::optional<SynchronizationAssociationId> synchronization_association;
  /// Dynamic post-guest value the lowerer must preserve or derive.
  DynamicResultRequirement dynamic_result = DynamicResultRequirement::None;
  /// Exact target-normalized address-bearing form selected by policy. This is
  /// present only on `AtomicAddressCapture`: every after-operation intent for
  /// the same association consumes that one captured address contract instead
  /// of retaining a second copy on atomic and fence decisions.
  std::optional<AtomicLoweringForm> atomic_lowering_form;

  bool operator==(const ProbeIntent &) const = default;
};

/// Complete target-normalized semantic-policy output for one mode request.
///
/// Separate decision sequences cover semantic access ranges, barriers, ordered
/// communication events, and fences; their shared intents cover physical
/// insertion points and complete synchronization associations. The plan is a
/// value object and contains no borrowed inventory references, so a result may
/// safely retain it across retries and image moves.
struct ObservationPlan {
  /// Selected mode shared by every decision and intent in this plan.
  Mode mode = Mode::None;
  /// Deterministically ordered semantic access decisions.
  std::vector<SiteDecision> site_decisions;
  /// Deterministically ordered semantic barrier-event decisions.
  std::vector<BarrierSiteDecision> barrier_site_decisions;
  /// Deterministically ordered ordered-communication decisions.
  std::vector<AtomicSiteDecision> atomic_site_decisions;
  /// Deterministically ordered cache/fence decisions.
  std::vector<FenceSiteDecision> fence_site_decisions;
  /// Deterministically ordered coalesced physical probe intents.
  std::vector<ProbeIntent> probe_intents;

  /// Verify plan-local identities and the decision/intent relationship.
  [[nodiscard]] bool valid() const;

  /// Resolve one plan-local intent identifier, rejecting invalid or stale IDs.
  [[nodiscard]] const ProbeIntent *intent(ProbeIntentId id) const;

  /// Transactionally append a valid same-mode policy fragment.
  ///
  /// Plan-local intent identifiers in the fragment are rebased and every
  /// decision reference is rewritten. On invalid input or a mode mismatch,
  /// this plan is unchanged and the function returns false.
  [[nodiscard]] bool append(const ObservationPlan &fragment);

  bool operator==(const ObservationPlan &) const = default;
};

/// Mutable request parameters consumed by pure access policy.
///
/// The borrowed spans and filter are read only during the policy call; no
/// output retains them. Production requests normally enable both access
/// families for ConSan, while SuperCollider reflects its two explicit probe
/// switches. `reserved_for_synchronization` prevents an ordinary memory event
/// from being double-counted before Slice 4C takes ownership of its ordering
/// semantics.
struct AccessPolicyRequest {
  /// Mode whose access contract should be evaluated.
  Mode mode = Mode::None;
  /// Whether native LDS and direct-to-LDS sites belong to this request.
  bool native_lds_enabled = false;
  /// Whether group-FLAT sites belong to this request.
  bool group_flat_enabled = false;
  /// Confidence threshold for conservatively inferred group-FLAT provenance.
  FlatProvenanceMode flat_provenance_mode = FlatProvenanceMode::Strict;
  /// Optional diagnostic-only container substring filter.
  std::string_view container_filter;
  /// Production exact kernel-entry allowlist. A shared physical site remains
  /// excluded unless every known execution owner is selected.
  std::span<const std::string> kernel_name_allowlist;
  /// Physical ordinary-memory sites delegated to synchronization policy.
  std::span<const PhysicalSiteId> reserved_for_synchronization;
};

/// Pure access-policy output plus typed construction failures.
///
/// A conflicting physical alias is represented both as unsupported decisions
/// and in `errors`, allowing unit tests and adapters to reject the transform
/// without parsing prose. No warning or error is emitted as a side effect.
struct AccessPolicyResult {
  /// Complete plan, including explicit non-admission decisions.
  ObservationPlan plan;
  /// Typed fatal policy-construction failures.
  std::vector<AccessPolicyReason> errors;

  /// Return whether policy produced a usable, internally valid plan.
  [[nodiscard]] bool valid() const { return errors.empty() && plan.valid(); }

  bool operator==(const AccessPolicyResult &) const = default;
};

/// Immutable request parameters consumed by pure barrier policy.
///
/// The request selects semantic applicability only. It contains no report
/// address, register assignment, code-cave budget, or mutable result object, so
/// the same inventory/request pair always yields the same plan fragment.
struct BarrierPolicyRequest {
  /// Mode whose barrier-observation contract should be evaluated.
  Mode mode = Mode::None;
  /// Whether decoded barrier events belong to this request.
  bool tracking_enabled = false;
  /// Optional diagnostic-only container substring filter.
  std::string_view container_filter;
  /// Production exact kernel-entry allowlist.
  std::span<const std::string> kernel_name_allowlist;
};

/// Pure barrier-policy output plus typed construction failures.
///
/// Ordinary exclusions remain explicit decisions and do not enter `errors`.
/// Only contradictions such as incompatible physical aliases make the result
/// unusable, allowing callers to fail closed without parsing diagnostics.
struct BarrierPolicyResult {
  /// Barrier-only observation-plan fragment, ready for transactional append.
  ObservationPlan plan;
  /// Typed fatal policy-construction failures.
  std::vector<BarrierPolicyReason> errors;

  /// Return whether policy produced a usable, internally valid fragment.
  [[nodiscard]] bool valid() const { return errors.empty() && plan.valid(); }

  bool operator==(const BarrierPolicyResult &) const = default;
};

/// Immutable request parameters consumed by pure atomic/fence policy.
///
/// `tracking_enabled` selects the atomic-ordering feature as a whole.
/// `directional_access_windows` is a generic coupling between ordered
/// communication and admitted access evidence. A mode vocabulary may
/// require an owner-local read window for acquire or write window for release.
/// The request contains no report address, register assignment, capacity, or
/// mutable result state.
struct DirectionalAccessAvailability {
  /// Kernel that can execute the admitted access window.
  ProgramContainerId owner;
  /// At least one selected read window exists for this owner.
  bool read = false;
  /// At least one selected write window exists for this owner.
  bool write = false;

  bool operator==(const DirectionalAccessAvailability &) const = default;
};

struct AtomicFencePolicyRequest {
  /// Include address-only publication modifications, independently of sync roles.
  bool publication_modifications_enabled = false;
  /// Mode whose ordered-communication contract should be evaluated.
  Mode mode = Mode::None;
  /// Whether atomic and associated-fence tracking belongs to this request.
  bool tracking_enabled = false;
  /// Directional, owner-local access windows able to consume ordering metadata.
  std::span<const DirectionalAccessAvailability> directional_access_windows;
  /// Optional diagnostic-only container substring filter.
  std::string_view container_filter;
  /// Production exact kernel-entry allowlist.
  std::span<const std::string> kernel_name_allowlist;
};

/// Pure atomic/fence-policy output plus typed construction failures.
///
/// The plan fragment owns all decisions, association identities, and
/// before/after intents. Ordinary exclusions remain usable output. Only
/// contradictions such as incompatible physical aliases enter the error
/// vectors and make the fragment invalid.
struct AtomicFencePolicyResult {
  /// Atomic/fence-only observation-plan fragment, ready for append.
  ObservationPlan plan;
  /// Typed fatal atomic-policy construction failures.
  std::vector<AtomicPolicyReason> atomic_errors;
  /// Typed fatal fence-policy construction failures.
  std::vector<FencePolicyReason> fence_errors;

  /// Return whether policy produced a usable, internally valid fragment.
  [[nodiscard]] bool valid() const {
    return atomic_errors.empty() && fence_errors.empty() && plan.valid();
  }

  bool operator==(const AtomicFencePolicyResult &) const = default;
};

/// Return whether shared synchronization analysis proved every fact required
/// by the current ConSan barrier-epoch contract. The predicate is declared at
/// the policy boundary because both pure admission and the temporary legacy
/// lowerer must agree during incremental migration.
[[nodiscard]] bool qualifies_barrier_sequence(const SyncSequence &sequence, bool owner_proven);

/// Final mechanism state of one admitted probe intent.
///
/// The outcome is independent of semantic admission: a target lowerer can
/// reject resources or placement, but cannot convert an unsupported decision
/// into an instrumented one.
enum class LoweringOutcomeKind : uint8_t {
  /// No lowerer has yet published a final result for the admitted intent.
  Pending,
  /// Every required machine-code operation for the intent was emitted.
  Instrumented,
  /// No correctness-preserving register or runtime-state assignment existed.
  ResourceRejected,
  /// Resources existed but transactional machine-code placement failed.
  PlacementRejected,
  /// One-past-the-end iteration bound; never a lowering result.
  Count,
};

/// Original-program attribution shared by every runtime-facing access map.
///
/// The intent IDs make this a forward product of semantic planning rather
/// than a later offset join. `original_site` and `original_semantic_sites`
/// retain pristine identities, while execution-owner handles state the
/// complete static kernel scope only when `owner_provenance_complete` is true.
struct StaticAccessAttribution {
  std::vector<ProbeIntentId> intent_ids;
  PhysicalSiteId original_site;
  std::vector<SemanticSiteId> original_semantic_sites;
  std::vector<ProgramContainerId> execution_owner_kernel_ids;
  bool owner_provenance_complete = false;
  /// Proven identical-value, identical-address lanes of one store instruction.
  bool uniform_lds_store = false;

  bool operator==(const StaticAccessAttribution &) const = default;
};

/// Static attribution of one ConSan access to its report slots and emitted
/// guest/probe locations.
///
/// These are named runtime facts, not a projection of `PatchInfo`.
/// Report decoding uses the slot window; stable diagnostics use the emitted
/// probe, relocated guest, and scratch-register values.
struct StaticAccessMapping {
  StaticAccessAttribution access;
  LdsAccessKind access_kind = LdsAccessKind::Other;
  uint32_t first_slot = 0;
  uint32_t range_count = 0;
  uint32_t bank_count = 0;
  uint64_t emitted_probe_text_offset = 0;
  std::optional<uint64_t> relocated_guest_text_offset;
  std::optional<uint16_t> scratch_vgpr;

  bool operator==(const StaticAccessMapping &) const = default;
};

/// Static ConSan access mappings produced by intent-bound lowering.
using StaticAccessMappings = std::vector<StaticAccessMapping>;

/// One machine-code location emitted as part of an intent-bound lowering.
///
/// The location is semantic telemetry rather than proof-by-patch-kind. It
/// names the original physical site whose lowering owns the emitted bytes and
/// records enough geometry for later static mapping without exposing the
/// lowerer's patch representation.
struct CommittedLoweringLocation {
  /// Original physical instruction implemented by this emitted operation.
  PhysicalSiteId original_site;
  /// Text-relative location of the committed machine-code operation.
  uint64_t emitted_text_offset = 0;
  /// Number of committed machine-code bytes at `emitted_text_offset`.
  uint32_t emitted_size = 0;
  /// Relocated guest instruction, when this operation displaced the guest.
  std::optional<uint64_t> relocated_guest_text_offset;

  bool operator==(const CommittedLoweringLocation &) const = default;
};

/// Authoritative result of one intent-bound lowering transaction.
///
/// A single emitted operation can satisfy several coalesced intents, while a
/// synchronization sequence can require several emitted locations for one
/// intent. Original sites remain authoritative on the ledger-owned observation
/// plan and are resolved through `intent_ids`; the transaction does not copy a
/// second identity projection from that plan.
struct CommittedLowering {
  /// Every admitted intent completed or rejected by this transaction.
  std::vector<ProbeIntentId> intent_ids;
  /// One or more committed locations, empty for a rejected transaction.
  std::vector<CommittedLoweringLocation> locations;
  /// Runtime-facing semantic/static mapping created by this same transaction.
  StaticAccessMappings runtime_mapping;
  /// Explicit final mechanism outcome shared by every bound intent.
  LoweringOutcomeKind outcome = LoweringOutcomeKind::Pending;
  /// Typed resource-planner rejection, when that component rejected the
  /// transaction. Other rejection domains leave this absent.
  std::optional<RegisterPlanReason> resource_rejection_reason;
  /// Optional mechanism detail with no semantic-policy meaning.
  std::string detail;

  bool operator==(const CommittedLowering &) const = default;
};

/// Prepare one authoritative lowering result from immutable intent identity.
/// Returns no value for stale/duplicate intent IDs, inconsistent locations,
/// pending outcomes, or emitted locations attached to a rejection.
[[nodiscard]] std::optional<CommittedLowering>
make_committed_lowering(const ObservationPlan &plan, std::span<const ProbeIntentId> intent_ids,
                        std::span<const CommittedLoweringLocation> locations,
                        LoweringOutcomeKind outcome, std::string detail = {},
                        StaticAccessMappings runtime_mapping = {},
                        std::optional<RegisterPlanReason> resource_rejection_reason = std::nullopt);

/// Lowering state for one plan-owned admitted intent.
///
/// The immutable intent itself remains in the ledger-owned observation plan;
/// this entry stores only its stable plan-local identity and mutable lowering
/// state. `detail` is diagnostic context only; control flow and trust
/// evaluation use the typed outcome.
struct IntentCoverageEntry {
  /// Plan-local intent whose mechanism status is being tracked.
  ProbeIntentId intent_id;
  /// Current finalization state for that intent.
  LoweringOutcomeKind lowering = LoweringOutcomeKind::Pending;
  /// Resource-planner reason published by the same lowering transaction.
  std::optional<RegisterPlanReason> resource_rejection_reason;
  /// Optional human-readable mechanism detail with no policy meaning.
  std::string detail;

  bool operator==(const IntentCoverageEntry &) const = default;
};

/// Joined semantic coverage state without conflating policy and lowering.
///
/// The ledger is the sole owner of the immutable observation plan and the
/// lowering state joined to its admitted intents. Callers may update only a
/// known admitted intent, and queries never infer success from a patch kind or
/// warning string.
class CoverageLedger {
public:
  CoverageLedger() = default;
  explicit CoverageLedger(ObservationPlan plan);

  /// Return the one immutable semantic plan owned by this ledger.
  [[nodiscard]] const ObservationPlan &observation_plan() const { return observation_plan_; }

  /// Resolve one admitted intent from the immutable owned plan.
  [[nodiscard]] const ProbeIntent *intent(ProbeIntentId id) const {
    return observation_plan_.intent(id);
  }

  /// Return immutable semantic decisions from the owned observation plan.
  [[nodiscard]] std::span<const SiteDecision> site_decisions() const {
    return observation_plan_.site_decisions;
  }

  /// Return immutable barrier-event decisions from the owned plan.
  [[nodiscard]] std::span<const BarrierSiteDecision> barrier_site_decisions() const {
    return observation_plan_.barrier_site_decisions;
  }

  /// Return immutable ordered-communication decisions from the owned plan.
  [[nodiscard]] std::span<const AtomicSiteDecision> atomic_site_decisions() const {
    return observation_plan_.atomic_site_decisions;
  }

  /// Return immutable fence decisions from the owned observation plan.
  [[nodiscard]] std::span<const FenceSiteDecision> fence_site_decisions() const {
    return observation_plan_.fence_site_decisions;
  }

  /// Return immutable joined intent/lowering entries.
  [[nodiscard]] std::span<const IntentCoverageEntry> intent_entries() const {
    return intent_entries_;
  }

  /// Resolve one admitted intent entry by its plan-local identifier.
  [[nodiscard]] const IntentCoverageEntry *intent_entry(ProbeIntentId id) const;

  /// Return intents that explicitly cover one original semantic site.
  [[nodiscard]] std::vector<ProbeIntentId> intent_ids_covering(const SemanticSiteId &site) const {
    std::vector<ProbeIntentId> ids;
    const auto candidates =
        intent_ids_by_semantic_text_offset_.find(site.physical.original_text_offset);
    if (candidates == intent_ids_by_semantic_text_offset_.end())
      return ids;
    for (ProbeIntentId id : candidates->second) {
      const ProbeIntent *candidate = intent(id);
      if (candidate != nullptr && std::ranges::find(candidate->covered_semantic_sites, site) !=
                                      candidate->covered_semantic_sites.end()) {
        ids.push_back(id);
      }
    }
    return ids;
  }

  /// Return pending intents of one semantic kind at one physical site.
  [[nodiscard]] std::vector<ProbeIntentId> pending_intent_ids(ProbeIntentKind kind,
                                                              const PhysicalSiteId &site) const {
    std::vector<ProbeIntentId> ids;
    const size_t kind_index = static_cast<size_t>(kind);
    if (kind_index >= intent_ids_by_kind_and_text_offset_.size())
      return ids;
    const auto &by_offset = intent_ids_by_kind_and_text_offset_[kind_index];
    const auto candidates = by_offset.find(site.original_text_offset);
    if (candidates == by_offset.end())
      return ids;
    for (ProbeIntentId id : candidates->second) {
      const ProbeIntent *candidate = intent(id);
      const IntentCoverageEntry *entry = intent_entry(id);
      if (candidate != nullptr && candidate->physical_site == site && entry != nullptr &&
          entry->lowering == LoweringOutcomeKind::Pending)
        ids.push_back(id);
    }
    return ids;
  }

  /// Publish one prepared lowering transaction directly against its intents.
  /// The operation validates every bound identity before changing any entry
  /// and refuses to overwrite an already finalized intent. Accepted commits
  /// remain owned by this ledger so coverage, proof, and runtime attribution
  /// cannot diverge into independently mutable representations.
  [[nodiscard]] bool publish_lowering_commit(CommittedLowering commit);

  /// Construct and publish one non-instrumented outcome against exact
  /// plan-owned intent identity. Rejection clients supply semantic identity
  /// and outcome only; empty location/runtime proof is owned by the ledger.
  [[nodiscard]] bool publish_lowering_rejection(
      std::span<const ProbeIntentId> intent_ids, LoweringOutcomeKind outcome,
      std::string detail = {},
      std::optional<RegisterPlanReason> resource_rejection_reason = std::nullopt);

  /// Atomically publish a complete group of lowering transactions. No entry
  /// changes when any transaction is stale or malformed.
  [[nodiscard]] bool publish_lowering_commits(std::vector<CommittedLowering> commits);

  /// Atomically publish instrumented synchronization transactions, merging
  /// accepted transactions that share an intent. One semantic synchronization
  /// intent may require several emitted locations, but its complete accepted
  /// transaction remains owned by this ledger rather than a caller-side
  /// staging inventory.
  [[nodiscard]] bool
  publish_coalescing_instrumented_commits(std::vector<CommittedLowering> commits);

  /// Atomically replace already-published geometry for the intents covered by
  /// a completed whole-text transaction. Every intent in an overlapping old
  /// commit must be present in the replacement batch; this prevents a partial
  /// relocation from silently discarding another semantic obligation.
  [[nodiscard]] bool publish_replacing_instrumented_commits(std::vector<CommittedLowering> commits);

  /// Return the authoritative lowering transactions in publication order.
  [[nodiscard]] std::span<const CommittedLowering> lowering_commits() const {
    return lowering_commits_;
  }

  /// Project runtime attribution from the authoritative lowering transactions.
  [[nodiscard]] StaticAccessMappings runtime_static_mapping() const;

  /// Forget emitted transactions while retaining explicit rejections. This is
  /// the one rollback operation used when candidate bytes are discarded.
  void discard_instrumented_lowerings();

  bool operator==(const CoverageLedger &) const = default;

private:
  ObservationPlan observation_plan_;
  std::vector<IntentCoverageEntry> intent_entries_;
  std::vector<CommittedLowering> lowering_commits_;
  std::array<std::unordered_map<uint64_t, std::vector<ProbeIntentId>>,
             static_cast<size_t>(ProbeIntentKind::Count)>
      intent_ids_by_kind_and_text_offset_;
  std::unordered_map<uint64_t, std::vector<ProbeIntentId>> intent_ids_by_semantic_text_offset_;
};

/// Complete mode-policy request consumed by observation assembly.
///
/// This is the only production request that composes access, barrier, and
/// optional atomic/fence policy. Borrowed filters and site reservations are
/// consumed during the call and are never retained by the resulting value.
struct ObservationPolicyRequest {
  /// Mode whose complete observation contract should be assembled.
  Mode mode = Mode::None;
  /// Whether native LDS and direct-to-LDS accesses belong to this request.
  bool native_lds_enabled = false;
  /// Whether group-FLAT accesses belong to this request.
  bool group_flat_enabled = false;
  /// Confidence threshold for conservatively inferred group-FLAT provenance.
  FlatProvenanceMode flat_provenance_mode = FlatProvenanceMode::Strict;
  /// Whether decoded barrier events should be observed.
  bool barrier_tracking_enabled = false;
  /// Whether this mode participates in atomic/fence policy assembly.
  ///
  /// SuperCollider currently has no atomic/fence observation contract and
  /// therefore leaves the fragment absent instead of manufacturing excluded
  /// decisions that were not part of its prior policy product.
  bool include_atomic_fence_policy = false;
  /// Whether atomic and associated-fence tracking is enabled when included.
  bool atomic_fence_tracking_enabled = false;
  /// Optional diagnostic-only container substring filter.
  std::string_view container_filter;
  /// Production exact kernel-entry allowlist.
  std::span<const std::string> kernel_name_allowlist;
  /// Physical ordinary-memory sites delegated to synchronization policy.
  std::span<const PhysicalSiteId> reserved_for_synchronization;
};

/// Decide whether a physical site belongs entirely to selected kernel entries.
/// Descriptor ownership is authoritative when present. The source-container
/// fallback supports legacy kernel-local inventory that predates ownership
/// analysis without permitting substring matches.
[[nodiscard]] bool
site_matches_kernel_allowlist(const ProgramInventory &inventory,
                              std::span<const uint64_t> owner_descriptor_file_offsets,
                              std::span<const std::string> source_containers,
                              std::span<const std::string> kernel_name_allowlist);

[[nodiscard]] bool
site_matches_kernel_allowlist(const ProgramInventory &inventory,
                              std::span<const ExecutionOwner> execution_owners,
                              std::span<const std::string> source_containers,
                              std::span<const std::string> kernel_name_allowlist);

/// Validated composition of every semantic policy fragment for one mode.
///
/// The initial ledger owns the one composed plan so mode lowerers cannot
/// append another policy fragment or initialize parallel policy state.
/// Typed fragment failures remain available for invariant tests; diagnostics
/// are rendered once at this boundary for transform consumers.
struct ObservationProduct {
  CoverageLedger initial_coverage;
  std::vector<AccessPolicyReason> access_errors;
  std::vector<BarrierPolicyReason> barrier_errors;
  std::vector<AtomicPolicyReason> atomic_errors;
  std::vector<FencePolicyReason> fence_errors;
  bool barrier_fragment_appended = false;
  bool atomic_fence_fragment_appended = false;
  bool atomic_fence_fragment_required = false;
  std::vector<std::string> diagnostics;

  [[nodiscard]] const ObservationPlan &plan() const { return initial_coverage.observation_plan(); }

  /// Return whether every requested fragment composed into one valid plan and
  /// the initial ledger is exactly the projection of that plan.
  [[nodiscard]] bool valid() const {
    return access_errors.empty() && barrier_errors.empty() && atomic_errors.empty() &&
           fence_errors.empty() && barrier_fragment_appended &&
           (!atomic_fence_fragment_required || atomic_fence_fragment_appended) && plan().valid() &&
           initial_coverage == CoverageLedger(plan()) && diagnostics.empty();
  }

  bool operator==(const ObservationProduct &) const = default;
};

/// Evaluate target-neutral access applicability and admission without mutating
/// inventory, options, diagnostics, candidates, resource plans, or patches.
[[nodiscard]] AccessPolicyResult plan_access_observation(const ProgramInventory &inventory,
                                                         const AccessPolicyRequest &request);

/// Evaluate target-neutral barrier applicability and sequence admission without
/// mutating inventory, options, diagnostics, candidates, resource plans, or
/// patches.
[[nodiscard]] BarrierPolicyResult plan_barrier_observation(const ProgramInventory &inventory,
                                                           const BarrierPolicyRequest &request);

/// Evaluate target-neutral ordered atomic and associated-fence applicability
/// without mutating inventory, options, diagnostics, candidates, resources, or
/// patches.
[[nodiscard]] AtomicFencePolicyResult
plan_atomic_fence_observation(const ProgramInventory &inventory,
                              const AtomicFencePolicyRequest &request);

/// Assemble the sole mode observation product in dependency order: access,
/// barrier, then optional atomic/fence policy, followed by its initial ledger.
[[nodiscard]] ObservationProduct
assemble_observation_product(const ProgramInventory &inventory,
                             const ObservationPolicyRequest &request);

/// Translate the complete typed caller request into the one observation
/// policy request and assemble its validated product. This is the sole
/// production authority for mode-specific domain selection, diagnostic
/// filtering, and ordinary-memory synchronization reservations.
[[nodiscard]] ObservationProduct assemble_observation_product(const ProgramInventory &inventory,
                                                              const Request &request,
                                                              const DebugOverrides &debug);

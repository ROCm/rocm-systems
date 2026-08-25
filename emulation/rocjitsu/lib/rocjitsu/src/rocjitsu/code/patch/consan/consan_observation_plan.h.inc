// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Stable identifier of one target-neutral instrumentation request in a
/// `ConSanObservationPlan`.
///
/// The identifier is an index into the plan's immutable intent sequence, not a
/// machine-code address or patch ordinal. Several semantic range decisions can
/// intentionally cite the same intent when one physical probe observes them
/// together. The maximum 32-bit value is reserved as the invalid default so a
/// zero-initialized identifier cannot silently select the first intent.
struct ConSanProbeIntentId {
  /// Reserved value used by the default constructor and rejected by plans.
  static constexpr uint32_t invalid_value = std::numeric_limits<uint32_t>::max();

  /// Zero-based position in the owning observation plan, or `invalid_value`.
  uint32_t value = invalid_value;

  /// Return whether this value can name an entry in some observation plan.
  /// The owning plan must still perform the upper-bound check.
  [[nodiscard]] constexpr bool valid() const { return value != invalid_value; }

  auto operator<=>(const ConSanProbeIntentId &) const = default;
};

/// Policy answer for one semantic site before register or placement planning.
///
/// `NotApplicable` is an intentional exclusion from the selected engine's
/// contract. `Unsupported` belongs to that contract but lacks a required
/// decoded fact or qualified implementation. `Admitted` requests one or more
/// probe intents. Lowering is not permitted to rewrite any of these answers;
/// its independent outcome is recorded by `ConSanCoverageLedger`.
enum class ConSanSiteDecisionKind : uint8_t {
  /// The semantic site is outside the selected engine/request contract.
  NotApplicable,
  /// The semantic site is relevant but cannot currently be instrumented.
  Unsupported,
  /// Policy has requested one or more target-neutral probe intents.
  Admitted,
  /// One-past-the-end iteration bound; never a policy answer.
  Count,
};

/// Complete iterable set of policy decision kinds.
inline constexpr std::array<ConSanSiteDecisionKind, 3> kConSanSiteDecisionKinds = {
    ConSanSiteDecisionKind::NotApplicable,
    ConSanSiteDecisionKind::Unsupported,
    ConSanSiteDecisionKind::Admitted,
};

/// Return the stable diagnostic spelling of a policy decision kind.
[[nodiscard]] constexpr std::string_view
consan_site_decision_kind_name(ConSanSiteDecisionKind kind) {
  switch (kind) {
  case ConSanSiteDecisionKind::NotApplicable:
    return "not-applicable";
  case ConSanSiteDecisionKind::Unsupported:
    return "unsupported";
  case ConSanSiteDecisionKind::Admitted:
    return "admitted";
  case ConSanSiteDecisionKind::Count:
    break;
  }
  return "invalid-site-decision-kind";
}

static_assert(kConSanSiteDecisionKinds.size() ==
              static_cast<size_t>(ConSanSiteDecisionKind::Count));

/// Machine-readable reason for an access-policy decision.
///
/// These values describe only semantic applicability and support. Register
/// pressure, branch reachability, cave allocation, and emitted patch shape are
/// later lowering facts and deliberately cannot be represented here. `None`
/// is valid only for `Admitted`; every other decision must retain the reason
/// that excluded it.
enum class ConSanAccessPolicyReason : uint8_t {
  /// The access was admitted and therefore needs no rejection reason.
  None,
  /// The selected request disabled this access family.
  AccessFamilyDisabled,
  /// The decoded operation kind is not an access observed by this engine.
  OperationKindExcluded,
  /// A later synchronization policy owns this ordinary-memory operation.
  ReservedForSynchronizationPolicy,
  /// A diagnostic-only container filter excluded every alias of the site.
  ContainerFilterExcluded,
  /// The selected target/engine capability contract excludes this form.
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
  /// A group-FLAT operation uses an instruction layout this engine cannot lower.
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

/// Complete iterable set of access-policy reasons, including `None`.
inline constexpr std::array<ConSanAccessPolicyReason, 19> kConSanAccessPolicyReasons = {
    ConSanAccessPolicyReason::None,
    ConSanAccessPolicyReason::AccessFamilyDisabled,
    ConSanAccessPolicyReason::OperationKindExcluded,
    ConSanAccessPolicyReason::ReservedForSynchronizationPolicy,
    ConSanAccessPolicyReason::ContainerFilterExcluded,
    ConSanAccessPolicyReason::TargetCapabilityUnavailable,
    ConSanAccessPolicyReason::NonAccessInstruction,
    ConSanAccessPolicyReason::InvalidInstructionSize,
    ConSanAccessPolicyReason::InvalidAccessWidth,
    ConSanAccessPolicyReason::MissingAddressOperand,
    ConSanAccessPolicyReason::RangeEncodingUnavailable,
    ConSanAccessPolicyReason::InstructionOutOfBounds,
    ConSanAccessPolicyReason::UnsupportedMnemonic,
    ConSanAccessPolicyReason::UnsupportedFlatEncoding,
    ConSanAccessPolicyReason::NonzeroFlatOffset,
    ConSanAccessPolicyReason::ReservedFlatAddressRegister,
    ConSanAccessPolicyReason::NonGroupAddressSpace,
    ConSanAccessPolicyReason::FlatProvenancePolicyExcluded,
    ConSanAccessPolicyReason::ConflictingPhysicalAliases,
};

/// Return the stable diagnostic spelling of an access-policy reason.
[[nodiscard]] constexpr std::string_view
consan_access_policy_reason_name(ConSanAccessPolicyReason reason) {
  switch (reason) {
  case ConSanAccessPolicyReason::None:
    return "none";
  case ConSanAccessPolicyReason::AccessFamilyDisabled:
    return "access-family-disabled";
  case ConSanAccessPolicyReason::OperationKindExcluded:
    return "operation-kind-excluded";
  case ConSanAccessPolicyReason::ReservedForSynchronizationPolicy:
    return "reserved-for-synchronization-policy";
  case ConSanAccessPolicyReason::ContainerFilterExcluded:
    return "container-filter-excluded";
  case ConSanAccessPolicyReason::TargetCapabilityUnavailable:
    return "target-capability-unavailable";
  case ConSanAccessPolicyReason::NonAccessInstruction:
    return "non-access-instruction";
  case ConSanAccessPolicyReason::InvalidInstructionSize:
    return "invalid-instruction-size";
  case ConSanAccessPolicyReason::InvalidAccessWidth:
    return "invalid-access-width";
  case ConSanAccessPolicyReason::MissingAddressOperand:
    return "missing-address-operand";
  case ConSanAccessPolicyReason::RangeEncodingUnavailable:
    return "range-encoding-unavailable";
  case ConSanAccessPolicyReason::InstructionOutOfBounds:
    return "instruction-out-of-bounds";
  case ConSanAccessPolicyReason::UnsupportedMnemonic:
    return "unsupported-mnemonic";
  case ConSanAccessPolicyReason::UnsupportedFlatEncoding:
    return "unsupported-flat-encoding";
  case ConSanAccessPolicyReason::NonzeroFlatOffset:
    return "nonzero-flat-offset";
  case ConSanAccessPolicyReason::ReservedFlatAddressRegister:
    return "reserved-flat-address-register";
  case ConSanAccessPolicyReason::NonGroupAddressSpace:
    return "non-group-address-space";
  case ConSanAccessPolicyReason::FlatProvenancePolicyExcluded:
    return "flat-provenance-policy-excluded";
  case ConSanAccessPolicyReason::ConflictingPhysicalAliases:
    return "conflicting-physical-aliases";
  case ConSanAccessPolicyReason::Count:
    break;
  }
  return "invalid-access-policy-reason";
}

static_assert(kConSanAccessPolicyReasons.size() ==
              static_cast<size_t>(ConSanAccessPolicyReason::Count));

/// Machine-readable reason for a barrier-policy decision.
///
/// Barrier policy owns only whether a decoded barrier event belongs to the
/// selected engine contract and whether inventory has enough semantic facts to
/// request an observation. Register ownership, report capacity, trampoline
/// reachability, and emitted patch shape are intentionally absent: those are
/// later mechanism outcomes recorded by `ConSanCoverageLedger`.
enum class ConSanBarrierPolicyReason : uint8_t {
  /// The event was admitted and therefore needs no exclusion reason.
  None,
  /// Barrier tracking was disabled for this request.
  TrackingDisabled,
  /// The selected engine mutates barriers but does not observe their epochs.
  EngineMutationOnly,
  /// A diagnostic-only container filter excluded every alias of the event.
  ContainerFilterExcluded,
  /// The event belongs only to a ROCclr runtime kernel excluded by policy.
  RuntimeKernelExcluded,
  /// The selected target/engine contract does not support this barrier scope.
  TargetCapabilityUnavailable,
  /// The barrier instruction is not the encoding width required by its engine.
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

/// Complete iterable set of barrier-policy reasons, including `None`.
inline constexpr std::array<ConSanBarrierPolicyReason, 12> kConSanBarrierPolicyReasons = {
    ConSanBarrierPolicyReason::None,
    ConSanBarrierPolicyReason::TrackingDisabled,
    ConSanBarrierPolicyReason::EngineMutationOnly,
    ConSanBarrierPolicyReason::ContainerFilterExcluded,
    ConSanBarrierPolicyReason::RuntimeKernelExcluded,
    ConSanBarrierPolicyReason::TargetCapabilityUnavailable,
    ConSanBarrierPolicyReason::InvalidBarrierEncoding,
    ConSanBarrierPolicyReason::UnqualifiedSyncSequence,
    ConSanBarrierPolicyReason::AmbiguousSequenceMembership,
    ConSanBarrierPolicyReason::MissingCompletingEvent,
    ConSanBarrierPolicyReason::RedundantAdjacentFullBarrier,
    ConSanBarrierPolicyReason::ConflictingPhysicalAliases,
};

/// Return the stable diagnostic spelling of a barrier-policy reason.
[[nodiscard]] constexpr std::string_view
consan_barrier_policy_reason_name(ConSanBarrierPolicyReason reason) {
  switch (reason) {
  case ConSanBarrierPolicyReason::None:
    return "none";
  case ConSanBarrierPolicyReason::TrackingDisabled:
    return "tracking-disabled";
  case ConSanBarrierPolicyReason::EngineMutationOnly:
    return "engine-mutation-only";
  case ConSanBarrierPolicyReason::ContainerFilterExcluded:
    return "container-filter-excluded";
  case ConSanBarrierPolicyReason::RuntimeKernelExcluded:
    return "runtime-kernel-excluded";
  case ConSanBarrierPolicyReason::TargetCapabilityUnavailable:
    return "target-capability-unavailable";
  case ConSanBarrierPolicyReason::InvalidBarrierEncoding:
    return "invalid-barrier-encoding";
  case ConSanBarrierPolicyReason::UnqualifiedSyncSequence:
    return "unqualified-sync-sequence";
  case ConSanBarrierPolicyReason::AmbiguousSequenceMembership:
    return "ambiguous-sequence-membership";
  case ConSanBarrierPolicyReason::MissingCompletingEvent:
    return "missing-completing-event";
  case ConSanBarrierPolicyReason::RedundantAdjacentFullBarrier:
    return "redundant-adjacent-full-barrier";
  case ConSanBarrierPolicyReason::ConflictingPhysicalAliases:
    return "conflicting-physical-aliases";
  case ConSanBarrierPolicyReason::Count:
    break;
  }
  return "invalid-barrier-policy-reason";
}

static_assert(kConSanBarrierPolicyReasons.size() ==
              static_cast<size_t>(ConSanBarrierPolicyReason::Count));

/// Target-neutral kind of evidence requested from one physical probe.
///
/// The kinds deliberately name the semantic observation rather than an ISA
/// sequence, report offset, or patch kind. This lets all targets share policy
/// while their lowerers remain free to implement the same intent differently.
enum class ConSanProbeIntentKind : uint8_t {
  /// SuperCollider observes whether a redundant access agrees with the guest.
  RedundantAccessObservation,
  /// Record/Replay appends an exact access record for later host processing.
  AccessRecord,
  /// Sampled conditionally records the access selected by its sampling policy.
  SampledAccess,
  /// InlineShadow updates and checks exact on-device shadow state.
  ExactShadowAccess,
  /// Record/Replay appends one exact barrier event for later host processing.
  BarrierRecord,
  /// Sampled closes one selected causal window at a logical barrier sequence.
  SampledBarrierEpoch,
  /// InlineShadow advances exact on-device epoch state at barrier completion.
  ExactBarrierEpoch,
  /// One-past-the-end iteration bound; never an observation intent.
  Count,
};

/// Complete iterable set of probe intent kinds.
inline constexpr std::array<ConSanProbeIntentKind, 7> kConSanProbeIntentKinds = {
    ConSanProbeIntentKind::RedundantAccessObservation,
    ConSanProbeIntentKind::AccessRecord,
    ConSanProbeIntentKind::SampledAccess,
    ConSanProbeIntentKind::ExactShadowAccess,
    ConSanProbeIntentKind::BarrierRecord,
    ConSanProbeIntentKind::SampledBarrierEpoch,
    ConSanProbeIntentKind::ExactBarrierEpoch,
};

/// Return the stable diagnostic spelling of an access probe intent kind.
[[nodiscard]] constexpr std::string_view consan_probe_intent_kind_name(ConSanProbeIntentKind kind) {
  switch (kind) {
  case ConSanProbeIntentKind::RedundantAccessObservation:
    return "redundant-access-observation";
  case ConSanProbeIntentKind::AccessRecord:
    return "access-record";
  case ConSanProbeIntentKind::SampledAccess:
    return "sampled-access";
  case ConSanProbeIntentKind::ExactShadowAccess:
    return "exact-shadow-access";
  case ConSanProbeIntentKind::BarrierRecord:
    return "barrier-record";
  case ConSanProbeIntentKind::SampledBarrierEpoch:
    return "sampled-barrier-epoch";
  case ConSanProbeIntentKind::ExactBarrierEpoch:
    return "exact-barrier-epoch";
  case ConSanProbeIntentKind::Count:
    break;
  }
  return "invalid-probe-intent-kind";
}

static_assert(kConSanProbeIntentKinds.size() == static_cast<size_t>(ConSanProbeIntentKind::Count));

/// Required ordering of a probe relative to the pristine guest instruction.
///
/// This is a semantic placement constraint. It does not choose inline versus
/// trampoline lowering or prescribe how a target preserves clobbered operands.
enum class ConSanProbePosition : uint8_t {
  /// Complete the observation before executing the guest instruction.
  Before,
  /// Complete the observation after executing the guest instruction.
  After,
  /// One-past-the-end iteration bound; never a placement constraint.
  Count,
};

/// Complete iterable set of relative probe positions.
inline constexpr std::array<ConSanProbePosition, 2> kConSanProbePositions = {
    ConSanProbePosition::Before,
    ConSanProbePosition::After,
};

/// Return the stable diagnostic spelling of a relative probe position.
[[nodiscard]] constexpr std::string_view consan_probe_position_name(ConSanProbePosition position) {
  switch (position) {
  case ConSanProbePosition::Before:
    return "before";
  case ConSanProbePosition::After:
    return "after";
  case ConSanProbePosition::Count:
    break;
  }
  return "invalid-probe-position";
}

static_assert(kConSanProbePositions.size() == static_cast<size_t>(ConSanProbePosition::Count));

/// Which lanes a probe must observe when it executes.
///
/// Access policy currently requires the complete active execution mask. The
/// named type leaves room for later single-lane control probes without letting
/// a lowerer silently weaken an access intent.
enum class ConSanLaneMaskPolicy : uint8_t {
  /// Observe every lane active at the original instruction.
  ActiveExecutionMask,
  /// One-past-the-end iteration bound; never a lane policy.
  Count,
};

/// Complete iterable set of lane-mask policies.
inline constexpr std::array<ConSanLaneMaskPolicy, 1> kConSanLaneMaskPolicies = {
    ConSanLaneMaskPolicy::ActiveExecutionMask,
};

/// Return the stable diagnostic spelling of a lane-mask policy.
[[nodiscard]] constexpr std::string_view consan_lane_mask_policy_name(ConSanLaneMaskPolicy policy) {
  switch (policy) {
  case ConSanLaneMaskPolicy::ActiveExecutionMask:
    return "active-execution-mask";
  case ConSanLaneMaskPolicy::Count:
    break;
  }
  return "invalid-lane-mask-policy";
}

static_assert(kConSanLaneMaskPolicies.size() == static_cast<size_t>(ConSanLaneMaskPolicy::Count));

/// Whether losing an admitted intent invalidates the requested static contract.
///
/// Required intents must lower successfully for ConSan to claim complete
/// static instrumentation. Best-effort intents may be lost without that claim,
/// but their loss remains visible in the coverage ledger.
enum class ConSanProbeRequirement : uint8_t {
  /// Lowering failure is a coverage gap for the requested engine contract.
  Required,
  /// Lowering may reject the enhancement while retaining minimum behavior.
  BestEffort,
  /// One-past-the-end iteration bound; never an intent requirement.
  Count,
};

/// Complete iterable set of probe requirements.
inline constexpr std::array<ConSanProbeRequirement, 2> kConSanProbeRequirements = {
    ConSanProbeRequirement::Required,
    ConSanProbeRequirement::BestEffort,
};

/// Return the stable diagnostic spelling of a probe requirement.
[[nodiscard]] constexpr std::string_view
consan_probe_requirement_name(ConSanProbeRequirement requirement) {
  switch (requirement) {
  case ConSanProbeRequirement::Required:
    return "required";
  case ConSanProbeRequirement::BestEffort:
    return "best-effort";
  case ConSanProbeRequirement::Count:
    break;
  }
  return "invalid-probe-requirement";
}

static_assert(kConSanProbeRequirements.size() ==
              static_cast<size_t>(ConSanProbeRequirement::Count));

/// One engine-policy decision for one normalized semantic access range.
///
/// Container aliases are attribution only and are retained as names for
/// diagnostics. The semantic identity is authoritative. An admitted decision
/// normally cites one coalesced physical intent; unsupported and not-applicable
/// decisions cite none.
struct ConSanSiteDecision {
  /// Engine whose contract made this decision.
  ConSanCapabilityEngine engine = ConSanCapabilityEngine::Count;
  /// Stable identity of the logical access range being decided.
  SemanticSiteId semantic_site;
  /// Applicability/admission result that lowering cannot change.
  ConSanSiteDecisionKind kind = ConSanSiteDecisionKind::NotApplicable;
  /// Typed explanation; `None` is reserved for admitted decisions.
  ConSanAccessPolicyReason reason = ConSanAccessPolicyReason::AccessFamilyDisabled;
  /// Target-neutral intents requested for this semantic range.
  std::vector<ConSanProbeIntentId> intent_ids;
  /// Every symbol-container name through which this physical site was decoded.
  std::vector<std::string> source_containers;

  bool operator==(const ConSanSiteDecision &) const = default;
};

/// One engine-policy decision for one decoded barrier event.
///
/// The decision is about original-program meaning, not a patch. A paired
/// signal and wait may therefore both cite one completion intent, while a
/// Record/Replay decision normally cites the per-event intent at its own
/// instruction. `source_containers` preserves every symbol alias solely for
/// diagnostics; `semantic_site` remains the authoritative identity.
struct ConSanBarrierSiteDecision {
  /// Engine whose barrier contract made this decision.
  ConSanCapabilityEngine engine = ConSanCapabilityEngine::Count;
  /// Stable identity of the original synchronization event being decided.
  SemanticSiteId semantic_site;
  /// Applicability/admission result that lowering cannot change.
  ConSanSiteDecisionKind kind = ConSanSiteDecisionKind::NotApplicable;
  /// Typed explanation; `None` is reserved for admitted decisions.
  ConSanBarrierPolicyReason reason = ConSanBarrierPolicyReason::TrackingDisabled;
  /// Target-neutral intents that collectively observe this event.
  std::vector<ConSanProbeIntentId> intent_ids;
  /// Every symbol-container name through which this event was decoded.
  std::vector<std::string> source_containers;

  bool operator==(const ConSanBarrierSiteDecision &) const = default;
};

/// One target-neutral observation requested at a physical instruction.
///
/// `covered_semantic_sites` can contain multiple access ranges from one
/// instruction or every event in one logical barrier sequence. That coalescing
/// is explicit and keeps placement capacity at one insertion point. The intent
/// contains no chosen registers, report addresses, machine words, branch route,
/// or patch kind.
struct ConSanProbeIntent {
  /// Stable plan-local identifier cited by semantic decisions.
  ConSanProbeIntentId id;
  /// Engine whose evidence semantics this intent implements.
  ConSanCapabilityEngine engine = ConSanCapabilityEngine::Count;
  /// Physical instruction at which lowering must place the coalesced probe.
  PhysicalSiteId physical_site;
  /// All logical access ranges or synchronization events observed by the probe.
  std::vector<SemanticSiteId> covered_semantic_sites;
  /// Semantic evidence operation requested from target lowering.
  ConSanProbeIntentKind kind = ConSanProbeIntentKind::Count;
  /// Required ordering relative to the guest instruction.
  ConSanProbePosition position = ConSanProbePosition::Count;
  /// Lanes whose original behavior the probe must observe.
  ConSanLaneMaskPolicy lane_mask = ConSanLaneMaskPolicy::Count;
  /// Whether losing this intent creates a contract coverage gap.
  ConSanProbeRequirement requirement = ConSanProbeRequirement::Count;

  bool operator==(const ConSanProbeIntent &) const = default;
};

/// Complete target-neutral semantic-policy output for one engine request.
///
/// Separate decision sequences cover semantic access ranges and barrier events;
/// their shared intents cover physical insertion points. The plan is a value
/// object and contains no borrowed inventory references, so a result may safely
/// retain it across retries and image moves.
struct ConSanObservationPlan {
  /// Selected engine shared by every decision and intent in this plan.
  ConSanCapabilityEngine engine = ConSanCapabilityEngine::Count;
  /// Deterministically ordered semantic access decisions.
  std::vector<ConSanSiteDecision> site_decisions;
  /// Deterministically ordered semantic barrier-event decisions.
  std::vector<ConSanBarrierSiteDecision> barrier_site_decisions;
  /// Deterministically ordered coalesced physical probe intents.
  std::vector<ConSanProbeIntent> probe_intents;

  /// Verify plan-local identities and the decision/intent relationship.
  [[nodiscard]] bool valid() const;

  /// Resolve one plan-local intent identifier, rejecting invalid or stale IDs.
  [[nodiscard]] const ConSanProbeIntent *intent(ConSanProbeIntentId id) const;

  /// Transactionally append a valid same-engine policy fragment.
  ///
  /// Plan-local intent identifiers in the fragment are rebased and every
  /// decision reference is rewritten. On invalid input or an engine mismatch,
  /// this plan is unchanged and the function returns false.
  [[nodiscard]] bool append(const ConSanObservationPlan &fragment);

  bool operator==(const ConSanObservationPlan &) const = default;
};

/// Mutable request parameters consumed by pure access policy.
///
/// The borrowed spans and filter are read only during the policy call; no
/// output retains them. Production requests normally enable both access
/// families for MOI, while SuperCollider reflects its two explicit probe
/// switches. `reserved_for_synchronization` prevents an ordinary memory event
/// from being double-counted before Slice 4C takes ownership of its ordering
/// semantics.
struct ConSanAccessPolicyRequest {
  /// Engine whose access contract should be evaluated.
  ConSanCapabilityEngine engine = ConSanCapabilityEngine::Count;
  /// Whether native LDS and direct-to-LDS sites belong to this request.
  bool native_lds_enabled = false;
  /// Whether group-FLAT sites belong to this request.
  bool group_flat_enabled = false;
  /// Confidence threshold for conservatively inferred group-FLAT provenance.
  ConSanFlatProvenanceMode flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  /// Optional diagnostic-only container substring filter.
  std::string_view container_filter;
  /// Physical ordinary-memory sites delegated to synchronization policy.
  std::span<const PhysicalSiteId> reserved_for_synchronization;
};

/// Pure access-policy output plus typed construction failures.
///
/// A conflicting physical alias is represented both as unsupported decisions
/// and in `errors`, allowing unit tests and adapters to reject the transform
/// without parsing prose. No warning or error is emitted as a side effect.
struct ConSanAccessPolicyResult {
  /// Complete plan, including explicit non-admission decisions.
  ConSanObservationPlan plan;
  /// Typed fatal policy-construction failures.
  std::vector<ConSanAccessPolicyReason> errors;

  /// Return whether policy produced a usable, internally valid plan.
  [[nodiscard]] bool valid() const { return errors.empty() && plan.valid(); }

  bool operator==(const ConSanAccessPolicyResult &) const = default;
};

/// Immutable request parameters consumed by pure barrier policy.
///
/// The request selects semantic applicability only. It contains no report
/// address, register assignment, code-cave budget, or mutable result object, so
/// the same inventory/request pair always yields the same plan fragment.
struct ConSanBarrierPolicyRequest {
  /// Engine whose barrier-observation contract should be evaluated.
  ConSanCapabilityEngine engine = ConSanCapabilityEngine::Count;
  /// Whether decoded barrier events belong to this request.
  bool tracking_enabled = false;
  /// Optional diagnostic-only container substring filter.
  std::string_view container_filter;
};

/// Pure barrier-policy output plus typed construction failures.
///
/// Ordinary exclusions remain explicit decisions and do not enter `errors`.
/// Only contradictions such as incompatible physical aliases make the result
/// unusable, allowing callers to fail closed without parsing diagnostics.
struct ConSanBarrierPolicyResult {
  /// Barrier-only observation-plan fragment, ready for transactional append.
  ConSanObservationPlan plan;
  /// Typed fatal policy-construction failures.
  std::vector<ConSanBarrierPolicyReason> errors;

  /// Return whether policy produced a usable, internally valid fragment.
  [[nodiscard]] bool valid() const { return errors.empty() && plan.valid(); }

  bool operator==(const ConSanBarrierPolicyResult &) const = default;
};

/// Return whether shared synchronization analysis proved every fact required
/// by the current Sampled barrier-epoch contract. The predicate is declared at
/// the policy boundary because both pure admission and the temporary legacy
/// lowerer must agree during incremental migration.
[[nodiscard]] bool
consan_moi_sampled_qualifies_barrier_sequence(const ConSanSyncSequence &sequence);

/// Final mechanism state of one admitted probe intent.
///
/// The outcome is independent of semantic admission: a target lowerer can
/// reject resources or placement, but cannot convert an unsupported decision
/// into an instrumented one.
enum class ConSanLoweringOutcomeKind : uint8_t {
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

/// Complete iterable set of lowering outcomes.
inline constexpr std::array<ConSanLoweringOutcomeKind, 4> kConSanLoweringOutcomeKinds = {
    ConSanLoweringOutcomeKind::Pending,
    ConSanLoweringOutcomeKind::Instrumented,
    ConSanLoweringOutcomeKind::ResourceRejected,
    ConSanLoweringOutcomeKind::PlacementRejected,
};

/// Return the stable diagnostic spelling of a lowering outcome.
[[nodiscard]] constexpr std::string_view
consan_lowering_outcome_kind_name(ConSanLoweringOutcomeKind kind) {
  switch (kind) {
  case ConSanLoweringOutcomeKind::Pending:
    return "pending";
  case ConSanLoweringOutcomeKind::Instrumented:
    return "instrumented";
  case ConSanLoweringOutcomeKind::ResourceRejected:
    return "resource-rejected";
  case ConSanLoweringOutcomeKind::PlacementRejected:
    return "placement-rejected";
  case ConSanLoweringOutcomeKind::Count:
    break;
  }
  return "invalid-lowering-outcome-kind";
}

static_assert(kConSanLoweringOutcomeKinds.size() ==
              static_cast<size_t>(ConSanLoweringOutcomeKind::Count));

/// One admitted intent joined to its current lowering result.
///
/// `detail` is diagnostic context only; control flow and trust evaluation use
/// the typed outcome. Retaining the complete intent makes ledger inspection
/// independent of candidate-vector or patch-vector lifetime.
struct ConSanIntentCoverageEntry {
  /// Target-neutral intent whose mechanism status is being tracked.
  ConSanProbeIntent intent;
  /// Current finalization state for that intent.
  ConSanLoweringOutcomeKind lowering = ConSanLoweringOutcomeKind::Pending;
  /// Optional human-readable mechanism detail with no policy meaning.
  std::string detail;

  bool operator==(const ConSanIntentCoverageEntry &) const = default;
};

/// Joined semantic coverage state without conflating policy and lowering.
///
/// The ledger owns copies of semantic decisions and admitted intents. Runtime
/// selection and published-evidence axes will be added by later slices; this
/// first slice establishes the static-policy/lowering boundary. Callers may
/// update only a known admitted intent, and queries never infer success from a
/// patch kind or warning string.
class ConSanCoverageLedger {
public:
  ConSanCoverageLedger() = default;
  explicit ConSanCoverageLedger(const ConSanObservationPlan &plan);

  /// Return immutable semantic decisions copied from the observation plan.
  [[nodiscard]] std::span<const ConSanSiteDecision> site_decisions() const {
    return site_decisions_;
  }

  /// Return immutable barrier-event decisions copied from the plan.
  [[nodiscard]] std::span<const ConSanBarrierSiteDecision> barrier_site_decisions() const {
    return barrier_site_decisions_;
  }

  /// Return immutable joined intent/lowering entries.
  [[nodiscard]] std::span<const ConSanIntentCoverageEntry> intent_entries() const {
    return intent_entries_;
  }

  /// Resolve one admitted intent entry by its plan-local identifier.
  [[nodiscard]] const ConSanIntentCoverageEntry *intent_entry(ConSanProbeIntentId id) const;

  /// Publish a final or pending mechanism outcome for one known intent.
  /// Returns false for an invalid/stale identifier or invalid enum value.
  [[nodiscard]] bool set_lowering_outcome(ConSanProbeIntentId id, ConSanLoweringOutcomeKind outcome,
                                          std::string detail = {});

  /// Update every intent placed at one physical instruction. This is the
  /// narrow compatibility seam used by legacy lowerers that do not yet retain
  /// plan-local intent identifiers. Returns the number of updated intents.
  [[nodiscard]] size_t set_physical_lowering_outcome(const PhysicalSiteId &physical_site,
                                                     ConSanLoweringOutcomeKind outcome,
                                                     std::string detail = {});

  /// Return whether every required admitted intent has been instrumented.
  /// This deliberately makes no claim about runtime selection or retained
  /// evidence, which are separate later ledger axes.
  [[nodiscard]] bool all_required_intents_instrumented() const;

  bool operator==(const ConSanCoverageLedger &) const = default;

private:
  std::vector<ConSanSiteDecision> site_decisions_;
  std::vector<ConSanBarrierSiteDecision> barrier_site_decisions_;
  std::vector<ConSanIntentCoverageEntry> intent_entries_;
};

/// Evaluate target-neutral access applicability and admission without mutating
/// inventory, options, diagnostics, candidates, resource plans, or patches.
[[nodiscard]] ConSanAccessPolicyResult
plan_consan_access_observation(const ProgramInventory &inventory,
                               const ConSanAccessPolicyRequest &request);

/// Evaluate target-neutral barrier applicability and sequence admission without
/// mutating inventory, options, diagnostics, candidates, resource plans, or
/// patches.
[[nodiscard]] ConSanBarrierPolicyResult
plan_consan_barrier_observation(const ProgramInventory &inventory,
                                const ConSanBarrierPolicyRequest &request);

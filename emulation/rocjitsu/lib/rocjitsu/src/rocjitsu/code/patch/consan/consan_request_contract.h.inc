// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Process and installation policy owned by the runtime hook rather than the
/// code-object transformer.
///
/// This type decides whether instrumentation is active, whether an inability
/// to install it rejects a load, and how much process-wide storage may remain
/// live. It contains no sanitizer-site policy and is intentionally not an
/// semantic input to the production transformer.
struct RuntimePolicy {
  /// Whether the ConSan hook participates in code-object loads.
  bool enabled = false;
  /// Reject a load when ConSan cannot establish its declared contract.
  bool fail_closed = false;
  /// Require at least one installed patch even when fail-closed is otherwise
  /// disabled.
  bool require_patch = false;
  /// Maximum bytes simultaneously reserved by in-progress transformations.
  std::optional<uint64_t> process_concurrent_transform_limit_bytes;
  /// Maximum retained replacement-image bytes across the process.
  std::optional<uint64_t> process_patched_image_limit_bytes;
  /// Maximum retained growth beyond original images across the process.
  std::optional<uint64_t> process_patched_image_growth_limit_bytes;

  bool operator==(const RuntimePolicy &) const = default;
};

/// Focused-test, diagnostic, and early-development controls that are outside
/// ConSan's supported semantic request.
///
/// Keeping this type separate makes a call site visibly opt into prototype
/// probes, forced allocation paths, or explicit registers. None of these
/// fields may influence the meaning of a normal `ConSanRequest`.
struct ConSanDebugOverrides {
  /// Replace a selected instruction with a NOP for focused patch testing.
  bool probe_nop = false;
  /// Route a selected instruction through a NOP trampoline.
  bool probe_trampoline_nop = false;
  /// Replace a selected instruction with program termination.
  bool probe_endpgm = false;
  /// Apply the termination probe only to native LDS instructions.
  bool probe_lds_endpgm = false;
  /// Replace a selected group-FLAT site with a direct trap probe.
  bool probe_flat_trap = false;
  /// Turn a statically unique unmatched barrier wait into a fail-fast probe.
  bool abort_unmatched_barrier_wait = false;
  /// Retain address-partition masks for diagnostic inspection.
  bool moi_partition_mask_debug = false;
  /// Force the legacy VGPR spill path even when a cheaper plan exists.
  bool test_force_vgpr_spill = false;
  /// Force private epoch state even when persistent VGPR state is available.
  bool test_force_private_epoch = false;
  /// Seed an odd Inline exact-shadow publication for a teardown fixture.
  bool test_seed_inline_exact_odd = false;
  /// Restrict candidate discovery to matching kernel names in focused tests.
  std::string test_kernel_name_filter;

  /// Explicit scratch VGPR, bypassing normal resource selection.
  std::optional<uint16_t> scratch_vgpr;
  /// Explicit even SGPR pair used to preserve EXEC.
  std::optional<uint16_t> moi_exec_save_sgpr;
  /// Explicit scalar owner register.
  std::optional<uint16_t> moi_owner_sgpr;
  /// Explicit vector owner register.
  std::optional<uint16_t> moi_owner_vgpr;
  /// Explicit vector epoch register.
  std::optional<uint16_t> moi_epoch_vgpr;

  /// Runtime assertion requiring at least one visible MOI record.
  bool moi_require_records = false;
  /// Runtime assertion requiring at least one emitted diagnostic.
  bool moi_require_diagnostics = false;
  /// Runtime assertion forbidding emitted diagnostics.
  bool moi_forbid_diagnostics = false;
  /// Runtime assertion requiring a Record/Replay conflict.
  bool moi_require_replay_conflict = false;
  /// Runtime assertion forbidding bounded-evidence overflow.
  bool moi_forbid_overflow = false;

  bool operator==(const ConSanDebugOverrides &) const = default;
};

/// Explicit request to corrupt or perturb a program for validation.
///
/// Mutation is transformational and is never implied by ordinary sanitizer
/// activation. Exact site identities and selection indices live here so they
/// cannot accidentally alter semantic inventory or report planning for a
/// normal request.
struct MutationRequest {
  bool fault_drop_barrier = false;
  bool fault_allow_destructive_incomplete_barrier_drop = false;
  bool fault_move_barrier = false;
  bool fault_allow_completing_conditional_barrier_move = false;
  bool fault_allow_destructive_divergent_barrier_move = false;
  bool fault_mutate_barrier_id_scope = false;
  bool fault_mutate_barrier_participants = false;
  bool fault_atomic_wrong_address = false;
  bool fault_atomic_weaken_order = false;
  ConSanAtomicOrderEdge fault_atomic_order_edge = ConSanAtomicOrderEdge::Any;
  bool fault_atomic_weaken_scope = false;
  bool fault_lds_wrong_address = false;
  bool fault_ordinary_wrong_address = false;
  bool fault_ordinary_weaken_order = false;
  bool fault_ordinary_weaken_scope = false;

  uint32_t fault_atomic_address_delta = 4;
  std::optional<uint16_t> fault_lds_address_vgpr;
  uint32_t fault_ordinary_address_delta = 4;
  bool fault_dry_run = false;
  bool fault_require_exactly_one = false;
  uint32_t fault_reservation_timeout_ms = 30000;
  std::optional<uint32_t> fault_load_occurrence;

  uint32_t fault_barrier_index = 0;
  uint32_t fault_atomic_index = 0;
  uint32_t fault_lds_index = 0;
  uint32_t fault_ordinary_index = 0;
  std::string fault_site_identity;
  std::string fault_barrier_destination_identity;
  std::string fault_barrier_sequence_identity;
  std::string fault_barrier_companion_site_identity;
  std::string fault_barrier_companion_sequence_identity;
  std::optional<int32_t> fault_barrier_target_id;
  std::optional<uint32_t> fault_barrier_target_participant_count;
  std::optional<uint64_t> fault_barrier_target_participant_mask;
  ConSanBarrierMoveDirection fault_barrier_move_direction =
      ConSanBarrierMoveDirection::LegacyMarker;

  ConSanPerturbationKind sc_perturb_kind = ConSanPerturbationKind::None;
  ConSanPerturbationEdge sc_perturb_edge = ConSanPerturbationEdge::Release;
  std::string sc_perturb_identity;
  uint32_t sc_perturb_index = 0;
  uint32_t sc_perturb_max = 1;
  uint32_t sc_perturb_sleep = 1;
  uint32_t sc_perturb_required_count = 0;

  /// Return whether a fault transform (as opposed to timing perturbation) was
  /// requested.
  [[nodiscard]] bool has_fault_mutation() const {
    return fault_drop_barrier || fault_move_barrier || fault_mutate_barrier_id_scope ||
           fault_mutate_barrier_participants || fault_atomic_wrong_address ||
           fault_atomic_weaken_order || fault_atomic_weaken_scope || fault_lds_wrong_address ||
           fault_ordinary_wrong_address || fault_ordinary_weaken_order ||
           fault_ordinary_weaken_scope;
  }

  /// Return whether either a fault transform or SuperCollider perturbation was
  /// requested.
  [[nodiscard]] bool has_mutation() const {
    return has_fault_mutation() || sc_perturb_kind != ConSanPerturbationKind::None;
  }

  bool operator==(const MutationRequest &) const = default;
};

/// Return a copy retaining selection/proof and perturbation data but with every
/// live fault-transform switch disabled. Inventory sizing uses this operation
/// to analyze pristine sanitizer semantics without losing the caller's exact
/// selection metadata.
[[nodiscard]] inline MutationRequest
without_consan_fault_mutations(const MutationRequest &request) {
  MutationRequest result = request;
  result.fault_drop_barrier = false;
  result.fault_move_barrier = false;
  result.fault_mutate_barrier_id_scope = false;
  result.fault_mutate_barrier_participants = false;
  result.fault_atomic_wrong_address = false;
  result.fault_atomic_weaken_order = false;
  result.fault_atomic_weaken_scope = false;
  result.fault_lds_wrong_address = false;
  result.fault_ordinary_wrong_address = false;
  result.fault_ordinary_weaken_order = false;
  result.fault_ordinary_weaken_scope = false;
  result.fault_require_exactly_one = false;
  return result;
}

/// Return whether semantic inventory must retain extended barrier pairs.
///
/// Extended pairs are needed by barrier mutations, unmatched-barrier abort
/// diagnostics, and Sampled barrier observation. This query deliberately uses
/// only typed semantic inputs: runtime bindings, target mechanics, and legacy
/// lowering options cannot change the inventory shape.
[[nodiscard]] constexpr bool
consan_requires_extended_barrier_pairs(const ConSanRequest &request,
                                       const ConSanDebugOverrides &debug,
                                       const MutationRequest &mutation) {
  return mutation.fault_drop_barrier || mutation.fault_move_barrier ||
         mutation.fault_mutate_barrier_id_scope || mutation.fault_mutate_barrier_participants ||
         debug.abort_unmatched_barrier_wait ||
         (request.moi_engine == ConSanMoiEngine::Sampled && request.moi_track_barriers);
}

/// Provenance of the runtime facts supplied to one ConSan transformation.
///
/// This identifies the query mechanism, not the GPU architecture and not the
/// sanitizer engine. Keeping provenance explicit lets fixtures supply
/// simulator facts without pretending that they came from HSA, while the
/// construction-time `Unknown` state prevents an unqueried default object from
/// being mistaken for a backend that lacks every optional facility.
enum class ConSanRuntimeBackend : uint8_t {
  /// No backend has supplied facts yet; validated production inputs reject
  /// this construction-time sentinel.
  Unknown,
  /// Capabilities were queried from a physical HSA agent.
  PhysicalHsa,
  /// Capabilities were supplied by a RocJitsu simulation configuration.
  RocJitsuSimulator,
  /// One past the last declared value. This is an iteration bound and an
  /// invalid runtime backend, not a backend that may supply capabilities.
  Count,
};

/// Every declared runtime-backend value, including the construction sentinel
/// but excluding the one-past-the-end iteration bound. Keeping this inventory
/// beside the enum makes exhaustive tests and diagnostic rendering fail
/// visibly when the backend vocabulary grows.
inline constexpr std::array<ConSanRuntimeBackend, 3> kConSanRuntimeBackends = {
    ConSanRuntimeBackend::Unknown,
    ConSanRuntimeBackend::PhysicalHsa,
    ConSanRuntimeBackend::RocJitsuSimulator,
};

/// Return the stable diagnostic spelling of a runtime-backend provenance.
/// Invalid values, including `Count`, deliberately share an invalid spelling
/// so they cannot masquerade as a queried backend in logs or test fixtures.
[[nodiscard]] constexpr std::string_view consan_runtime_backend_name(ConSanRuntimeBackend backend) {
  switch (backend) {
  case ConSanRuntimeBackend::Unknown:
    return "unknown";
  case ConSanRuntimeBackend::PhysicalHsa:
    return "physical-hsa";
  case ConSanRuntimeBackend::RocJitsuSimulator:
    return "rocjitsu-simulator";
  case ConSanRuntimeBackend::Count:
    break;
  }
  return "invalid-runtime-backend";
}

static_assert(kConSanRuntimeBackends.size() == static_cast<size_t>(ConSanRuntimeBackend::Count));

/// Facts discovered from the selected execution backend and agent.
///
/// These are runtime facts, not implications of a `gfx*` name. Optional
/// quantities distinguish an unavailable query from a genuine zero-sized
/// facility. Consumers state their requirements explicitly and reject missing
/// properties before allocating or lowering code that depends on them.
struct RuntimeCapabilities {
  ConSanRuntimeBackend backend = ConSanRuntimeBackend::Unknown;
  /// A global allocation can be made visible to both host and selected agent.
  bool host_device_visible_memory = false;
  /// Host observations are coherent with committed device publications.
  bool host_device_coherent_memory = false;
  /// The report memory supports the atomic publication protocol required by
  /// the selected evidence schema.
  bool device_atomic_publication = false;
  /// Maximum eligible report allocation exposed by the backend query.
  std::optional<uint64_t> max_report_allocation_bytes;
  /// Maximum effective group-segment aperture for a workgroup dispatch.
  std::optional<uint32_t> max_workgroup_lds_bytes;
  /// The backend can bind a replacement code object to an executable.
  bool executable_binding = false;
  /// The backend can apply planned private/group segment additions at dispatch.
  bool dispatch_segment_binding = false;

  bool operator==(const RuntimeCapabilities &) const = default;
};

/// Subset of runtime properties required by one prospective operation.
///
/// A consumer builds this value before allocating a report, binding an
/// executable, or planning a dispatch segment, then validates it against the
/// separately queried `RuntimeCapabilities`. A false boolean means the
/// operation does not depend on that property; it never claims that the
/// backend lacks the facility. The optional minimum is a lower bound rather
/// than an allocation request, so this type owns no resource or address.
struct RuntimeCapabilityRequirements {
  bool host_device_visible_memory = false;
  bool host_device_coherent_memory = false;
  bool device_atomic_publication = false;
  std::optional<uint64_t> minimum_report_allocation_bytes;
  bool max_workgroup_lds_bytes = false;
  bool executable_binding = false;
  bool dispatch_segment_binding = false;

  bool operator==(const RuntimeCapabilityRequirements &) const = default;
};

/// Lifetime domain attached to addresses in `BoundRuntimeResources`.
///
/// The scope tells the runtime coordinator when an address must be rebound or
/// retired; it does not transfer ownership to the transformer. `Unbound` is a
/// construction-time state in which all addresses must be absent. Every other
/// value states the narrowest runtime identity that keeps the binding valid.
enum class ConSanRuntimeResourceScope : uint8_t {
  /// The resource has not been bound; addresses must be absent.
  Unbound,
  /// The resource belongs to one code-object reader load.
  CodeObject,
  /// The resource remains valid for an executable lifetime.
  Executable,
  /// The resource is rebound for each dispatch generation.
  Dispatch,
  /// One past the last declared value. This is an iteration bound and never a
  /// valid resource lifetime.
  Count,
};

/// Every valid resource-scope value, including the explicitly unbound state
/// but excluding the one-past-the-end iteration bound.
inline constexpr std::array<ConSanRuntimeResourceScope, 4> kConSanRuntimeResourceScopes = {
    ConSanRuntimeResourceScope::Unbound,
    ConSanRuntimeResourceScope::CodeObject,
    ConSanRuntimeResourceScope::Executable,
    ConSanRuntimeResourceScope::Dispatch,
};

/// Return the stable diagnostic spelling of a runtime-resource lifetime.
/// Invalid values never acquire a plausible lifetime name.
[[nodiscard]] constexpr std::string_view
consan_runtime_resource_scope_name(ConSanRuntimeResourceScope scope) {
  switch (scope) {
  case ConSanRuntimeResourceScope::Unbound:
    return "unbound";
  case ConSanRuntimeResourceScope::CodeObject:
    return "code-object";
  case ConSanRuntimeResourceScope::Executable:
    return "executable";
  case ConSanRuntimeResourceScope::Dispatch:
    return "dispatch";
  case ConSanRuntimeResourceScope::Count:
    break;
  }
  return "invalid-runtime-resource-scope";
}

static_assert(kConSanRuntimeResourceScopes.size() ==
              static_cast<size_t>(ConSanRuntimeResourceScope::Count));

/// Concrete runtime allocations and identities bound after address-free
/// planning.
///
/// This value may change when an inventory is retried with an automatic report
/// allocation, but each individual value is immutable once handed to the
/// transformer. It contains no sanitizer policy, mutation request, or test
/// override. Lifetime ownership remains with the runtime coordinator; this
/// projection carries only the identity needed by legacy lowering.
struct BoundRuntimeResources {
  ConSanRuntimeResourceScope scope = ConSanRuntimeResourceScope::Unbound;
  std::optional<uint64_t> report_buffer_address;
  std::optional<uint64_t> moi_report_buffer_address;
  uint64_t moi_report_buffer_size = 0;
  std::optional<ConSanMoiReportLayoutOverride> moi_report_layout;
  uint64_t moi_report_generation = 0;
  uint64_t moi_report_dispatch_id = 0;

  [[nodiscard]] bool bound() const {
    return report_buffer_address.has_value() || moi_report_buffer_address.has_value();
  }

  bool operator==(const BoundRuntimeResources &) const = default;
};

/// Typed reason that one immutable Slice-2 contract is invalid or cannot
/// satisfy a consumer. The enum is intentionally shared by the pure validation
/// functions so parsers and runtime adapters can use one machine-readable
/// vocabulary while rendering their own contextual messages.
enum class ConSanContractIssue : uint8_t {
  /// The contract is valid and all requested runtime facts are present.
  None,
  /// Request construction ended without selecting a sanitizer flavor.
  MissingFlavor,
  /// A typed flavor or engine value lies outside its declared enum.
  InvalidMode,
  /// A sampling stride is zero or outside the supported bounded range.
  InvalidSampleStride,
  /// A sampling offset is not smaller than its associated stride.
  InvalidSampleOffset,
  /// A configured patch budget cannot admit even one patch.
  InvalidPatchBudget,
  /// A debug register override is outside its legal architectural envelope.
  InvalidDebugRegister,
  /// A mutation opt-in is present without its required base mutation.
  InvalidMutationDependency,
  /// A mutation address delta is zero, misaligned, or not encodable.
  InvalidMutationAddressDelta,
  /// SuperCollider perturbation bounds are inconsistent.
  InvalidPerturbationBounds,
  /// A request uses a mode-specific feature with another sanitizer mode.
  ModeConflict,
  /// Two runtime assertions require mutually exclusive outcomes.
  ConflictingRuntimeAssertions,
  /// No execution backend supplied capability facts.
  MissingRuntimeBackend,
  /// Suitable host/device-visible memory was not reported.
  MissingVisibleMemory,
  /// Required coherent host/device observation was not reported.
  MissingCoherentMemory,
  /// Required atomic device publication was not reported.
  MissingAtomicPublication,
  /// The backend did not report a sufficient allocation limit.
  InsufficientReportAllocation,
  /// The backend did not report its workgroup LDS aperture.
  MissingWorkgroupLdsLimit,
  /// The backend cannot bind replacement executables.
  MissingExecutableBinding,
  /// The backend cannot bind dispatch segment additions.
  MissingDispatchSegmentBinding,
  /// An unbound resource carries an address or a bound scope has no address.
  InvalidResourceScope,
  /// A concrete report address is the reserved null address.
  InvalidResourceAddress,
  /// A concrete MOI address and its byte size disagree.
  InvalidResourceSize,
  /// One past the last declared issue. This is an iteration bound and never a
  /// validation result.
  Count,
};

/// Every validation result that may be returned by a contract validator. The
/// one-past-the-end bound is excluded so callers cannot mistake it for an
/// actionable issue.
inline constexpr std::array<ConSanContractIssue, 23> kConSanContractIssues = {
    ConSanContractIssue::None,
    ConSanContractIssue::MissingFlavor,
    ConSanContractIssue::InvalidMode,
    ConSanContractIssue::InvalidSampleStride,
    ConSanContractIssue::InvalidSampleOffset,
    ConSanContractIssue::InvalidPatchBudget,
    ConSanContractIssue::InvalidDebugRegister,
    ConSanContractIssue::InvalidMutationDependency,
    ConSanContractIssue::InvalidMutationAddressDelta,
    ConSanContractIssue::InvalidPerturbationBounds,
    ConSanContractIssue::ModeConflict,
    ConSanContractIssue::ConflictingRuntimeAssertions,
    ConSanContractIssue::MissingRuntimeBackend,
    ConSanContractIssue::MissingVisibleMemory,
    ConSanContractIssue::MissingCoherentMemory,
    ConSanContractIssue::MissingAtomicPublication,
    ConSanContractIssue::InsufficientReportAllocation,
    ConSanContractIssue::MissingWorkgroupLdsLimit,
    ConSanContractIssue::MissingExecutableBinding,
    ConSanContractIssue::MissingDispatchSegmentBinding,
    ConSanContractIssue::InvalidResourceScope,
    ConSanContractIssue::InvalidResourceAddress,
    ConSanContractIssue::InvalidResourceSize,
};

static_assert(kConSanContractIssues.size() == static_cast<size_t>(ConSanContractIssue::Count));

[[nodiscard]] constexpr std::string_view consan_contract_issue_name(ConSanContractIssue issue) {
  switch (issue) {
  case ConSanContractIssue::None:
    return "none";
  case ConSanContractIssue::MissingFlavor:
    return "missing-flavor";
  case ConSanContractIssue::InvalidMode:
    return "invalid-mode";
  case ConSanContractIssue::InvalidSampleStride:
    return "invalid-sample-stride";
  case ConSanContractIssue::InvalidSampleOffset:
    return "invalid-sample-offset";
  case ConSanContractIssue::InvalidPatchBudget:
    return "invalid-patch-budget";
  case ConSanContractIssue::InvalidDebugRegister:
    return "invalid-debug-register";
  case ConSanContractIssue::InvalidMutationDependency:
    return "invalid-mutation-dependency";
  case ConSanContractIssue::InvalidMutationAddressDelta:
    return "invalid-mutation-address-delta";
  case ConSanContractIssue::InvalidPerturbationBounds:
    return "invalid-perturbation-bounds";
  case ConSanContractIssue::ModeConflict:
    return "mode-conflict";
  case ConSanContractIssue::ConflictingRuntimeAssertions:
    return "conflicting-runtime-assertions";
  case ConSanContractIssue::MissingRuntimeBackend:
    return "missing-runtime-backend";
  case ConSanContractIssue::MissingVisibleMemory:
    return "missing-visible-memory";
  case ConSanContractIssue::MissingCoherentMemory:
    return "missing-coherent-memory";
  case ConSanContractIssue::MissingAtomicPublication:
    return "missing-atomic-publication";
  case ConSanContractIssue::InsufficientReportAllocation:
    return "insufficient-report-allocation";
  case ConSanContractIssue::MissingWorkgroupLdsLimit:
    return "missing-workgroup-lds-limit";
  case ConSanContractIssue::MissingExecutableBinding:
    return "missing-executable-binding";
  case ConSanContractIssue::MissingDispatchSegmentBinding:
    return "missing-dispatch-segment-binding";
  case ConSanContractIssue::InvalidResourceScope:
    return "invalid-resource-scope";
  case ConSanContractIssue::InvalidResourceAddress:
    return "invalid-resource-address";
  case ConSanContractIssue::InvalidResourceSize:
    return "invalid-resource-size";
  case ConSanContractIssue::Count:
    break;
  }
  return "invalid-contract-issue";
}

[[nodiscard]] constexpr ConSanContractIssue validate_consan_request(const ConSanRequest &request) {
  if (!request.flavor)
    return ConSanContractIssue::MissingFlavor;
  switch (*request.flavor) {
  case ConSanFlavor::None:
  case ConSanFlavor::SuperCollider:
  case ConSanFlavor::Moi:
    break;
  default:
    return ConSanContractIssue::InvalidMode;
  }
  switch (request.moi_engine) {
  case ConSanMoiEngine::RecordReplay:
  case ConSanMoiEngine::InlineShadow:
  case ConSanMoiEngine::Sampled:
    break;
  default:
    return ConSanContractIssue::InvalidMode;
  }
  if (request.moi_sample_stride == 0 || request.moi_runtime_sample_stride == 0 ||
      request.moi_runtime_sample_stride > (1u << 24u) ||
      (request.moi_runtime_sample_stride & (request.moi_runtime_sample_stride - 1u)) != 0) {
    return ConSanContractIssue::InvalidSampleStride;
  }
  if (request.moi_sample_offset >= request.moi_sample_stride ||
      request.moi_runtime_sample_offset >= request.moi_runtime_sample_stride) {
    return ConSanContractIssue::InvalidSampleOffset;
  }
  if (request.moi_dynamic_access_records && (*request.flavor != ConSanFlavor::Moi ||
                                             request.moi_engine != ConSanMoiEngine::RecordReplay)) {
    return ConSanContractIssue::ModeConflict;
  }
  if (request.moi_sampled_check &&
      (*request.flavor != ConSanFlavor::Moi || request.moi_engine != ConSanMoiEngine::Sampled)) {
    return ConSanContractIssue::ModeConflict;
  }
  return ConSanContractIssue::None;
}

[[nodiscard]] constexpr ConSanContractIssue
validate_transform_policy(const TransformPolicy &policy) {
  if (!consan_patched_image_growth_limit_bytes(policy.patched_image_growth_limit, 1u))
    return ConSanContractIssue::InvalidMode;
  return policy.max_patches == 0 ? ConSanContractIssue::InvalidPatchBudget
                                 : ConSanContractIssue::None;
}

/// Validate runtime policy independently of any code-object or device facts.
/// Process ceilings intentionally permit zero: that is a valid policy which
/// admits no corresponding live storage. Requiring installation while the
/// runtime path is disabled is the sole intrinsic conflict.
[[nodiscard]] constexpr ConSanContractIssue validate_runtime_policy(const RuntimePolicy &policy) {
  return !policy.enabled && (policy.fail_closed || policy.require_patch)
             ? ConSanContractIssue::ModeConflict
             : ConSanContractIssue::None;
}

[[nodiscard]] constexpr ConSanContractIssue
validate_consan_debug_overrides(const ConSanDebugOverrides &debug) {
  if ((debug.scratch_vgpr && *debug.scratch_vgpr > 255) ||
      (debug.moi_owner_vgpr && *debug.moi_owner_vgpr > 255) ||
      (debug.moi_epoch_vgpr && *debug.moi_epoch_vgpr > 255) ||
      (debug.moi_owner_sgpr && *debug.moi_owner_sgpr > 105) ||
      (debug.moi_exec_save_sgpr &&
       (*debug.moi_exec_save_sgpr > 104 || *debug.moi_exec_save_sgpr % 2 != 0))) {
    return ConSanContractIssue::InvalidDebugRegister;
  }
  if (debug.moi_require_diagnostics && debug.moi_forbid_diagnostics)
    return ConSanContractIssue::ConflictingRuntimeAssertions;
  return ConSanContractIssue::None;
}

[[nodiscard]] constexpr ConSanContractIssue
validate_mutation_request(const MutationRequest &mutation, const ConSanRequest &request) {
  if (mutation.fault_allow_destructive_incomplete_barrier_drop && !mutation.fault_drop_barrier) {
    return ConSanContractIssue::InvalidMutationDependency;
  }
  if ((mutation.fault_allow_completing_conditional_barrier_move ||
       mutation.fault_allow_destructive_divergent_barrier_move) &&
      (!mutation.fault_move_barrier ||
       mutation.fault_barrier_move_direction != ConSanBarrierMoveDirection::Earlier)) {
    return ConSanContractIssue::InvalidMutationDependency;
  }
  const auto invalid_delta = [](uint32_t delta) {
    return delta == 0 || delta % 4u != 0 || delta > 0x7fffffu;
  };
  if ((mutation.fault_atomic_wrong_address && invalid_delta(mutation.fault_atomic_address_delta)) ||
      (mutation.fault_ordinary_wrong_address &&
       invalid_delta(mutation.fault_ordinary_address_delta))) {
    return ConSanContractIssue::InvalidMutationAddressDelta;
  }
  if (mutation.fault_lds_wrong_address && !mutation.fault_lds_address_vgpr)
    return ConSanContractIssue::InvalidMutationDependency;
  if (mutation.sc_perturb_max == 0 || mutation.sc_perturb_max > 2 ||
      mutation.sc_perturb_sleep == 0 || mutation.sc_perturb_sleep > 15 ||
      mutation.sc_perturb_required_count > mutation.sc_perturb_max) {
    return ConSanContractIssue::InvalidPerturbationBounds;
  }
  if (mutation.sc_perturb_kind != ConSanPerturbationKind::None &&
      request.flavor != ConSanFlavor::SuperCollider) {
    return ConSanContractIssue::ModeConflict;
  }
  if (mutation.fault_reservation_timeout_ms == 0)
    return ConSanContractIssue::InvalidMutationDependency;
  if (mutation.fault_load_occurrence &&
      (mutation.fault_site_identity.empty() || !mutation.fault_require_exactly_one ||
       mutation.fault_dry_run)) {
    return ConSanContractIssue::InvalidMutationDependency;
  }
  return ConSanContractIssue::None;
}

[[nodiscard]] constexpr ConSanContractIssue
validate_runtime_capabilities(const RuntimeCapabilities &capabilities,
                              const RuntimeCapabilityRequirements &requirements = {}) {
  switch (capabilities.backend) {
  case ConSanRuntimeBackend::PhysicalHsa:
  case ConSanRuntimeBackend::RocJitsuSimulator:
    break;
  case ConSanRuntimeBackend::Unknown:
  case ConSanRuntimeBackend::Count:
  default:
    return ConSanContractIssue::MissingRuntimeBackend;
  }
  if (requirements.host_device_visible_memory && !capabilities.host_device_visible_memory)
    return ConSanContractIssue::MissingVisibleMemory;
  if (requirements.host_device_coherent_memory && !capabilities.host_device_coherent_memory)
    return ConSanContractIssue::MissingCoherentMemory;
  if (requirements.device_atomic_publication && !capabilities.device_atomic_publication)
    return ConSanContractIssue::MissingAtomicPublication;
  if (requirements.minimum_report_allocation_bytes &&
      (!capabilities.max_report_allocation_bytes ||
       *capabilities.max_report_allocation_bytes < *requirements.minimum_report_allocation_bytes)) {
    return ConSanContractIssue::InsufficientReportAllocation;
  }
  if (requirements.max_workgroup_lds_bytes && !capabilities.max_workgroup_lds_bytes)
    return ConSanContractIssue::MissingWorkgroupLdsLimit;
  if (requirements.executable_binding && !capabilities.executable_binding)
    return ConSanContractIssue::MissingExecutableBinding;
  if (requirements.dispatch_segment_binding && !capabilities.dispatch_segment_binding)
    return ConSanContractIssue::MissingDispatchSegmentBinding;
  return ConSanContractIssue::None;
}

[[nodiscard]] constexpr ConSanContractIssue
validate_bound_runtime_resources(const BoundRuntimeResources &resources) {
  switch (resources.scope) {
  case ConSanRuntimeResourceScope::Unbound:
  case ConSanRuntimeResourceScope::CodeObject:
  case ConSanRuntimeResourceScope::Executable:
  case ConSanRuntimeResourceScope::Dispatch:
    break;
  case ConSanRuntimeResourceScope::Count:
  default:
    return ConSanContractIssue::InvalidResourceScope;
  }
  if ((resources.report_buffer_address && *resources.report_buffer_address == 0) ||
      (resources.moi_report_buffer_address && *resources.moi_report_buffer_address == 0)) {
    return ConSanContractIssue::InvalidResourceAddress;
  }
  if (resources.bound() == (resources.scope == ConSanRuntimeResourceScope::Unbound))
    return ConSanContractIssue::InvalidResourceScope;
  if (resources.moi_report_buffer_address.has_value() != (resources.moi_report_buffer_size != 0))
    return ConSanContractIssue::InvalidResourceSize;
  return ConSanContractIssue::None;
}

/// Validate all configuration-source outputs in one deterministic order.
/// Runtime capability satisfaction is intentionally separate because those
/// facts are discovered only after the parser has selected an HSA/simulator
/// backend and agent.
[[nodiscard]] constexpr ConSanContractIssue
validate_consan_configuration(const ConSanRequest &request, const TransformPolicy &transform_policy,
                              const RuntimePolicy &runtime_policy,
                              const ConSanDebugOverrides &debug, const MutationRequest &mutation,
                              const BoundRuntimeResources &resources) {
  if (const auto issue = validate_consan_request(request); issue != ConSanContractIssue::None)
    return issue;
  if (const auto issue = validate_transform_policy(transform_policy);
      issue != ConSanContractIssue::None) {
    return issue;
  }
  if (const auto issue = validate_runtime_policy(runtime_policy);
      issue != ConSanContractIssue::None) {
    return issue;
  }
  if (const auto issue = validate_consan_debug_overrides(debug);
      issue != ConSanContractIssue::None) {
    return issue;
  }
  if (const auto issue = validate_mutation_request(mutation, request);
      issue != ConSanContractIssue::None) {
    return issue;
  }
  return validate_bound_runtime_resources(resources);
}

/// Temporary, one-way compatibility boundary from immutable Slice-2 inputs to
/// the current mutable lowering structure.
///
/// The adapter always returns a fresh `ConSanOptions`; it never accepts or
/// mutates a caller-owned legacy value.
class LegacyOptionsAdapter {
public:
  [[nodiscard]] static ConSanOptions
  adapt(const ConSanRequest &request, const TransformPolicy &transform_policy,
        const ConSanDebugOverrides &debug, const MutationRequest &mutation,
        const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources) {
    ConSanOptions options;
    static_cast<ConSanRequest &>(options) = request;
    options.flavor = request.flavor.value_or(ConSanFlavor::None);
    options.moi_inline_workgroup_shadow =
        request.flavor == ConSanFlavor::Moi && request.moi_engine == ConSanMoiEngine::InlineShadow;

    static_cast<TransformPolicy &>(options) = transform_policy;

    options.probe_nop = debug.probe_nop;
    options.probe_trampoline_nop = debug.probe_trampoline_nop;
    options.probe_endpgm = debug.probe_endpgm;
    options.probe_lds_endpgm = debug.probe_lds_endpgm;
    options.probe_flat_trap = debug.probe_flat_trap;
    options.abort_unmatched_barrier_wait = debug.abort_unmatched_barrier_wait;
    options.moi_partition_mask_debug = debug.moi_partition_mask_debug;
    options.force_vgpr_spill = debug.test_force_vgpr_spill;
    options.force_private_epoch = debug.test_force_private_epoch;
    options.test_kernel_name_filter = debug.test_kernel_name_filter;
    options.scratch_vgpr = debug.scratch_vgpr;
    options.moi_exec_save_sgpr = debug.moi_exec_save_sgpr;
    options.moi_owner_sgpr = debug.moi_owner_sgpr;
    options.moi_owner_vgpr = debug.moi_owner_vgpr;
    options.moi_epoch_vgpr = debug.moi_epoch_vgpr;

    options.fault_drop_barrier = mutation.fault_drop_barrier;
    options.fault_allow_destructive_incomplete_barrier_drop =
        mutation.fault_allow_destructive_incomplete_barrier_drop;
    options.fault_move_barrier = mutation.fault_move_barrier;
    options.collect_barrier_move_destinations = mutation.fault_move_barrier;
    options.fault_allow_completing_conditional_barrier_move =
        mutation.fault_allow_completing_conditional_barrier_move;
    options.fault_allow_destructive_divergent_barrier_move =
        mutation.fault_allow_destructive_divergent_barrier_move;
    options.fault_mutate_barrier_id_scope = mutation.fault_mutate_barrier_id_scope;
    options.fault_mutate_barrier_participants = mutation.fault_mutate_barrier_participants;
    options.fault_atomic_wrong_address = mutation.fault_atomic_wrong_address;
    options.fault_atomic_weaken_order = mutation.fault_atomic_weaken_order;
    options.fault_atomic_order_edge = mutation.fault_atomic_order_edge;
    options.fault_atomic_weaken_scope = mutation.fault_atomic_weaken_scope;
    options.fault_lds_wrong_address = mutation.fault_lds_wrong_address;
    options.fault_ordinary_wrong_address = mutation.fault_ordinary_wrong_address;
    options.fault_ordinary_weaken_order = mutation.fault_ordinary_weaken_order;
    options.fault_ordinary_weaken_scope = mutation.fault_ordinary_weaken_scope;
    options.fault_atomic_address_delta = mutation.fault_atomic_address_delta;
    options.fault_lds_address_vgpr = mutation.fault_lds_address_vgpr;
    options.fault_ordinary_address_delta = mutation.fault_ordinary_address_delta;
    options.fault_dry_run = mutation.fault_dry_run;
    options.fault_require_exactly_one = mutation.fault_require_exactly_one;
    options.fault_barrier_index = mutation.fault_barrier_index;
    options.fault_atomic_index = mutation.fault_atomic_index;
    options.fault_lds_index = mutation.fault_lds_index;
    options.fault_ordinary_index = mutation.fault_ordinary_index;
    options.fault_site_identity = mutation.fault_site_identity;
    options.fault_barrier_destination_identity = mutation.fault_barrier_destination_identity;
    options.fault_barrier_sequence_identity = mutation.fault_barrier_sequence_identity;
    options.fault_barrier_companion_site_identity = mutation.fault_barrier_companion_site_identity;
    options.fault_barrier_companion_sequence_identity =
        mutation.fault_barrier_companion_sequence_identity;
    options.fault_barrier_target_id = mutation.fault_barrier_target_id;
    options.fault_barrier_target_participant_count =
        mutation.fault_barrier_target_participant_count;
    options.fault_barrier_target_participant_mask = mutation.fault_barrier_target_participant_mask;
    options.fault_barrier_move_direction = mutation.fault_barrier_move_direction;
    options.sc_perturb_kind = mutation.sc_perturb_kind;
    options.sc_perturb_edge = mutation.sc_perturb_edge;
    options.sc_perturb_identity = mutation.sc_perturb_identity;
    options.sc_perturb_index = mutation.sc_perturb_index;
    options.sc_perturb_max = mutation.sc_perturb_max;
    options.sc_perturb_sleep = mutation.sc_perturb_sleep;
    options.sc_perturb_required_count = mutation.sc_perturb_required_count;

    options.report_buffer_address = resources.report_buffer_address;
    options.moi_report_buffer_address = resources.moi_report_buffer_address;
    options.moi_report_buffer_size = resources.moi_report_buffer_size;
    options.moi_report_layout = resources.moi_report_layout;
    options.moi_report_generation = resources.moi_report_generation;
    options.moi_report_dispatch_id = resources.moi_report_dispatch_id;
    options.moi_max_workgroup_lds_bytes = capabilities.max_workgroup_lds_bytes;
    return options;
  }
};

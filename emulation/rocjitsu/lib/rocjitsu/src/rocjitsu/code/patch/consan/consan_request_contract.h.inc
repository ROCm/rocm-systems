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
      (debug.requested_moi_owner_vgpr && *debug.requested_moi_owner_vgpr > 255) ||
      (debug.requested_moi_epoch_vgpr && *debug.requested_moi_epoch_vgpr > 255) ||
      (debug.requested_moi_owner_sgpr && *debug.requested_moi_owner_sgpr > 105) ||
      (debug.requested_moi_exec_save_sgpr && (*debug.requested_moi_exec_save_sgpr > 104 ||
                                              *debug.requested_moi_exec_save_sgpr % 2 != 0))) {
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

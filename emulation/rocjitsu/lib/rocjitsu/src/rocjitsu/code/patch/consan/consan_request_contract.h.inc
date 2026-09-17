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

/// Disable every live fault-transform switch while retaining exact selection,
/// proof, and perturbation data.
inline void clear_fault_mutations(MutationRequest &request) {
  request.fault_drop_barrier = false;
  request.fault_move_barrier = false;
  request.fault_mutate_barrier_id_scope = false;
  request.fault_mutate_barrier_participants = false;
  request.fault_atomic_wrong_address = false;
  request.fault_atomic_weaken_order = false;
  request.fault_atomic_weaken_scope = false;
  request.fault_lds_wrong_address = false;
  request.fault_ordinary_wrong_address = false;
  request.fault_ordinary_weaken_order = false;
  request.fault_ordinary_weaken_scope = false;
}

/// Return a copy suitable for pristine semantic inventory sizing.
[[nodiscard]] inline MutationRequest without_fault_mutations(const MutationRequest &request) {
  MutationRequest result = request;
  clear_fault_mutations(result);
  result.fault_require_exactly_one = false;
  return result;
}

/// Return whether semantic inventory must retain extended barrier pairs.
///
/// Extended pairs are needed by barrier mutations, unmatched-barrier abort
/// diagnostics, and ConSan barrier observation. This query deliberately uses
/// only typed semantic inputs: runtime bindings, target mechanics, and legacy
/// lowering options cannot change the inventory shape.
[[nodiscard]] constexpr bool requires_extended_barrier_pairs(const Request &request,
                                                             const DebugOverrides &debug,
                                                             const MutationRequest &mutation) {
  return mutation.fault_drop_barrier || mutation.fault_move_barrier ||
         mutation.fault_mutate_barrier_id_scope || mutation.fault_mutate_barrier_participants ||
         debug.abort_unmatched_barrier_wait || request.track_barriers;
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
enum class ContractIssue : uint8_t {
  /// The contract is valid and all requested runtime facts are present.
  None,
  /// Request construction ended without selecting a sanitizer mode.
  MissingMode,
  /// A typed mode or mode value lies outside its declared enum.
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
  /// A concrete ConSan address and its byte size disagree.
  InvalidResourceSize,
  /// One past the last declared issue. This is an iteration bound and never a
  /// validation result.
  Count,
};

/// Every validation result that may be returned by a contract validator. The
/// one-past-the-end bound is excluded so callers cannot mistake it for an
/// actionable issue.
inline constexpr auto kContractIssues = [] {
  using E = ContractIssue;
  return make_enum_vocabulary(
      "invalid-contract-issue", enum_entry(E::None, "none"),
      enum_entry(E::MissingMode, "missing-mode"), enum_entry(E::InvalidMode, "invalid-mode"),
      enum_entry(E::InvalidSampleStride, "invalid-sample-stride"),
      enum_entry(E::InvalidSampleOffset, "invalid-sample-offset"),
      enum_entry(E::InvalidPatchBudget, "invalid-patch-budget"),
      enum_entry(E::InvalidDebugRegister, "invalid-debug-register"),
      enum_entry(E::InvalidMutationDependency, "invalid-mutation-dependency"),
      enum_entry(E::InvalidMutationAddressDelta, "invalid-mutation-address-delta"),
      enum_entry(E::InvalidPerturbationBounds, "invalid-perturbation-bounds"),
      enum_entry(E::ModeConflict, "mode-conflict"),
      enum_entry(E::ConflictingRuntimeAssertions, "conflicting-runtime-assertions"),
      enum_entry(E::MissingRuntimeBackend, "missing-runtime-backend"),
      enum_entry(E::MissingVisibleMemory, "missing-visible-memory"),
      enum_entry(E::MissingCoherentMemory, "missing-coherent-memory"),
      enum_entry(E::MissingAtomicPublication, "missing-atomic-publication"),
      enum_entry(E::InsufficientReportAllocation, "insufficient-report-allocation"),
      enum_entry(E::MissingWorkgroupLdsLimit, "missing-workgroup-lds-limit"),
      enum_entry(E::MissingExecutableBinding, "missing-executable-binding"),
      enum_entry(E::MissingDispatchSegmentBinding, "missing-dispatch-segment-binding"),
      enum_entry(E::InvalidResourceScope, "invalid-resource-scope"),
      enum_entry(E::InvalidResourceAddress, "invalid-resource-address"),
      enum_entry(E::InvalidResourceSize, "invalid-resource-size"));
}();

static_assert(kContractIssues.size() == static_cast<size_t>(ContractIssue::Count));

[[nodiscard]] constexpr std::string_view contract_issue_name(ContractIssue issue) {
  return kContractIssues.name(issue);
}

[[nodiscard]] constexpr ContractIssue validate_request(const Request &request) {
  if (!request.mode)
    return ContractIssue::MissingMode;
  switch (*request.mode) {
  case Mode::None:
  case Mode::SuperCollider:
  case Mode::Default:
    break;
  default:
    return ContractIssue::InvalidMode;
  }
  switch (request.supercollider_evidence_mode) {
  case SuperColliderEvidenceMode::TrapOnly:
  case SuperColliderEvidenceMode::StickyMarker:
    break;
  default:
    return ContractIssue::InvalidMode;
  }
  if (request.sample_stride == 0 || request.runtime_sample_stride == 0 ||
      request.runtime_sample_stride > (1u << 24u) ||
      (request.runtime_sample_stride & (request.runtime_sample_stride - 1u)) != 0) {
    return ContractIssue::InvalidSampleStride;
  }
  if (request.sample_offset >= request.sample_stride ||
      request.runtime_sample_offset >= request.runtime_sample_stride) {
    return ContractIssue::InvalidSampleOffset;
  }
  if (request.cell_selection || request.watchpoint_banks != 0) {
    if (*request.mode != Mode::Default)
      return ContractIssue::ModeConflict;
    const auto cell = request.cell_selector();
    if (cell.stride == 0 || cell.stride > (1u << 24u) || (cell.stride & (cell.stride - 1u)) != 0)
      return ContractIssue::InvalidSampleStride;
    if (cell.offset >= cell.stride)
      return ContractIssue::InvalidSampleOffset;
    if (request.watchpoint_banks > 1024 ||
        (request.watchpoint_banks & (request.watchpoint_banks - 1u)) != 0)
      return ContractIssue::InvalidMode;
  }
  if (request.device_conflict_check && (*request.mode != Mode::Default)) {
    return ContractIssue::ModeConflict;
  }
  return ContractIssue::None;
}

[[nodiscard]] constexpr ContractIssue validate_transform_policy(const TransformPolicy &policy) {
  if (!patched_image_growth_limit_bytes(policy.patched_image_growth_limit, 1u))
    return ContractIssue::InvalidMode;
  return policy.max_patches == 0 ? ContractIssue::InvalidPatchBudget : ContractIssue::None;
}

/// Validate runtime policy independently of any code-object or device facts.
/// Process ceilings intentionally permit zero: that is a valid policy which
/// admits no corresponding live storage. Requiring installation while the
/// runtime path is disabled is the sole intrinsic conflict.
[[nodiscard]] constexpr ContractIssue validate_runtime_policy(const RuntimePolicy &policy) {
  return !policy.enabled && (policy.fail_closed || policy.require_patch)
             ? ContractIssue::ModeConflict
             : ContractIssue::None;
}

[[nodiscard]] constexpr ContractIssue validate_debug_overrides(const DebugOverrides &debug) {
  if ((debug.scratch_vgpr && *debug.scratch_vgpr > 255) ||
      (debug.requested_owner_vgpr && *debug.requested_owner_vgpr > 255) ||
      (debug.requested_epoch_vgpr && *debug.requested_epoch_vgpr > 255) ||
      (debug.requested_owner_sgpr && *debug.requested_owner_sgpr > 105) ||
      (debug.requested_exec_save_sgpr &&
       (*debug.requested_exec_save_sgpr > 104 || *debug.requested_exec_save_sgpr % 2 != 0))) {
    return ContractIssue::InvalidDebugRegister;
  }
  if (debug.require_diagnostics && debug.forbid_diagnostics)
    return ContractIssue::ConflictingRuntimeAssertions;
  return ContractIssue::None;
}

[[nodiscard]] constexpr ContractIssue validate_mutation_request(const MutationRequest &mutation,
                                                                const Request &request) {
  if (mutation.fault_allow_destructive_incomplete_barrier_drop && !mutation.fault_drop_barrier) {
    return ContractIssue::InvalidMutationDependency;
  }
  if ((mutation.fault_allow_completing_conditional_barrier_move ||
       mutation.fault_allow_destructive_divergent_barrier_move) &&
      (!mutation.fault_move_barrier ||
       mutation.fault_barrier_move_direction != BarrierMoveDirection::Earlier)) {
    return ContractIssue::InvalidMutationDependency;
  }
  const auto invalid_delta = [](uint32_t delta) {
    return delta == 0 || delta % 4u != 0 || delta > 0x7fffffu;
  };
  if ((mutation.fault_atomic_wrong_address && invalid_delta(mutation.fault_atomic_address_delta)) ||
      (mutation.fault_ordinary_wrong_address &&
       invalid_delta(mutation.fault_ordinary_address_delta))) {
    return ContractIssue::InvalidMutationAddressDelta;
  }
  if (mutation.fault_lds_wrong_address && !mutation.fault_lds_address_vgpr)
    return ContractIssue::InvalidMutationDependency;
  if (mutation.supercollider_perturb_max == 0 || mutation.supercollider_perturb_max > 2 ||
      mutation.supercollider_perturb_sleep == 0 || mutation.supercollider_perturb_sleep > 15 ||
      mutation.supercollider_perturb_required_count > mutation.supercollider_perturb_max) {
    return ContractIssue::InvalidPerturbationBounds;
  }
  if (mutation.supercollider_perturb_kind != SuperColliderPerturbationKind::None &&
      request.mode != Mode::SuperCollider) {
    return ContractIssue::ModeConflict;
  }
  if (mutation.fault_reservation_timeout_ms == 0)
    return ContractIssue::InvalidMutationDependency;
  if (mutation.fault_load_occurrence &&
      (mutation.fault_site_identity.empty() || !mutation.fault_require_exactly_one ||
       mutation.fault_dry_run)) {
    return ContractIssue::InvalidMutationDependency;
  }
  return ContractIssue::None;
}

[[nodiscard]] constexpr ContractIssue
validate_runtime_capabilities(const RuntimeCapabilities &capabilities,
                              const RuntimeCapabilityRequirements &requirements = {}) {
  switch (capabilities.backend) {
  case RuntimeBackend::PhysicalHsa:
  case RuntimeBackend::RocJitsuSimulator:
    break;
  case RuntimeBackend::Unknown:
  case RuntimeBackend::Count:
  default:
    return ContractIssue::MissingRuntimeBackend;
  }
  if (requirements.host_device_visible_memory && !capabilities.host_device_visible_memory)
    return ContractIssue::MissingVisibleMemory;
  if (requirements.host_device_coherent_memory && !capabilities.host_device_coherent_memory)
    return ContractIssue::MissingCoherentMemory;
  if (requirements.device_atomic_publication && !capabilities.device_atomic_publication)
    return ContractIssue::MissingAtomicPublication;
  if (requirements.minimum_report_allocation_bytes &&
      (!capabilities.max_report_allocation_bytes ||
       *capabilities.max_report_allocation_bytes < *requirements.minimum_report_allocation_bytes)) {
    return ContractIssue::InsufficientReportAllocation;
  }
  if (requirements.max_workgroup_lds_bytes && !capabilities.max_workgroup_lds_bytes)
    return ContractIssue::MissingWorkgroupLdsLimit;
  if (requirements.executable_binding && !capabilities.executable_binding)
    return ContractIssue::MissingExecutableBinding;
  if (requirements.dispatch_segment_binding && !capabilities.dispatch_segment_binding)
    return ContractIssue::MissingDispatchSegmentBinding;
  return ContractIssue::None;
}

[[nodiscard]] constexpr ContractIssue
validate_bound_runtime_resources(const BoundRuntimeResources &resources) {
  switch (resources.scope) {
  case RuntimeResourceScope::Unbound:
  case RuntimeResourceScope::CodeObject:
  case RuntimeResourceScope::Executable:
  case RuntimeResourceScope::Dispatch:
    break;
  case RuntimeResourceScope::Count:
  default:
    return ContractIssue::InvalidResourceScope;
  }
  if ((resources.supercollider_report_buffer_address &&
       *resources.supercollider_report_buffer_address == 0) ||
      (resources.report_buffer_address && *resources.report_buffer_address == 0)) {
    return ContractIssue::InvalidResourceAddress;
  }
  if (resources.bound() == (resources.scope == RuntimeResourceScope::Unbound))
    return ContractIssue::InvalidResourceScope;
  if (resources.report_buffer_address.has_value() != (resources.report_buffer_size != 0))
    return ContractIssue::InvalidResourceSize;
  return ContractIssue::None;
}

/// Validate all configuration-source outputs in one deterministic order.
/// Runtime capability satisfaction is intentionally separate because those
/// facts are discovered only after the parser has selected an HSA/simulator
/// backend and agent.
[[nodiscard]] constexpr ContractIssue
validate_configuration(const Request &request, const TransformPolicy &transform_policy,
                       const RuntimePolicy &runtime_policy, const DebugOverrides &debug,
                       const MutationRequest &mutation, const BoundRuntimeResources &resources) {
  if (const auto issue = validate_request(request); issue != ContractIssue::None)
    return issue;
  if (const auto issue = validate_transform_policy(transform_policy);
      issue != ContractIssue::None) {
    return issue;
  }
  if (const auto issue = validate_runtime_policy(runtime_policy); issue != ContractIssue::None) {
    return issue;
  }
  if (const auto issue = validate_debug_overrides(debug); issue != ContractIssue::None) {
    return issue;
  }
  if (const auto issue = validate_mutation_request(mutation, request);
      issue != ContractIssue::None) {
    return issue;
  }
  return validate_bound_runtime_resources(resources);
}

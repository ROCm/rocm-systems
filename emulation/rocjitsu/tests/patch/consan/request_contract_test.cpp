// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace rocjitsu::consan {
namespace {

template <typename T>
concept HasReportBufferAddress = requires(T value) { value.supercollider_report_buffer_address; };

template <typename T>
concept HasFaultDropBarrier = requires(T value) { value.fault_drop_barrier; };

template <typename T>
concept HasTestKernelFilter = requires(T value) { value.test_kernel_name_filter; };

template <typename T>
concept HasFailClosed = requires(T value) { value.fail_closed; };

static_assert(!HasReportBufferAddress<Request>);
static_assert(!HasFaultDropBarrier<Request>);
static_assert(!HasTestKernelFilter<Request>);
static_assert(!HasFailClosed<Request>);
static_assert(!HasFailClosed<Options>);
static_assert(!HasReportBufferAddress<TransformPolicy>);
static_assert(!HasFaultDropBarrier<RuntimePolicy>);
static_assert(!HasReportBufferAddress<MutationRequest>);
static_assert(!HasFaultDropBarrier<BoundRuntimeResources>);
static_assert(HasReportBufferAddress<BoundRuntimeResources>);
static_assert(HasFaultDropBarrier<MutationRequest>);
static_assert(HasTestKernelFilter<DebugOverrides>);
static_assert(HasFailClosed<RuntimePolicy>);

[[nodiscard]] Request valid_request() {
  Request request;
  request.mode = Mode::Default;
  return request;
}

[[nodiscard]] RuntimeCapabilities physical_runtime_capabilities() {
  RuntimeCapabilities capabilities;
  capabilities.backend = RuntimeBackend::PhysicalHsa;
  return capabilities;
}

TEST(ConSanRequestContractTest, DefaultsExposeOnlyConstructionSentinel) {
  const Request request;
  EXPECT_EQ(validate_request(request), ContractIssue::MissingMode);
  EXPECT_EQ(request.sample_stride, 1u);
  EXPECT_EQ(request.runtime_sample_stride, 1u);
  EXPECT_EQ(request.supercollider_delay_mode, SuperColliderDelayMode::Nop);
  EXPECT_EQ(request.supercollider_report_marker, 1u);
  EXPECT_EQ(request.supercollider_evidence_mode, SuperColliderEvidenceMode::StickyMarker);
  EXPECT_EQ(request.auto_report_buffer_size, 0u);

  Request selected = request;
  selected.mode = Mode::Default;
  EXPECT_EQ(validate_request(selected), ContractIssue::None);
  EXPECT_NE(selected, request);
  EXPECT_EQ(selected, selected);
}

TEST(ConSanRequestContractTest, AcceptsEverySupportedModeAndRejectsInvalidEnums) {
  for (Mode mode : {Mode::None, Mode::SuperCollider, Mode::Default}) {
    Request request;
    request.mode = mode;
    EXPECT_EQ(validate_request(request), ContractIssue::None);
  }
  EXPECT_EQ(validate_request(valid_request()), ContractIssue::None);
  for (SuperColliderEvidenceMode mode :
       {SuperColliderEvidenceMode::TrapOnly, SuperColliderEvidenceMode::StickyMarker}) {
    Request request;
    request.mode = Mode::SuperCollider;
    request.supercollider_evidence_mode = mode;
    EXPECT_EQ(validate_request(request), ContractIssue::None);
  }

  Request invalid_mode = valid_request();
  invalid_mode.mode = static_cast<Mode>(255);
  EXPECT_EQ(validate_request(invalid_mode), ContractIssue::InvalidMode);
  Request invalid_supercollider_evidence;
  invalid_supercollider_evidence.mode = Mode::SuperCollider;
  invalid_supercollider_evidence.supercollider_evidence_mode = SuperColliderEvidenceMode::Count;
  EXPECT_EQ(validate_request(invalid_supercollider_evidence), ContractIssue::InvalidMode);
}

TEST(ConSanRequestContractTest, IndependentSelectorsAndBankRequestsAreBounded) {
  auto request = valid_request();
  request.runtime_sample_stride = 256;
  request.runtime_sample_offset = 17;
  EXPECT_EQ(request.cell_selector(), (SampleSelector{256, 17}));
  request.cell_selection = SampleSelector{4, 3};
  EXPECT_EQ(request.cell_selector(), (SampleSelector{4, 3}));
  for (uint32_t banks : {0u, 1u, 2u, 4u, 8u, 16u, 256u, 1024u}) {
    request.watchpoint_banks = banks;
    EXPECT_EQ(validate_request(request), ContractIssue::None);
  }
  for (uint32_t banks : {3u, 2048u, UINT32_MAX}) {
    request.watchpoint_banks = banks;
    EXPECT_EQ(validate_request(request), ContractIssue::InvalidMode);
  }
  request.watchpoint_banks = 0;
  for (uint32_t stride : {0u, 3u, 1u << 25u}) {
    request.cell_selection = SampleSelector{stride, 0};
    EXPECT_EQ(validate_request(request), ContractIssue::InvalidSampleStride);
  }
  request.cell_selection = SampleSelector{4, 4};
  EXPECT_EQ(validate_request(request), ContractIssue::InvalidSampleOffset);
}

TEST(ConSanRequestContractTest, ValidatesStaticAndRuntimeSamplingBoundaries) {
  Request request = valid_request();
  request.sample_stride = 7;
  request.sample_offset = 6;
  request.runtime_sample_stride = 1u << 24u;
  request.runtime_sample_offset = (1u << 24u) - 1u;
  EXPECT_EQ(validate_request(request), ContractIssue::None);

  request.sample_stride = 0;
  EXPECT_EQ(validate_request(request), ContractIssue::InvalidSampleStride);
  request.sample_stride = 7;
  request.sample_offset = 7;
  EXPECT_EQ(validate_request(request), ContractIssue::InvalidSampleOffset);
  request.sample_offset = 0;
  request.runtime_sample_stride = 3;
  EXPECT_EQ(validate_request(request), ContractIssue::InvalidSampleStride);
  request.runtime_sample_stride = (1u << 24u) + (1u << 23u);
  EXPECT_EQ(validate_request(request), ContractIssue::InvalidSampleStride);
  request.runtime_sample_stride = 8;
  request.runtime_sample_offset = 8;
  EXPECT_EQ(validate_request(request), ContractIssue::InvalidSampleOffset);
}

TEST(TransformPolicyContractTest, DefaultsAndBothGrowthFormsAreValueSemantic) {
  const TransformPolicy defaults;
  EXPECT_EQ(validate_transform_policy(defaults), ContractIssue::None);
  EXPECT_EQ(defaults.patched_image_growth_limit.kind, PatchedImageGrowthLimitKind::AbsoluteBytes);
  EXPECT_EQ(defaults.max_patches, 1u);
  EXPECT_TRUE(defaults.max_patches_is_expert_limit);

  TransformPolicy percent = defaults;
  percent.patched_image_growth_limit = {
      .kind = PatchedImageGrowthLimitKind::InputPercent,
      .absolute_bytes = 0,
      .input_percent = 125,
  };
  EXPECT_EQ(validate_transform_policy(percent), ContractIssue::None);
  EXPECT_NE(percent, defaults);
  TransformPolicy copy = percent;
  EXPECT_EQ(copy, percent);

  percent.max_patches = 0;
  EXPECT_EQ(validate_transform_policy(percent), ContractIssue::InvalidPatchBudget);
  percent.max_patches = 1;
  percent.patched_image_growth_limit.kind = static_cast<PatchedImageGrowthLimitKind>(255);
  EXPECT_EQ(validate_transform_policy(percent), ContractIssue::InvalidMode);
}

TEST(RuntimePolicyContractTest, ActivationAndFailurePolicyAreIndependentOfZeroCeilings) {
  RuntimePolicy policy;
  EXPECT_EQ(validate_runtime_policy(policy), ContractIssue::None);
  policy.fail_closed = true;
  EXPECT_EQ(validate_runtime_policy(policy), ContractIssue::ModeConflict);
  policy.fail_closed = false;
  policy.require_patch = true;
  EXPECT_EQ(validate_runtime_policy(policy), ContractIssue::ModeConflict);
  policy.enabled = true;
  policy.process_concurrent_transform_limit_bytes = 0;
  policy.process_patched_image_limit_bytes = 0;
  policy.process_patched_image_growth_limit_bytes = 0;
  EXPECT_EQ(validate_runtime_policy(policy), ContractIssue::None);
  RuntimePolicy copy = policy;
  EXPECT_EQ(copy, policy);
  copy.require_patch = false;
  EXPECT_NE(copy, policy);
}

TEST(ConSanDebugOverridesContractTest, ValidatesRegisterEnvelopesAndAssertions) {
  DebugOverrides debug;
  EXPECT_EQ(validate_debug_overrides(debug), ContractIssue::None);
  debug.scratch_vgpr = 255;
  debug.requested_owner_vgpr = 255;
  debug.requested_epoch_vgpr = 255;
  debug.requested_owner_sgpr = 105;
  debug.requested_exec_save_sgpr = 104;
  EXPECT_EQ(validate_debug_overrides(debug), ContractIssue::None);

  debug.requested_exec_save_sgpr = 103;
  EXPECT_EQ(validate_debug_overrides(debug), ContractIssue::InvalidDebugRegister);
  debug.requested_exec_save_sgpr = 104;
  debug.requested_owner_sgpr = 106;
  EXPECT_EQ(validate_debug_overrides(debug), ContractIssue::InvalidDebugRegister);
  debug.requested_owner_sgpr.reset();
  debug.require_diagnostics = true;
  debug.forbid_diagnostics = true;
  EXPECT_EQ(validate_debug_overrides(debug), ContractIssue::ConflictingRuntimeAssertions);

  DebugOverrides copy = debug;
  EXPECT_EQ(copy, debug);
  copy.test_kernel_name_filter = "another-kernel";
  EXPECT_NE(copy, debug);
}

TEST(ConSanDebugOverridesContractTest, RejectsEveryOutOfEnvelopeRegisterKind) {
  constexpr std::array cases{
      std::pair{&DebugOverrides::scratch_vgpr, uint16_t{256}},
      std::pair{&DebugOverrides::requested_owner_vgpr, uint16_t{256}},
      std::pair{&DebugOverrides::requested_epoch_vgpr, uint16_t{256}},
      std::pair{&DebugOverrides::requested_owner_sgpr, uint16_t{106}},
      std::pair{&DebugOverrides::requested_exec_save_sgpr, uint16_t{106}},
  };
  for (const auto &[field, first_invalid] : cases) {
    DebugOverrides debug;
    debug.*field = first_invalid;
    EXPECT_EQ(validate_debug_overrides(debug), ContractIssue::InvalidDebugRegister);
  }
}

TEST(MutationRequestContractTest, EnabledCoversEveryMutationFamily) {
  const MutationRequest none;
  EXPECT_FALSE(none.has_mutation());
  EXPECT_FALSE(none.has_fault_mutation());
  EXPECT_EQ(none.fault_mutation_count(), 0u);

  MutationRequest request;
  request.fault_drop_barrier = true;
  EXPECT_TRUE(request.has_mutation());
  EXPECT_TRUE(request.has_fault_mutation());
  request = {};
  request.fault_atomic_weaken_order = true;
  EXPECT_TRUE(request.has_mutation());
  request = {};
  request.fault_ordinary_wrong_address = true;
  EXPECT_TRUE(request.has_mutation());
  request = {};
  request.supercollider_perturb_kind = SuperColliderPerturbationKind::Barrier;
  EXPECT_TRUE(request.has_mutation());
  EXPECT_FALSE(request.has_fault_mutation());
}

TEST(MutationRequestContractTest, EveryLiveFaultSwitchParticipatesInDerivedPredicates) {
  constexpr std::array fault_switches{
      &MutationRequest::fault_drop_barrier,
      &MutationRequest::fault_move_barrier,
      &MutationRequest::fault_mutate_barrier_id_scope,
      &MutationRequest::fault_mutate_barrier_participants,
      &MutationRequest::fault_atomic_wrong_address,
      &MutationRequest::fault_atomic_weaken_order,
      &MutationRequest::fault_atomic_weaken_scope,
      &MutationRequest::fault_lds_wrong_address,
      &MutationRequest::fault_ordinary_wrong_address,
      &MutationRequest::fault_ordinary_weaken_order,
      &MutationRequest::fault_ordinary_weaken_scope,
  };
  for (bool MutationRequest::*fault_switch : fault_switches) {
    MutationRequest request;
    request.*fault_switch = true;
    EXPECT_TRUE(request.has_fault_mutation());
    EXPECT_TRUE(request.has_mutation());
    EXPECT_EQ(request.fault_mutation_count(), 1u);
  }
}

TEST(MutationRequestContractTest, PristineInventoryProjectionDisablesOnlyLiveFaults) {
  MutationRequest request;
  request.fault_drop_barrier = true;
  request.fault_atomic_weaken_order = true;
  request.fault_ordinary_weaken_scope = true;
  request.fault_require_exactly_one = true;
  request.fault_site_identity = "site";
  request.fault_barrier_sequence_identity = "sequence";
  request.supercollider_perturb_kind = SuperColliderPerturbationKind::Barrier;
  request.supercollider_perturb_identity = "perturb";
  const MutationRequest original = request;

  const MutationRequest pristine = without_fault_mutations(request);
  EXPECT_EQ(request, original);
  EXPECT_NE(pristine, request);
  EXPECT_FALSE(pristine.has_fault_mutation());
  EXPECT_TRUE(pristine.has_mutation());
  EXPECT_FALSE(pristine.fault_require_exactly_one);
  EXPECT_EQ(pristine.fault_site_identity, "site");
  EXPECT_EQ(pristine.fault_barrier_sequence_identity, "sequence");
  EXPECT_EQ(pristine.supercollider_perturb_kind, SuperColliderPerturbationKind::Barrier);
  EXPECT_EQ(pristine.supercollider_perturb_identity, "perturb");
}

TEST(MutationRequestContractTest, PristineProjectionClearsEveryLiveFaultSwitch) {
  MutationRequest request;
  request.fault_drop_barrier = true;
  request.fault_move_barrier = true;
  request.fault_mutate_barrier_id_scope = true;
  request.fault_mutate_barrier_participants = true;
  request.fault_atomic_wrong_address = true;
  request.fault_atomic_weaken_order = true;
  request.fault_atomic_weaken_scope = true;
  request.fault_lds_wrong_address = true;
  request.fault_ordinary_wrong_address = true;
  request.fault_ordinary_weaken_order = true;
  request.fault_ordinary_weaken_scope = true;

  const MutationRequest pristine = without_fault_mutations(request);
  EXPECT_FALSE(pristine.has_fault_mutation());
  EXPECT_FALSE(pristine.has_mutation());
}

TEST(MutationRequestContractTest, ValidatesDependenciesAddressBoundsAndPerturbation) {
  const Request moi = valid_request();
  MutationRequest mutation;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::None);

  mutation.fault_allow_destructive_incomplete_barrier_drop = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationDependency);
  mutation.fault_drop_barrier = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::None);

  mutation = {};
  mutation.fault_allow_completing_conditional_barrier_move = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationDependency);
  mutation.fault_move_barrier = true;
  mutation.fault_barrier_move_direction = BarrierMoveDirection::Earlier;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::None);

  mutation = {};
  mutation.fault_allow_destructive_divergent_barrier_move = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationDependency);
  mutation.fault_move_barrier = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationDependency);
  mutation.fault_barrier_move_direction = BarrierMoveDirection::Earlier;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::None);

  mutation = {};
  mutation.fault_atomic_wrong_address = true;
  for (uint32_t bad_delta : {0u, 2u, 0x800000u}) {
    mutation.fault_atomic_address_delta = bad_delta;
    EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationAddressDelta);
  }
  mutation.fault_atomic_address_delta = 0x7ffffcu;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::None);

  mutation = {};
  mutation.fault_ordinary_wrong_address = true;
  mutation.fault_ordinary_address_delta = 2;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationAddressDelta);
  mutation.fault_ordinary_address_delta = 4;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::None);

  mutation = {};
  mutation.fault_lds_wrong_address = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationDependency);
  mutation.fault_lds_address_vgpr = 255;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::None);

  mutation = {};
  mutation.supercollider_perturb_max = 2;
  mutation.supercollider_perturb_sleep = 15;
  mutation.supercollider_perturb_required_count = 2;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::None);
  mutation.supercollider_perturb_required_count = 3;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidPerturbationBounds);

  mutation = {};
  mutation.supercollider_perturb_max = 0;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidPerturbationBounds);
  mutation.supercollider_perturb_max = 3;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidPerturbationBounds);
  mutation.supercollider_perturb_max = 1;
  mutation.supercollider_perturb_sleep = 0;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidPerturbationBounds);
  mutation.supercollider_perturb_sleep = 16;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidPerturbationBounds);
}

TEST(MutationRequestContractTest, RequiresSuperColliderAndValidProcessSelection) {
  const Request moi = valid_request();
  Request supercollider;
  supercollider.mode = Mode::SuperCollider;
  MutationRequest mutation;
  mutation.supercollider_perturb_kind = SuperColliderPerturbationKind::Atomic;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::ModeConflict);
  EXPECT_EQ(validate_mutation_request(mutation, supercollider), ContractIssue::None);

  mutation = {};
  mutation.fault_reservation_timeout_ms = 0;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationDependency);
  mutation.fault_reservation_timeout_ms = 1;
  mutation.fault_load_occurrence = 2;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationDependency);
  mutation.fault_site_identity = "lds:0x20";
  mutation.fault_require_exactly_one = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::None);
  mutation.fault_dry_run = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ContractIssue::InvalidMutationDependency);
}

TEST(RuntimeCapabilitiesContractTest, PhysicalAndSimulatorFixturesShareOneFactModel) {
  const RuntimeCapabilities physical{
      .backend = RuntimeBackend::PhysicalHsa,
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .max_report_allocation_bytes = 1u << 20u,
      .max_workgroup_lds_bytes = 163840u,
      .executable_binding = true,
      .dispatch_segment_binding = true,
  };
  RuntimeCapabilities simulator = physical;
  simulator.backend = RuntimeBackend::RocJitsuSimulator;
  simulator.max_workgroup_lds_bytes = 327680u;
  const RuntimeCapabilityRequirements all{
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .minimum_report_allocation_bytes = 1u << 20u,
      .max_workgroup_lds_bytes = true,
      .executable_binding = true,
      .dispatch_segment_binding = true,
  };
  EXPECT_EQ(validate_runtime_capabilities(physical, all), ContractIssue::None);
  EXPECT_EQ(validate_runtime_capabilities(simulator, all), ContractIssue::None);
  EXPECT_NE(simulator, physical);
  RuntimeCapabilities copy = simulator;
  EXPECT_EQ(copy, simulator);

  RuntimeCapabilityRequirements requirements_copy = all;
  EXPECT_EQ(requirements_copy, all);
  requirements_copy.dispatch_segment_binding = false;
  EXPECT_NE(requirements_copy, all);
}

TEST(RuntimeCapabilitiesContractTest, RejectsEachMissingRequiredFact) {
  RuntimeCapabilities capabilities{
      .backend = RuntimeBackend::PhysicalHsa,
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .max_report_allocation_bytes = 64,
      .max_workgroup_lds_bytes = 128,
      .executable_binding = true,
      .dispatch_segment_binding = true,
  };
  EXPECT_EQ(validate_runtime_capabilities({}, {}), ContractIssue::MissingRuntimeBackend);
  capabilities.backend = static_cast<RuntimeBackend>(255);
  EXPECT_EQ(validate_runtime_capabilities(capabilities, {}), ContractIssue::MissingRuntimeBackend);
  capabilities.backend = RuntimeBackend::PhysicalHsa;

  RuntimeCapabilityRequirements requirement;
  requirement.host_device_visible_memory = true;
  capabilities.host_device_visible_memory = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ContractIssue::MissingVisibleMemory);
  capabilities.host_device_visible_memory = true;
  requirement = {};
  requirement.host_device_coherent_memory = true;
  capabilities.host_device_coherent_memory = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ContractIssue::MissingCoherentMemory);
  capabilities.host_device_coherent_memory = true;
  requirement = {};
  requirement.device_atomic_publication = true;
  capabilities.device_atomic_publication = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ContractIssue::MissingAtomicPublication);
  capabilities.device_atomic_publication = true;
  requirement = {.minimum_report_allocation_bytes = 65};
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ContractIssue::InsufficientReportAllocation);
  capabilities.max_report_allocation_bytes.reset();
  requirement.minimum_report_allocation_bytes = 1;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ContractIssue::InsufficientReportAllocation);
  capabilities.max_report_allocation_bytes = 64;
  capabilities.max_workgroup_lds_bytes.reset();
  requirement = {};
  requirement.max_workgroup_lds_bytes = true;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ContractIssue::MissingWorkgroupLdsLimit);
  capabilities.max_workgroup_lds_bytes = 128;
  capabilities.executable_binding = false;
  requirement = {};
  requirement.executable_binding = true;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ContractIssue::MissingExecutableBinding);
  capabilities.executable_binding = true;
  capabilities.dispatch_segment_binding = false;
  requirement = {};
  requirement.dispatch_segment_binding = true;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ContractIssue::MissingDispatchSegmentBinding);
}

TEST(BoundRuntimeResourcesContractTest, DistinguishesUnboundAndConcreteLifetimes) {
  BoundRuntimeResources resources;
  EXPECT_FALSE(resources.bound());
  EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::None);

  resources.scope = RuntimeResourceScope::CodeObject;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::InvalidResourceScope);
  resources.supercollider_report_buffer_address = 0x1000;
  EXPECT_TRUE(resources.bound());
  EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::None);
  resources.supercollider_report_buffer_address = 0;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::InvalidResourceAddress);

  resources = {};
  resources.scope = RuntimeResourceScope::Executable;
  resources.report_buffer_address = 0x2000;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::InvalidResourceSize);
  resources.report_buffer_size = 4096;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::None);
  resources.report_buffer_address.reset();
  EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::InvalidResourceScope);

  resources = {};
  resources.scope = static_cast<RuntimeResourceScope>(255);
  resources.supercollider_report_buffer_address = 0x3000;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::InvalidResourceScope);
}

TEST(BoundRuntimeResourcesContractTest, AcceptsEveryConcreteResourceScope) {
  for (RuntimeResourceScope scope :
       {RuntimeResourceScope::CodeObject, RuntimeResourceScope::Executable,
        RuntimeResourceScope::Dispatch}) {
    BoundRuntimeResources resources;
    resources.scope = scope;
    resources.supercollider_report_buffer_address = 0x1000;
    EXPECT_TRUE(resources.bound());
    EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::None);
  }
}

TEST(ConSanRuntimeResourceScopeTest, EnumeratesAndNamesEveryDeclaredValue) {
  constexpr std::array expected_names{
      std::string_view{"unbound"},
      std::string_view{"code-object"},
      std::string_view{"executable"},
      std::string_view{"dispatch"},
  };
  static_assert(kRuntimeResourceScopes.size() == expected_names.size());

  std::unordered_set<std::string_view> unique_names;
  for (size_t i = 0; i < kRuntimeResourceScopes.size(); ++i) {
    EXPECT_EQ(static_cast<size_t>(kRuntimeResourceScopes[i]), i);
    const std::string_view name = runtime_resource_scope_name(kRuntimeResourceScopes[i]);
    EXPECT_EQ(name, expected_names[i]);
    EXPECT_TRUE(unique_names.insert(name).second) << name;
  }
  EXPECT_EQ(runtime_resource_scope_name(RuntimeResourceScope::Count),
            "invalid-runtime-resource-scope");
  EXPECT_EQ(runtime_resource_scope_name(static_cast<RuntimeResourceScope>(255)),
            "invalid-runtime-resource-scope");
}

TEST(BoundRuntimeResourcesContractTest, OwnsLayoutGenerationAndDispatchValueSemantics) {
  ReportBufferLayout layout;
  layout.watchpoint_capacity = 19;
  layout.required_bytes = 4096;
  BoundRuntimeResources resources;
  resources.scope = RuntimeResourceScope::Dispatch;
  resources.report_buffer_address = 0x4000;
  resources.report_buffer_size = 4096;
  resources.report_layout = layout;
  resources.report_generation = 7;
  resources.report_dispatch_id = 9;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ContractIssue::None);
  BoundRuntimeResources copy = resources;
  EXPECT_EQ(copy, resources);
  copy.report_layout->watchpoint_capacity = 20;
  EXPECT_NE(copy, resources);
}

TEST(ConSanConfigurationContractTest, ReturnsFirstOwnedFailureAndAcceptsCompleteValues) {
  Request request = valid_request();
  TransformPolicy transform;
  RuntimePolicy runtime;
  runtime.enabled = true;
  DebugOverrides debug;
  MutationRequest mutation;
  BoundRuntimeResources resources;
  EXPECT_EQ(validate_configuration(request, transform, runtime, debug, mutation, resources),
            ContractIssue::None);

  request.mode.reset();
  transform.max_patches = 0;
  runtime.enabled = false;
  runtime.fail_closed = true;
  debug.require_diagnostics = true;
  debug.forbid_diagnostics = true;
  mutation.fault_reservation_timeout_ms = 0;
  resources.scope = RuntimeResourceScope::CodeObject;
  EXPECT_EQ(validate_configuration(request, transform, runtime, debug, mutation, resources),
            ContractIssue::MissingMode);
  request.mode = Mode::Default;
  EXPECT_EQ(validate_configuration(request, transform, runtime, debug, mutation, resources),
            ContractIssue::InvalidPatchBudget);
  transform.max_patches = 1;
  EXPECT_EQ(validate_configuration(request, transform, runtime, debug, mutation, resources),
            ContractIssue::ModeConflict);
  runtime.enabled = true;
  EXPECT_EQ(validate_configuration(request, transform, runtime, debug, mutation, resources),
            ContractIssue::ConflictingRuntimeAssertions);
  debug.forbid_diagnostics = false;
  EXPECT_EQ(validate_configuration(request, transform, runtime, debug, mutation, resources),
            ContractIssue::InvalidMutationDependency);
  mutation.fault_reservation_timeout_ms = 1;
  EXPECT_EQ(validate_configuration(request, transform, runtime, debug, mutation, resources),
            ContractIssue::InvalidResourceScope);
  resources.scope = RuntimeResourceScope::Unbound;
  EXPECT_EQ(validate_configuration(request, transform, runtime, debug, mutation, resources),
            ContractIssue::None);
}

TEST(ConSanContractIssueTest, EveryValueHasAStableUniqueNameAndInvalidValuesFailClosed) {
  std::unordered_set<std::string_view> names;
  for (size_t i = 0; i < kContractIssues.size(); ++i) {
    const ContractIssue issue = kContractIssues[i];
    EXPECT_EQ(static_cast<size_t>(issue), i);
    const std::string_view name = contract_issue_name(issue);
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name, "invalid-contract-issue");
    EXPECT_TRUE(names.insert(name).second) << name;
  }
  EXPECT_EQ(contract_issue_name(ContractIssue::Count), "invalid-contract-issue");
  EXPECT_EQ(contract_issue_name(static_cast<ContractIssue>(255)), "invalid-contract-issue");
}

TEST(ConSanOptionsConstructionTest, PreservesRequestWithoutMutatingInput) {
  Request request = valid_request();
  request.owner_source = OwnerSource::HwId;
  request.flat_provenance_mode = FlatProvenanceMode::Strict;
  request.probe_lds_check_trap = true;
  request.probe_flat_check_trap = true;
  request.init_owner_epoch = true;
  request.track_barriers = true;
  request.track_atomics = true;
  request.sample_stride = 11;
  request.sample_offset = 7;
  request.runtime_sample_stride = 16;
  request.runtime_sample_offset = 9;
  request.supercollider_delay_mode = SuperColliderDelayMode::SleepVar;
  request.supercollider_delay_nops = 13;
  request.supercollider_delay_var_ssrc = 99;
  request.supercollider_report_marker = 17;
  const Request original = request;

  const Options options(request, TransformPolicy{}, DebugOverrides{}, MutationRequest{},
                        physical_runtime_capabilities(), BoundRuntimeResources{});
  EXPECT_EQ(request, original);
  EXPECT_EQ(options.mode, Mode::Default);
  EXPECT_EQ(options.owner_source, request.owner_source);
  EXPECT_EQ(options.flat_provenance_mode, request.flat_provenance_mode);
  EXPECT_TRUE(options.probe_lds_check_trap);
  EXPECT_TRUE(options.probe_flat_check_trap);
  EXPECT_TRUE(options.init_owner_epoch);
  EXPECT_TRUE(options.track_barriers);
  EXPECT_TRUE(options.track_atomics);
  EXPECT_EQ(options.sample_stride, 11u);
  EXPECT_EQ(options.sample_offset, 7u);
  EXPECT_EQ(options.runtime_sample_stride, 16u);
  EXPECT_EQ(options.runtime_sample_offset, 9u);
  EXPECT_EQ(options.supercollider_delay_mode, SuperColliderDelayMode::SleepVar);
  EXPECT_EQ(options.supercollider_delay_nops, 13u);
  EXPECT_EQ(options.supercollider_delay_var_ssrc, 99u);
  EXPECT_EQ(options.supercollider_report_marker, 17u);
}

TEST(ConSanOptionsConstructionTest, PreservesModeSpecificRequestControls) {
  Request sampled = valid_request();
  sampled.device_conflict_check = true;
  const Options options = Options(sampled, TransformPolicy{}, DebugOverrides{}, MutationRequest{},
                                  physical_runtime_capabilities(), BoundRuntimeResources{});
  EXPECT_TRUE(options.device_conflict_check);
}

TEST(ConSanOptionsConstructionTest, PreservesPolicyDebugAndRuntimeCapabilityFields) {
  TransformPolicy transform;
  transform.patched_image_growth_limit.absolute_bytes = 1234;
  transform.max_patches = 23;
  transform.max_patches_is_expert_limit = false;
  DebugOverrides debug;
  debug.abort_unmatched_barrier_wait = true;
  debug.test_force_vgpr_spill = true;
  debug.test_force_private_epoch = true;
  debug.test_kernel_name_filter = "kernel";
  debug.scratch_vgpr = 1;
  debug.requested_exec_save_sgpr = 2;
  debug.requested_owner_sgpr = 3;
  debug.requested_owner_vgpr = 4;
  debug.requested_epoch_vgpr = 5;
  debug.require_records = true;
  debug.require_diagnostics = true;
  debug.forbid_diagnostics = true;
  debug.forbid_overflow = true;
  RuntimeCapabilities capabilities;
  capabilities.backend = RuntimeBackend::RocJitsuSimulator;
  capabilities.host_device_visible_memory = true;
  capabilities.host_device_coherent_memory = true;
  capabilities.device_atomic_publication = true;
  capabilities.max_report_allocation_bytes = 65536;
  capabilities.max_workgroup_lds_bytes = 327680;
  capabilities.executable_binding = true;
  capabilities.dispatch_segment_binding = true;

  const Options options(valid_request(), transform, debug, MutationRequest{}, capabilities,
                        BoundRuntimeResources{});
  EXPECT_EQ(static_cast<const RuntimeCapabilities &>(options), capabilities);
  EXPECT_EQ(static_cast<const DebugOverrides &>(options), debug);
  EXPECT_EQ(options.patched_image_growth_limit.absolute_bytes, 1234u);
  EXPECT_EQ(options.max_patches, 23u);
  EXPECT_FALSE(options.max_patches_is_expert_limit);
  EXPECT_TRUE(options.abort_unmatched_barrier_wait);
  EXPECT_EQ(options.test_kernel_name_filter, "kernel");
  EXPECT_EQ(options.scratch_vgpr, 1);
  EXPECT_EQ(options.requested_exec_save_sgpr, 2);
  EXPECT_EQ(options.requested_owner_sgpr, 3);
  EXPECT_EQ(options.requested_owner_vgpr, 4);
  EXPECT_EQ(options.requested_epoch_vgpr, 5);
  EXPECT_EQ(options.max_workgroup_lds_bytes, 327680u);
}

TEST(ConSanOptionsConstructionTest, PreservesEveryMutationAndBoundResourceFamily) {
  MutationRequest mutation;
  mutation.fault_drop_barrier = true;
  mutation.fault_allow_destructive_incomplete_barrier_drop = true;
  mutation.fault_move_barrier = true;
  mutation.fault_allow_completing_conditional_barrier_move = true;
  mutation.fault_allow_destructive_divergent_barrier_move = true;
  mutation.fault_mutate_barrier_id_scope = true;
  mutation.fault_mutate_barrier_participants = true;
  mutation.fault_atomic_wrong_address = true;
  mutation.fault_atomic_weaken_order = true;
  mutation.fault_atomic_order_edge = AtomicOrderEdge::Acquire;
  mutation.fault_atomic_weaken_scope = true;
  mutation.fault_lds_wrong_address = true;
  mutation.fault_ordinary_wrong_address = true;
  mutation.fault_ordinary_weaken_order = true;
  mutation.fault_ordinary_weaken_scope = true;
  mutation.fault_atomic_address_delta = 8;
  mutation.fault_lds_address_vgpr = 9;
  mutation.fault_ordinary_address_delta = 12;
  mutation.fault_dry_run = true;
  mutation.fault_require_exactly_one = true;
  mutation.fault_reservation_timeout_ms = 13;
  mutation.fault_load_occurrence = 14;
  mutation.fault_barrier_index = 1;
  mutation.fault_atomic_index = 2;
  mutation.fault_lds_index = 3;
  mutation.fault_ordinary_index = 4;
  mutation.fault_site_identity = "site";
  mutation.fault_barrier_destination_identity = "destination";
  mutation.fault_barrier_sequence_identity = "sequence";
  mutation.fault_barrier_companion_site_identity = "companion-site";
  mutation.fault_barrier_companion_sequence_identity = "companion-sequence";
  mutation.fault_barrier_target_id = 5;
  mutation.fault_barrier_target_participant_count = 6;
  mutation.fault_barrier_target_participant_mask = 7;
  mutation.fault_barrier_move_direction = BarrierMoveDirection::Later;
  mutation.supercollider_perturb_kind = SuperColliderPerturbationKind::Atomic;
  mutation.supercollider_perturb_edge = SuperColliderPerturbationEdge::Acquire;
  mutation.supercollider_perturb_identity = "perturb";
  mutation.supercollider_perturb_index = 8;
  mutation.supercollider_perturb_max = 2;
  mutation.supercollider_perturb_sleep = 9;
  mutation.supercollider_perturb_required_count = 1;
  ReportBufferLayout layout;
  layout.required_bytes = 4096;
  BoundRuntimeResources resources{
      .scope = RuntimeResourceScope::Executable,
      .supercollider_report_buffer_address = 0x1000,
      .report_buffer_address = 0x2000,
      .report_buffer_size = 4096,
      .report_layout = layout,
      .report_generation = 10,
      .report_dispatch_id = 11,
  };

  const Options options(valid_request(), TransformPolicy{}, DebugOverrides{}, mutation,
                        physical_runtime_capabilities(), resources);
  EXPECT_EQ(static_cast<const MutationRequest &>(options), mutation);
  EXPECT_EQ(static_cast<const BoundRuntimeResources &>(options), resources);
}

TEST(ConSanOptionsConstructionTest, ProducesFreshValues) {
  const Request request = valid_request();
  Options first(request, TransformPolicy{}, DebugOverrides{}, MutationRequest{},
                physical_runtime_capabilities(), BoundRuntimeResources{});
  Options second(request, TransformPolicy{}, DebugOverrides{}, MutationRequest{},
                 physical_runtime_capabilities(), BoundRuntimeResources{});
  first.sample_stride = 99;
  EXPECT_EQ(second.sample_stride, 1u);
  EXPECT_EQ(request.sample_stride, 1u);
}

} // namespace
} // namespace rocjitsu::consan

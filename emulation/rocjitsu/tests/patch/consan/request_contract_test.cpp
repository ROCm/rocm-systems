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

namespace rocjitsu {
namespace {

template <typename T>
concept HasReportBufferAddress = requires(T value) { value.report_buffer_address; };

template <typename T>
concept HasFaultDropBarrier = requires(T value) { value.fault_drop_barrier; };

template <typename T>
concept HasTestKernelFilter = requires(T value) { value.test_kernel_name_filter; };

template <typename T>
concept HasFailClosed = requires(T value) { value.fail_closed; };

static_assert(!HasReportBufferAddress<ConSanRequest>);
static_assert(!HasFaultDropBarrier<ConSanRequest>);
static_assert(!HasTestKernelFilter<ConSanRequest>);
static_assert(!HasFailClosed<ConSanRequest>);
static_assert(!HasFailClosed<ConSanOptions>);
static_assert(!HasReportBufferAddress<TransformPolicy>);
static_assert(!HasFaultDropBarrier<RuntimePolicy>);
static_assert(!HasReportBufferAddress<MutationRequest>);
static_assert(!HasFaultDropBarrier<BoundRuntimeResources>);
static_assert(HasReportBufferAddress<BoundRuntimeResources>);
static_assert(HasFaultDropBarrier<MutationRequest>);
static_assert(HasTestKernelFilter<ConSanDebugOverrides>);
static_assert(HasFailClosed<RuntimePolicy>);

[[nodiscard]] ConSanRequest valid_moi_request(ConSanMoiEngine engine) {
  ConSanRequest request;
  request.flavor = ConSanFlavor::Moi;
  request.moi_engine = engine;
  return request;
}

[[nodiscard]] RuntimeCapabilities physical_runtime_capabilities() {
  RuntimeCapabilities capabilities;
  capabilities.backend = ConSanRuntimeBackend::PhysicalHsa;
  return capabilities;
}

TEST(ConSanRequestContractTest, DefaultsExposeOnlyConstructionSentinel) {
  const ConSanRequest request;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::MissingFlavor);
  EXPECT_EQ(request.moi_engine, ConSanMoiEngine::RecordReplay);
  EXPECT_EQ(request.moi_sample_stride, 1u);
  EXPECT_EQ(request.moi_runtime_sample_stride, 1u);
  EXPECT_EQ(request.delay_mode, ConSanDelayMode::Nop);
  EXPECT_EQ(request.report_marker, 1u);
  EXPECT_EQ(request.moi_auto_report_buffer_size, 0u);

  ConSanRequest selected = request;
  selected.flavor = ConSanFlavor::Moi;
  EXPECT_EQ(validate_consan_request(selected), ConSanContractIssue::None);
  EXPECT_NE(selected, request);
  EXPECT_EQ(selected, selected);
}

TEST(ConSanRequestContractTest, AcceptsEverySupportedModeAndRejectsInvalidEnums) {
  for (ConSanFlavor flavor : {ConSanFlavor::None, ConSanFlavor::SuperCollider, ConSanFlavor::Moi}) {
    ConSanRequest request;
    request.flavor = flavor;
    EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::None);
  }
  for (ConSanMoiEngine engine :
       {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::InlineShadow, ConSanMoiEngine::Sampled}) {
    EXPECT_EQ(validate_consan_request(valid_moi_request(engine)), ConSanContractIssue::None);
  }

  ConSanRequest invalid_flavor = valid_moi_request(ConSanMoiEngine::RecordReplay);
  invalid_flavor.flavor = static_cast<ConSanFlavor>(255);
  EXPECT_EQ(validate_consan_request(invalid_flavor), ConSanContractIssue::InvalidMode);
  ConSanRequest invalid_engine = valid_moi_request(ConSanMoiEngine::RecordReplay);
  invalid_engine.moi_engine = static_cast<ConSanMoiEngine>(255);
  EXPECT_EQ(validate_consan_request(invalid_engine), ConSanContractIssue::InvalidMode);
}

TEST(ConSanRequestContractTest, ValidatesStaticAndRuntimeSamplingBoundaries) {
  ConSanRequest request = valid_moi_request(ConSanMoiEngine::Sampled);
  request.moi_sample_stride = 7;
  request.moi_sample_offset = 6;
  request.moi_runtime_sample_stride = 1u << 24u;
  request.moi_runtime_sample_offset = (1u << 24u) - 1u;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::None);

  request.moi_sample_stride = 0;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::InvalidSampleStride);
  request.moi_sample_stride = 7;
  request.moi_sample_offset = 7;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::InvalidSampleOffset);
  request.moi_sample_offset = 0;
  request.moi_runtime_sample_stride = 3;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::InvalidSampleStride);
  request.moi_runtime_sample_stride = (1u << 24u) + (1u << 23u);
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::InvalidSampleStride);
  request.moi_runtime_sample_stride = 8;
  request.moi_runtime_sample_offset = 8;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::InvalidSampleOffset);
}

TEST(ConSanRequestContractTest, RejectsEngineSpecificFeaturesOnAnotherMode) {
  ConSanRequest request = valid_moi_request(ConSanMoiEngine::Sampled);
  request.moi_dynamic_access_records = true;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::ModeConflict);
  request.moi_dynamic_access_records = false;
  request.moi_sampled_check = true;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::None);

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::ModeConflict);
  request.moi_sampled_check = false;
  request.moi_dynamic_access_records = true;
  EXPECT_EQ(validate_consan_request(request), ConSanContractIssue::None);
}

TEST(TransformPolicyContractTest, DefaultsAndBothGrowthFormsAreValueSemantic) {
  const TransformPolicy defaults;
  EXPECT_EQ(validate_transform_policy(defaults), ConSanContractIssue::None);
  EXPECT_EQ(defaults.patched_image_growth_limit.kind,
            ConSanPatchedImageGrowthLimitKind::AbsoluteBytes);
  EXPECT_EQ(defaults.max_patches, 1u);
  EXPECT_TRUE(defaults.max_patches_is_expert_limit);

  TransformPolicy percent = defaults;
  percent.patched_image_growth_limit = {
      .kind = ConSanPatchedImageGrowthLimitKind::InputPercent,
      .absolute_bytes = 0,
      .input_percent = 125,
  };
  percent.sopp_relay_planning_work_limit = {.base = 9, .per_input = 3};
  EXPECT_EQ(validate_transform_policy(percent), ConSanContractIssue::None);
  EXPECT_NE(percent, defaults);
  TransformPolicy copy = percent;
  EXPECT_EQ(copy, percent);

  percent.max_patches = 0;
  EXPECT_EQ(validate_transform_policy(percent), ConSanContractIssue::InvalidPatchBudget);
  percent.max_patches = 1;
  percent.patched_image_growth_limit.kind = static_cast<ConSanPatchedImageGrowthLimitKind>(255);
  EXPECT_EQ(validate_transform_policy(percent), ConSanContractIssue::InvalidMode);
}

TEST(RuntimePolicyContractTest, ActivationAndFailurePolicyAreIndependentOfZeroCeilings) {
  RuntimePolicy policy;
  EXPECT_EQ(validate_runtime_policy(policy), ConSanContractIssue::None);
  policy.fail_closed = true;
  EXPECT_EQ(validate_runtime_policy(policy), ConSanContractIssue::ModeConflict);
  policy.fail_closed = false;
  policy.require_patch = true;
  EXPECT_EQ(validate_runtime_policy(policy), ConSanContractIssue::ModeConflict);
  policy.enabled = true;
  policy.process_concurrent_transform_limit_bytes = 0;
  policy.process_patched_image_limit_bytes = 0;
  policy.process_patched_image_growth_limit_bytes = 0;
  EXPECT_EQ(validate_runtime_policy(policy), ConSanContractIssue::None);
  RuntimePolicy copy = policy;
  EXPECT_EQ(copy, policy);
  copy.require_patch = false;
  EXPECT_NE(copy, policy);
}

TEST(ConSanDebugOverridesContractTest, ValidatesRegisterEnvelopesAndAssertions) {
  ConSanDebugOverrides debug;
  EXPECT_EQ(validate_consan_debug_overrides(debug), ConSanContractIssue::None);
  debug.scratch_vgpr = 255;
  debug.moi_owner_vgpr = 255;
  debug.moi_epoch_vgpr = 255;
  debug.moi_owner_sgpr = 105;
  debug.moi_exec_save_sgpr = 104;
  EXPECT_EQ(validate_consan_debug_overrides(debug), ConSanContractIssue::None);

  debug.moi_exec_save_sgpr = 103;
  EXPECT_EQ(validate_consan_debug_overrides(debug), ConSanContractIssue::InvalidDebugRegister);
  debug.moi_exec_save_sgpr = 104;
  debug.moi_owner_sgpr = 106;
  EXPECT_EQ(validate_consan_debug_overrides(debug), ConSanContractIssue::InvalidDebugRegister);
  debug.moi_owner_sgpr.reset();
  debug.moi_require_diagnostics = true;
  debug.moi_forbid_diagnostics = true;
  EXPECT_EQ(validate_consan_debug_overrides(debug),
            ConSanContractIssue::ConflictingRuntimeAssertions);

  ConSanDebugOverrides copy = debug;
  EXPECT_EQ(copy, debug);
  copy.test_kernel_name_filter = "another-kernel";
  EXPECT_NE(copy, debug);
}

TEST(ConSanDebugOverridesContractTest, RejectsEveryOutOfEnvelopeRegisterKind) {
  constexpr std::array cases{
      std::pair{&ConSanDebugOverrides::scratch_vgpr, uint16_t{256}},
      std::pair{&ConSanDebugOverrides::moi_owner_vgpr, uint16_t{256}},
      std::pair{&ConSanDebugOverrides::moi_epoch_vgpr, uint16_t{256}},
      std::pair{&ConSanDebugOverrides::moi_owner_sgpr, uint16_t{106}},
      std::pair{&ConSanDebugOverrides::moi_exec_save_sgpr, uint16_t{106}},
  };
  for (const auto &[field, first_invalid] : cases) {
    ConSanDebugOverrides debug;
    debug.*field = first_invalid;
    EXPECT_EQ(validate_consan_debug_overrides(debug), ConSanContractIssue::InvalidDebugRegister);
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
  request.sc_perturb_kind = ConSanPerturbationKind::Barrier;
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
  request.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  request.sc_perturb_identity = "perturb";
  const MutationRequest original = request;

  const MutationRequest pristine = without_consan_fault_mutations(request);
  EXPECT_EQ(request, original);
  EXPECT_NE(pristine, request);
  EXPECT_FALSE(pristine.has_fault_mutation());
  EXPECT_TRUE(pristine.has_mutation());
  EXPECT_FALSE(pristine.fault_require_exactly_one);
  EXPECT_EQ(pristine.fault_site_identity, "site");
  EXPECT_EQ(pristine.fault_barrier_sequence_identity, "sequence");
  EXPECT_EQ(pristine.sc_perturb_kind, ConSanPerturbationKind::Barrier);
  EXPECT_EQ(pristine.sc_perturb_identity, "perturb");
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

  const MutationRequest pristine = without_consan_fault_mutations(request);
  EXPECT_FALSE(pristine.has_fault_mutation());
  EXPECT_FALSE(pristine.has_mutation());
}

TEST(MutationRequestContractTest, ValidatesDependenciesAddressBoundsAndPerturbation) {
  const ConSanRequest moi = valid_moi_request(ConSanMoiEngine::RecordReplay);
  MutationRequest mutation;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::None);

  mutation.fault_allow_destructive_incomplete_barrier_drop = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidMutationDependency);
  mutation.fault_drop_barrier = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::None);

  mutation = {};
  mutation.fault_allow_completing_conditional_barrier_move = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidMutationDependency);
  mutation.fault_move_barrier = true;
  mutation.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::None);

  mutation = {};
  mutation.fault_allow_destructive_divergent_barrier_move = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidMutationDependency);
  mutation.fault_move_barrier = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidMutationDependency);
  mutation.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::None);

  mutation = {};
  mutation.fault_atomic_wrong_address = true;
  for (uint32_t bad_delta : {0u, 2u, 0x800000u}) {
    mutation.fault_atomic_address_delta = bad_delta;
    EXPECT_EQ(validate_mutation_request(mutation, moi),
              ConSanContractIssue::InvalidMutationAddressDelta);
  }
  mutation.fault_atomic_address_delta = 0x7ffffcu;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::None);

  mutation = {};
  mutation.fault_ordinary_wrong_address = true;
  mutation.fault_ordinary_address_delta = 2;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidMutationAddressDelta);
  mutation.fault_ordinary_address_delta = 4;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::None);

  mutation = {};
  mutation.fault_lds_wrong_address = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidMutationDependency);
  mutation.fault_lds_address_vgpr = 255;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::None);

  mutation = {};
  mutation.sc_perturb_max = 2;
  mutation.sc_perturb_sleep = 15;
  mutation.sc_perturb_required_count = 2;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::None);
  mutation.sc_perturb_required_count = 3;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidPerturbationBounds);

  mutation = {};
  mutation.sc_perturb_max = 0;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidPerturbationBounds);
  mutation.sc_perturb_max = 3;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidPerturbationBounds);
  mutation.sc_perturb_max = 1;
  mutation.sc_perturb_sleep = 0;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidPerturbationBounds);
  mutation.sc_perturb_sleep = 16;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidPerturbationBounds);
}

TEST(MutationRequestContractTest, RequiresSuperColliderAndValidProcessSelection) {
  const ConSanRequest moi = valid_moi_request(ConSanMoiEngine::RecordReplay);
  ConSanRequest supercollider;
  supercollider.flavor = ConSanFlavor::SuperCollider;
  MutationRequest mutation;
  mutation.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::ModeConflict);
  EXPECT_EQ(validate_mutation_request(mutation, supercollider), ConSanContractIssue::None);

  mutation = {};
  mutation.fault_reservation_timeout_ms = 0;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidMutationDependency);
  mutation.fault_reservation_timeout_ms = 1;
  mutation.fault_load_occurrence = 2;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidMutationDependency);
  mutation.fault_site_identity = "lds:0x20";
  mutation.fault_require_exactly_one = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi), ConSanContractIssue::None);
  mutation.fault_dry_run = true;
  EXPECT_EQ(validate_mutation_request(mutation, moi),
            ConSanContractIssue::InvalidMutationDependency);
}

TEST(RuntimeCapabilitiesContractTest, PhysicalAndSimulatorFixturesShareOneFactModel) {
  const RuntimeCapabilities physical{
      .backend = ConSanRuntimeBackend::PhysicalHsa,
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .max_report_allocation_bytes = 1u << 20u,
      .max_workgroup_lds_bytes = 163840u,
      .executable_binding = true,
      .dispatch_segment_binding = true,
  };
  RuntimeCapabilities simulator = physical;
  simulator.backend = ConSanRuntimeBackend::RocJitsuSimulator;
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
  EXPECT_EQ(validate_runtime_capabilities(physical, all), ConSanContractIssue::None);
  EXPECT_EQ(validate_runtime_capabilities(simulator, all), ConSanContractIssue::None);
  EXPECT_NE(simulator, physical);
  RuntimeCapabilities copy = simulator;
  EXPECT_EQ(copy, simulator);

  RuntimeCapabilityRequirements requirements_copy = all;
  EXPECT_EQ(requirements_copy, all);
  requirements_copy.dispatch_segment_binding = false;
  EXPECT_NE(requirements_copy, all);
}

TEST(ConSanRuntimeBackendTest, EnumeratesAndNamesEveryDeclaredValue) {
  constexpr std::array expected_names{
      std::string_view{"unknown"},
      std::string_view{"physical-hsa"},
      std::string_view{"rocjitsu-simulator"},
  };
  static_assert(kConSanRuntimeBackends.size() == expected_names.size());

  std::unordered_set<std::string_view> unique_names;
  for (size_t i = 0; i < kConSanRuntimeBackends.size(); ++i) {
    EXPECT_EQ(static_cast<size_t>(kConSanRuntimeBackends[i]), i);
    const std::string_view name = consan_runtime_backend_name(kConSanRuntimeBackends[i]);
    EXPECT_EQ(name, expected_names[i]);
    EXPECT_TRUE(unique_names.insert(name).second) << name;
  }
  EXPECT_EQ(consan_runtime_backend_name(ConSanRuntimeBackend::Count), "invalid-runtime-backend");
  EXPECT_EQ(consan_runtime_backend_name(static_cast<ConSanRuntimeBackend>(255)),
            "invalid-runtime-backend");
}

TEST(RuntimeCapabilitiesContractTest, RejectsEachMissingRequiredFact) {
  RuntimeCapabilities capabilities{
      .backend = ConSanRuntimeBackend::PhysicalHsa,
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .max_report_allocation_bytes = 64,
      .max_workgroup_lds_bytes = 128,
      .executable_binding = true,
      .dispatch_segment_binding = true,
  };
  EXPECT_EQ(validate_runtime_capabilities({}, {}), ConSanContractIssue::MissingRuntimeBackend);
  capabilities.backend = static_cast<ConSanRuntimeBackend>(255);
  EXPECT_EQ(validate_runtime_capabilities(capabilities, {}),
            ConSanContractIssue::MissingRuntimeBackend);
  capabilities.backend = ConSanRuntimeBackend::PhysicalHsa;

  RuntimeCapabilityRequirements requirement;
  requirement.host_device_visible_memory = true;
  capabilities.host_device_visible_memory = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ConSanContractIssue::MissingVisibleMemory);
  capabilities.host_device_visible_memory = true;
  requirement = {};
  requirement.host_device_coherent_memory = true;
  capabilities.host_device_coherent_memory = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ConSanContractIssue::MissingCoherentMemory);
  capabilities.host_device_coherent_memory = true;
  requirement = {};
  requirement.device_atomic_publication = true;
  capabilities.device_atomic_publication = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ConSanContractIssue::MissingAtomicPublication);
  capabilities.device_atomic_publication = true;
  requirement = {.minimum_report_allocation_bytes = 65};
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ConSanContractIssue::InsufficientReportAllocation);
  capabilities.max_report_allocation_bytes.reset();
  requirement.minimum_report_allocation_bytes = 1;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ConSanContractIssue::InsufficientReportAllocation);
  capabilities.max_report_allocation_bytes = 64;
  capabilities.max_workgroup_lds_bytes.reset();
  requirement = {};
  requirement.max_workgroup_lds_bytes = true;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ConSanContractIssue::MissingWorkgroupLdsLimit);
  capabilities.max_workgroup_lds_bytes = 128;
  capabilities.executable_binding = false;
  requirement = {};
  requirement.executable_binding = true;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ConSanContractIssue::MissingExecutableBinding);
  capabilities.executable_binding = true;
  capabilities.dispatch_segment_binding = false;
  requirement = {};
  requirement.dispatch_segment_binding = true;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirement),
            ConSanContractIssue::MissingDispatchSegmentBinding);
}

TEST(BoundRuntimeResourcesContractTest, DistinguishesUnboundAndConcreteLifetimes) {
  BoundRuntimeResources resources;
  EXPECT_FALSE(resources.bound());
  EXPECT_EQ(validate_bound_runtime_resources(resources), ConSanContractIssue::None);

  resources.scope = ConSanRuntimeResourceScope::CodeObject;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ConSanContractIssue::InvalidResourceScope);
  resources.report_buffer_address = 0x1000;
  EXPECT_TRUE(resources.bound());
  EXPECT_EQ(validate_bound_runtime_resources(resources), ConSanContractIssue::None);
  resources.report_buffer_address = 0;
  EXPECT_EQ(validate_bound_runtime_resources(resources),
            ConSanContractIssue::InvalidResourceAddress);

  resources = {};
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x2000;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ConSanContractIssue::InvalidResourceSize);
  resources.moi_report_buffer_size = 4096;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ConSanContractIssue::None);
  resources.moi_report_buffer_address.reset();
  EXPECT_EQ(validate_bound_runtime_resources(resources), ConSanContractIssue::InvalidResourceScope);

  resources = {};
  resources.scope = static_cast<ConSanRuntimeResourceScope>(255);
  resources.report_buffer_address = 0x3000;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ConSanContractIssue::InvalidResourceScope);
}

TEST(BoundRuntimeResourcesContractTest, AcceptsEveryConcreteResourceScope) {
  for (ConSanRuntimeResourceScope scope :
       {ConSanRuntimeResourceScope::CodeObject, ConSanRuntimeResourceScope::Executable,
        ConSanRuntimeResourceScope::Dispatch}) {
    BoundRuntimeResources resources;
    resources.scope = scope;
    resources.report_buffer_address = 0x1000;
    EXPECT_TRUE(resources.bound());
    EXPECT_EQ(validate_bound_runtime_resources(resources), ConSanContractIssue::None);
  }
}

TEST(ConSanRuntimeResourceScopeTest, EnumeratesAndNamesEveryDeclaredValue) {
  constexpr std::array expected_names{
      std::string_view{"unbound"},
      std::string_view{"code-object"},
      std::string_view{"executable"},
      std::string_view{"dispatch"},
  };
  static_assert(kConSanRuntimeResourceScopes.size() == expected_names.size());

  std::unordered_set<std::string_view> unique_names;
  for (size_t i = 0; i < kConSanRuntimeResourceScopes.size(); ++i) {
    EXPECT_EQ(static_cast<size_t>(kConSanRuntimeResourceScopes[i]), i);
    const std::string_view name =
        consan_runtime_resource_scope_name(kConSanRuntimeResourceScopes[i]);
    EXPECT_EQ(name, expected_names[i]);
    EXPECT_TRUE(unique_names.insert(name).second) << name;
  }
  EXPECT_EQ(consan_runtime_resource_scope_name(ConSanRuntimeResourceScope::Count),
            "invalid-runtime-resource-scope");
  EXPECT_EQ(consan_runtime_resource_scope_name(static_cast<ConSanRuntimeResourceScope>(255)),
            "invalid-runtime-resource-scope");
}

TEST(BoundRuntimeResourcesContractTest, OwnsLayoutGenerationAndDispatchValueSemantics) {
  ConSanMoiReportBufferLayout layout;
  layout.engine = ConSanMoiEngine::Sampled;
  layout.access_record_capacity = 19;
  layout.required_bytes = 4096;
  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Dispatch;
  resources.moi_report_buffer_address = 0x4000;
  resources.moi_report_buffer_size = 4096;
  resources.moi_report_layout = layout;
  resources.moi_report_generation = 7;
  resources.moi_report_dispatch_id = 9;
  EXPECT_EQ(validate_bound_runtime_resources(resources), ConSanContractIssue::None);
  BoundRuntimeResources copy = resources;
  EXPECT_EQ(copy, resources);
  copy.moi_report_layout->access_record_capacity = 20;
  EXPECT_NE(copy, resources);
}

TEST(ConSanConfigurationContractTest, ReturnsFirstOwnedFailureAndAcceptsCompleteValues) {
  ConSanRequest request = valid_moi_request(ConSanMoiEngine::RecordReplay);
  TransformPolicy transform;
  RuntimePolicy runtime;
  runtime.enabled = true;
  ConSanDebugOverrides debug;
  MutationRequest mutation;
  BoundRuntimeResources resources;
  EXPECT_EQ(validate_consan_configuration(request, transform, runtime, debug, mutation, resources),
            ConSanContractIssue::None);

  request.flavor.reset();
  transform.max_patches = 0;
  runtime.enabled = false;
  runtime.fail_closed = true;
  debug.moi_require_diagnostics = true;
  debug.moi_forbid_diagnostics = true;
  mutation.fault_reservation_timeout_ms = 0;
  resources.scope = ConSanRuntimeResourceScope::CodeObject;
  EXPECT_EQ(validate_consan_configuration(request, transform, runtime, debug, mutation, resources),
            ConSanContractIssue::MissingFlavor);
  request.flavor = ConSanFlavor::Moi;
  EXPECT_EQ(validate_consan_configuration(request, transform, runtime, debug, mutation, resources),
            ConSanContractIssue::InvalidPatchBudget);
  transform.max_patches = 1;
  EXPECT_EQ(validate_consan_configuration(request, transform, runtime, debug, mutation, resources),
            ConSanContractIssue::ModeConflict);
  runtime.enabled = true;
  EXPECT_EQ(validate_consan_configuration(request, transform, runtime, debug, mutation, resources),
            ConSanContractIssue::ConflictingRuntimeAssertions);
  debug.moi_forbid_diagnostics = false;
  EXPECT_EQ(validate_consan_configuration(request, transform, runtime, debug, mutation, resources),
            ConSanContractIssue::InvalidMutationDependency);
  mutation.fault_reservation_timeout_ms = 1;
  EXPECT_EQ(validate_consan_configuration(request, transform, runtime, debug, mutation, resources),
            ConSanContractIssue::InvalidResourceScope);
  resources.scope = ConSanRuntimeResourceScope::Unbound;
  EXPECT_EQ(validate_consan_configuration(request, transform, runtime, debug, mutation, resources),
            ConSanContractIssue::None);
}

TEST(ConSanContractIssueTest, EveryValueHasAStableUniqueNameAndInvalidValuesFailClosed) {
  std::unordered_set<std::string_view> names;
  for (size_t i = 0; i < kConSanContractIssues.size(); ++i) {
    const ConSanContractIssue issue = kConSanContractIssues[i];
    EXPECT_EQ(static_cast<size_t>(issue), i);
    const std::string_view name = consan_contract_issue_name(issue);
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name, "invalid-contract-issue");
    EXPECT_TRUE(names.insert(name).second) << name;
  }
  EXPECT_EQ(consan_contract_issue_name(ConSanContractIssue::Count), "invalid-contract-issue");
  EXPECT_EQ(consan_contract_issue_name(static_cast<ConSanContractIssue>(255)),
            "invalid-contract-issue");
}

TEST(ConSanOptionsConstructionTest, PreservesRequestWithoutMutatingInput) {
  ConSanRequest request = valid_moi_request(ConSanMoiEngine::InlineShadow);
  request.moi_owner_source = ConSanMoiOwnerSource::HwId;
  request.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  request.probe_lds_check_trap = true;
  request.probe_flat_check_trap = true;
  request.moi_init_owner_epoch = true;
  request.moi_track_barriers = true;
  request.moi_track_atomics = true;
  request.moi_dynamic_access_records = false;
  request.moi_sample_stride = 11;
  request.moi_sample_offset = 7;
  request.moi_runtime_sample_stride = 16;
  request.moi_runtime_sample_offset = 9;
  request.delay_mode = ConSanDelayMode::SleepVar;
  request.delay_nops = 13;
  request.delay_var_ssrc = 99;
  request.report_marker = 17;
  const ConSanRequest original = request;

  const ConSanOptions options(request, TransformPolicy{}, ConSanDebugOverrides{}, MutationRequest{},
                              physical_runtime_capabilities(), BoundRuntimeResources{});
  EXPECT_EQ(request, original);
  EXPECT_EQ(options.flavor, ConSanFlavor::Moi);
  EXPECT_EQ(options.moi_engine, ConSanMoiEngine::InlineShadow);
  EXPECT_EQ(options.moi_owner_source, request.moi_owner_source);
  EXPECT_EQ(options.flat_provenance_mode, request.flat_provenance_mode);
  EXPECT_TRUE(options.probe_lds_check_trap);
  EXPECT_TRUE(options.probe_flat_check_trap);
  EXPECT_TRUE(options.moi_init_owner_epoch);
  EXPECT_TRUE(options.moi_track_barriers);
  EXPECT_TRUE(options.moi_track_atomics);
  EXPECT_EQ(options.moi_sample_stride, 11u);
  EXPECT_EQ(options.moi_sample_offset, 7u);
  EXPECT_EQ(options.moi_runtime_sample_stride, 16u);
  EXPECT_EQ(options.moi_runtime_sample_offset, 9u);
  EXPECT_EQ(options.delay_mode, ConSanDelayMode::SleepVar);
  EXPECT_EQ(options.delay_nops, 13u);
  EXPECT_EQ(options.delay_var_ssrc, 99u);
  EXPECT_EQ(options.report_marker, 17u);
}

TEST(ConSanOptionsConstructionTest, PreservesEngineSpecificRequestControls) {
  ConSanRequest record_replay = valid_moi_request(ConSanMoiEngine::RecordReplay);
  record_replay.moi_dynamic_access_records = true;
  ConSanOptions options(record_replay, TransformPolicy{}, ConSanDebugOverrides{}, MutationRequest{},
                        physical_runtime_capabilities(), BoundRuntimeResources{});
  EXPECT_TRUE(options.moi_dynamic_access_records);
  EXPECT_FALSE(options.moi_sampled_check);

  ConSanRequest sampled = valid_moi_request(ConSanMoiEngine::Sampled);
  sampled.moi_sampled_check = true;
  options = ConSanOptions(sampled, TransformPolicy{}, ConSanDebugOverrides{}, MutationRequest{},
                          physical_runtime_capabilities(), BoundRuntimeResources{});
  EXPECT_FALSE(options.moi_dynamic_access_records);
  EXPECT_TRUE(options.moi_sampled_check);
}

TEST(ConSanOptionsConstructionTest, PreservesPolicyDebugAndRuntimeCapabilityFields) {
  TransformPolicy transform;
  transform.patched_image_growth_limit.absolute_bytes = 1234;
  transform.max_patches = 23;
  transform.max_patches_is_expert_limit = false;
  transform.sopp_relay_planning_work_limit = {1, 2};
  transform.direct_reservoir_planning_work_limit = {3, 4};
  transform.lds_relay_layout_planning_work_limit = {5, 6};
  transform.lds_convergence_planning_work_limit = {7, 8};
  ConSanDebugOverrides debug;
  debug.probe_nop = true;
  debug.probe_trampoline_nop = true;
  debug.probe_endpgm = true;
  debug.probe_lds_endpgm = true;
  debug.probe_flat_trap = true;
  debug.abort_unmatched_barrier_wait = true;
  debug.moi_partition_mask_debug = true;
  debug.test_force_vgpr_spill = true;
  debug.test_force_private_epoch = true;
  debug.test_seed_inline_exact_odd = true;
  debug.test_kernel_name_filter = "kernel";
  debug.scratch_vgpr = 1;
  debug.moi_exec_save_sgpr = 2;
  debug.moi_owner_sgpr = 3;
  debug.moi_owner_vgpr = 4;
  debug.moi_epoch_vgpr = 5;
  debug.moi_require_records = true;
  debug.moi_require_diagnostics = true;
  debug.moi_forbid_diagnostics = true;
  debug.moi_require_replay_conflict = true;
  debug.moi_forbid_overflow = true;
  RuntimeCapabilities capabilities;
  capabilities.backend = ConSanRuntimeBackend::RocJitsuSimulator;
  capabilities.host_device_visible_memory = true;
  capabilities.host_device_coherent_memory = true;
  capabilities.device_atomic_publication = true;
  capabilities.max_report_allocation_bytes = 65536;
  capabilities.max_workgroup_lds_bytes = 327680;
  capabilities.executable_binding = true;
  capabilities.dispatch_segment_binding = true;

  const ConSanOptions options(valid_moi_request(ConSanMoiEngine::RecordReplay), transform, debug,
                              MutationRequest{}, capabilities, BoundRuntimeResources{});
  EXPECT_EQ(static_cast<const RuntimeCapabilities &>(options), capabilities);
  EXPECT_EQ(static_cast<const ConSanDebugOverrides &>(options), debug);
  EXPECT_EQ(options.patched_image_growth_limit.absolute_bytes, 1234u);
  EXPECT_EQ(options.max_patches, 23u);
  EXPECT_FALSE(options.max_patches_is_expert_limit);
  EXPECT_EQ(options.sopp_relay_planning_work_limit.base, 1u);
  EXPECT_EQ(options.sopp_relay_planning_work_limit.per_input, 2u);
  EXPECT_EQ(options.direct_reservoir_planning_work_limit.base, 3u);
  EXPECT_EQ(options.direct_reservoir_planning_work_limit.per_input, 4u);
  EXPECT_EQ(options.lds_relay_layout_planning_work_limit.base, 5u);
  EXPECT_EQ(options.lds_relay_layout_planning_work_limit.per_input, 6u);
  EXPECT_EQ(options.lds_convergence_planning_work_limit.base, 7u);
  EXPECT_EQ(options.lds_convergence_planning_work_limit.per_input, 8u);
  EXPECT_TRUE(options.probe_nop);
  EXPECT_TRUE(options.probe_trampoline_nop);
  EXPECT_TRUE(options.probe_endpgm);
  EXPECT_TRUE(options.probe_lds_endpgm);
  EXPECT_TRUE(options.probe_flat_trap);
  EXPECT_TRUE(options.abort_unmatched_barrier_wait);
  EXPECT_TRUE(options.moi_partition_mask_debug);
  EXPECT_EQ(options.test_kernel_name_filter, "kernel");
  EXPECT_EQ(options.scratch_vgpr, 1);
  EXPECT_EQ(options.moi_exec_save_sgpr, 2);
  EXPECT_EQ(options.moi_owner_sgpr, 3);
  EXPECT_EQ(options.moi_owner_vgpr, 4);
  EXPECT_EQ(options.moi_epoch_vgpr, 5);
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
  mutation.fault_atomic_order_edge = ConSanAtomicOrderEdge::Acquire;
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
  mutation.fault_barrier_move_direction = ConSanBarrierMoveDirection::Later;
  mutation.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  mutation.sc_perturb_edge = ConSanPerturbationEdge::Acquire;
  mutation.sc_perturb_identity = "perturb";
  mutation.sc_perturb_index = 8;
  mutation.sc_perturb_max = 2;
  mutation.sc_perturb_sleep = 9;
  mutation.sc_perturb_required_count = 1;
  ConSanMoiReportBufferLayout layout;
  layout.required_bytes = 4096;
  BoundRuntimeResources resources{
      .scope = ConSanRuntimeResourceScope::Executable,
      .report_buffer_address = 0x1000,
      .moi_report_buffer_address = 0x2000,
      .moi_report_buffer_size = 4096,
      .moi_report_layout = layout,
      .moi_report_generation = 10,
      .moi_report_dispatch_id = 11,
  };

  const ConSanOptions options(valid_moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
                              ConSanDebugOverrides{}, mutation, physical_runtime_capabilities(),
                              resources);
  EXPECT_EQ(static_cast<const MutationRequest &>(options), mutation);
  EXPECT_EQ(static_cast<const BoundRuntimeResources &>(options), resources);
  EXPECT_TRUE(options.collect_barrier_move_destinations);
}

TEST(ConSanOptionsConstructionTest, ProducesFreshValues) {
  const ConSanRequest request = valid_moi_request(ConSanMoiEngine::RecordReplay);
  ConSanOptions first(request, TransformPolicy{}, ConSanDebugOverrides{}, MutationRequest{},
                      physical_runtime_capabilities(), BoundRuntimeResources{});
  ConSanOptions second(request, TransformPolicy{}, ConSanDebugOverrides{}, MutationRequest{},
                       physical_runtime_capabilities(), BoundRuntimeResources{});
  first.moi_sample_stride = 99;
  EXPECT_EQ(second.moi_sample_stride, 1u);
  EXPECT_EQ(request.moi_sample_stride, 1u);
}

} // namespace
} // namespace rocjitsu

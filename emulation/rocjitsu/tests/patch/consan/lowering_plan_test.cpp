// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_access_target.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"

namespace rocjitsu::consan {
namespace {

using detail::append_materialize_direct_to_lds_address;
using detail::ObjectFacts;
using detail::operand_overlap_spill;
using detail::PersistentStateFacts;
using detail::plan_object;
using detail::plan_persistent_state_demand;
using detail::plan_special_state;
using detail::requires_dispatch_identity;
using detail::resolve_access_resource_facts;
using detail::scalar_preservation_state;

TEST(ConSanLoweringPlan, AccessResourceFactsNormalizePointAndTargetBeforeModePolicy) {
  OperatingPoint point;
  point.exec_save_sgpr = 10u;
  point.initialize_owner_epoch = true;
  point.set_owner_epoch_vgprs(12u, 13u);
  point.persistent_sgprs.set_owner_epoch(20u, 21u);
  point.dynamic_stack_spill = true;

  ProgramSite candidate_site;
  candidate_site.lowering.form.emplace();
  candidate_site.lowering.form->kind = AccessLoweringFormKind::NativeTwoRange;
  candidate_site.lowering.form->encoded_offset_scale_bytes = 16u;
  candidate_site.lowering.form->address_vgpr = 4u;
  candidate_site.lowering.form->address_vgpr_count = 1u;
  candidate_site.lowering.form->destination_vgpr = 4u;
  candidate_site.lowering.form->destination_register_count = 1u;
  Candidate candidate(candidate_site);
  candidate.incoming_vgpr_bank_mode = 1u;

  const auto gfx1250 = resolve_access_resource_facts(point, candidate, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(gfx1250.address_scratch_vgpr_count, 1u);
  EXPECT_EQ(gfx1250.two_address_replay_vgpr_count, 1u);
  EXPECT_GT(gfx1250.dynamic_stack_reservoir_vgpr_count, 0u);
  EXPECT_TRUE(gfx1250.has_exec_save);
  EXPECT_TRUE(gfx1250.initialize_owner_epoch);
  EXPECT_TRUE(gfx1250.has_persistent_owner_vgpr);
  EXPECT_FALSE(gfx1250.uses_private_epoch);
  EXPECT_TRUE(gfx1250.has_complete_persistent_sgprs);
  EXPECT_TRUE(gfx1250.target_available);
  EXPECT_FALSE(gfx1250.supports_native_lds_spill_recovery);
  EXPECT_FALSE(gfx1250.supports_clobbered_address_spill_reload);
  EXPECT_TRUE(gfx1250.guest_replay_requires_disjoint_address_scratch);

  const auto gfx1201 = resolve_access_resource_facts(point, candidate, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ(gfx1201.two_address_replay_vgpr_count, 0u);
  EXPECT_GT(gfx1201.dynamic_stack_reservoir_vgpr_count, 0u);
  EXPECT_TRUE(gfx1201.supports_native_lds_spill_recovery);
  EXPECT_FALSE(gfx1201.supports_clobbered_address_spill_reload);
  EXPECT_FALSE(gfx1201.guest_replay_requires_disjoint_address_scratch);

  candidate_site.lowering.form->kind = AccessLoweringFormKind::NativeSingleRange;
  candidate_site.lowering.form->encoded_offset_scale_bytes = 0u;
  candidate.incoming_vgpr_bank_mode.reset();
  const auto gfx950 = resolve_access_resource_facts(point, candidate, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_EQ(gfx950.two_address_replay_vgpr_count, 0u);
  EXPECT_EQ(gfx950.dynamic_stack_reservoir_vgpr_count, 0u);
  EXPECT_TRUE(gfx950.supports_native_lds_spill_recovery);
  EXPECT_TRUE(gfx950.supports_clobbered_address_spill_reload);

  point.persistent_sgprs.reset_owner_epoch();
  point.automatic_private_epoch = true;
  const auto private_epoch =
      resolve_access_resource_facts(point, candidate, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_TRUE(private_epoch.uses_private_epoch);
  EXPECT_FALSE(private_epoch.has_complete_persistent_sgprs);
}

TEST(ConSanLoweringPlan, DirectToLdsAddressMaterializationUsesNormalizedForm) {
  ProgramSite candidate_site;
  candidate_site.lowering.form.emplace();
  candidate_site.lowering.form->kind = AccessLoweringFormKind::DirectToLdsExplicitAddress;
  candidate_site.lowering.form->address_vgpr = 7u;
  Candidate candidate(candidate_site);
  std::vector<uint32_t> words;
  EXPECT_TRUE(append_materialize_direct_to_lds_address(words, candidate.site(), 20u, 21u, 30u,
                                                       ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_FALSE(words.empty());

  candidate_site.lowering.form->kind = AccessLoweringFormKind::DirectToLdsLaneAddressed;
  candidate_site.lowering.form->address_vgpr.reset();
  candidate_site.lowering.form->element_width_bits = 32u;
  words.clear();
  EXPECT_TRUE(append_materialize_direct_to_lds_address(words, candidate.site(), 20u, 21u, 30u,
                                                       ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(words.empty());

  candidate_site.lowering.form->kind = AccessLoweringFormKind::NativeSingleRange;
  EXPECT_FALSE(append_materialize_direct_to_lds_address(words, candidate.site(), 20u, 21u, 30u,
                                                        ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(ConSanLoweringPlan, ScalarPreservationStateExcludesUnrelatedPlacement) {
  OperatingPoint point;
  point.exec_save_sgpr = 20u;
  point.automatic_scalar_spill_layout = ScalarSpillLayout::Compact;
  point.scalar_spill_setup = ScalarSpillSetup{
      .temporaries = ScalarSpillTemporaries{40u, 42u},
  };
  point.branch_only_spill = BranchOnlyScalarSpill{};

  const auto preservation_state = scalar_preservation_state(point);
  EXPECT_EQ(preservation_state.exec_save_sgpr, 20u);
  EXPECT_TRUE(preservation_state.has_scalar_spill());
  EXPECT_TRUE(preservation_state.has_compact_spill());
  EXPECT_EQ(preservation_state.scalar_spill_setup, point.scalar_spill_setup);
  EXPECT_TRUE(preservation_state.has_branch_only_spill);

  point.initialize_owner_epoch = true;
  point.set_owner_epoch_vgprs(60u, 61u);
  point.automatic_private_epoch = true;
  point.persistent_sgprs.set_owner_epoch(62u, 63u);
  EXPECT_EQ(scalar_preservation_state(point), preservation_state);
}

TEST(ConSanLoweringPlan, RequiresAnAccessConsumerForAtomicMetadata) {
  Request request;
  const BoundRuntimeResources resources;
  const TransformPolicy policy;
  request.track_atomics = true;

  const auto no_access = plan_object(request, resources, policy, {.admitted_atomic_count = 1u});
  EXPECT_FALSE(no_access.track_atomics);
  EXPECT_FALSE(no_access.warning.empty());

  const auto access = plan_object(request, resources, policy,
                                  {.has_access_candidate = true, .admitted_atomic_count = 1u});
  EXPECT_TRUE(access.track_atomics);
  EXPECT_TRUE(access.warning.empty());
}

TEST(ConSanLoweringPlan, OwnerDefaultAndOverride) {
  Request request;
  const BoundRuntimeResources resources;
  const TransformPolicy policy;
  const ObjectFacts facts{.has_access_candidate = true};

  EXPECT_EQ(plan_object(request, resources, policy, facts).owner_source, OwnerSource::WorkitemId);
  request.owner_source = OwnerSource::HwId;
  EXPECT_EQ(plan_object(request, resources, policy, facts).owner_source, OwnerSource::HwId);
}

TEST(ConSanLoweringPlan, ReportLayoutAndReservedAtomicBudget) {
  Request request;
  BoundRuntimeResources resources;
  TransformPolicy policy;
  const ObjectFacts facts{
      .has_access_candidate = true,
      .admitted_atomic_count = 4u,
  };
  resources.report_buffer_size = 1u << 20u;
  policy.max_patches = 2u;

  request.track_barriers = true;
  request.track_atomics = true;
  const auto plan = plan_object(request, resources, policy, facts);
  EXPECT_EQ(plan.report_layout,
            direct_report_buffer_layout_for_bytes(resources.report_buffer_size));
  EXPECT_EQ(plan.reserved_atomic_patch_count, 2u);
}

TEST(ConSanLoweringPlan, DynamicStackSpillsRequireSupportedTarget) {
  EXPECT_TRUE(is_capability_arch(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(is_capability_arch(ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_FALSE(is_capability_arch(ROCJITSU_CODE_ARCH_INVALID));
}

TEST(ConSanLoweringPlan, AtomicOperandOverlapSpill) {
  Request request;
  const OperatingPoint point;
  const ProgramSite candidate_site;
  const Candidate candidate(candidate_site);
  const auto plan = [&](ResourceSiteKind kind, rj_code_arch_t arch, bool disjoint,
                        const Candidate *access) {
    auto resource_facts = resolve_access_resource_facts(point, candidate, arch);
    resource_facts.guest_replay_requires_disjoint_address_scratch = disjoint;
    return operand_overlap_spill({request, resource_facts, access, kind});
  };

  EXPECT_TRUE(plan(ResourceSiteKind::Atomic, ROCJITSU_CODE_ARCH_CDNA5, false, nullptr).supported);
}

TEST(ConSanLoweringPlan, DispatchIdentityDemand) {
  Request request;

  request.runtime_sample_stride = 8u;
  EXPECT_TRUE(requires_dispatch_identity(request, {}));
  EXPECT_FALSE(
      requires_dispatch_identity(request, {.access_reports_need_explicit_identity = false}));
  request.runtime_sample_stride = 1u;
  EXPECT_TRUE(requires_dispatch_identity(request, {.access_reports_need_explicit_identity = true,
                                                   .has_access_or_atomic_consumer = true}));
  EXPECT_FALSE(requires_dispatch_identity(request, {.access_reports_need_explicit_identity = true,
                                                    .has_access_or_atomic_consumer = false}));
}

TEST(ConSanLoweringPlan, ScalarAbiPreservesGuestState) {
  OperatingPoint point;
  point.exec_save_sgpr = 20u;

  point.automatic_scalar_spill_layout = ScalarSpillLayout::Compact;
  point.scalar_spill_setup = ScalarSpillSetup{
      .temporaries = ScalarSpillTemporaries{40u, 42u},
  };
  auto plan = plan_special_state(scalar_preservation_state(point));
  EXPECT_EQ(plan, (detail::SpecialStateSgprs{22u, 42u}));

  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->vcc_save_sgpr, 22u);
  EXPECT_EQ(plan->scc_save_sgpr, 42u);
}

TEST(ConSanLoweringPlan, PersistentStateFollowsConsumers) {
  Request request;
  OperatingPoint point;
  point.initialize_owner_epoch = false;

  request.track_barriers = false;
  auto demand = plan_persistent_state_demand(request, point, {});
  EXPECT_FALSE(demand.needs_persistent_state);
  EXPECT_FALSE(demand.private_state_supported);
  EXPECT_FALSE(demand.scalar_state_required_for_private_or_overflow);

  demand = plan_persistent_state_demand(request, point, {.atomic_count = 1u});
  EXPECT_TRUE(demand.needs_entry_workgroup_tuple);
  EXPECT_TRUE(demand.needs_persistent_state);
  EXPECT_TRUE(demand.synchronization_requires_persistent_owner);
  EXPECT_TRUE(demand.private_workgroup_tuple_supported);
}

} // namespace
} // namespace rocjitsu::consan

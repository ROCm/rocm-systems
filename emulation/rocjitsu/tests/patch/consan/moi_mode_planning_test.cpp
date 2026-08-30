// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"

namespace rocjitsu {
namespace {

using consan_moi_impl::MoiObjectFacts;
using consan_moi_impl::MoiPersistentStateFacts;
using consan_moi_impl::plan_moi_dispatch_identity;
using consan_moi_impl::plan_moi_dynamic_stack_spill;
using consan_moi_impl::plan_moi_object_mode;
using consan_moi_impl::plan_moi_operand_overlap_spill;
using consan_moi_impl::plan_moi_persistent_state_demand;
using consan_moi_impl::plan_moi_scalar_abi;

TEST(ConSanMoiModePlanning, EachEngineOwnsItsAutomaticOwnerDefault) {
  ConSanRequest request;
  ConSanMoiOperatingPoint point;
  const MoiObjectFacts facts{.has_access_candidate = true};
  const ConSanObservationPlan observation;

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  EXPECT_EQ(plan_moi_object_mode(request, point, facts, observation).owner_source,
            ConSanMoiOwnerSource::WorkitemId);
  request.moi_engine = ConSanMoiEngine::Sampled;
  EXPECT_EQ(plan_moi_object_mode(request, point, facts, observation).owner_source,
            ConSanMoiOwnerSource::WorkitemId);
  request.moi_engine = ConSanMoiEngine::InlineShadow;
  EXPECT_EQ(plan_moi_object_mode(request, point, facts, observation).owner_source,
            ConSanMoiOwnerSource::HwId);

  request.moi_owner_source = ConSanMoiOwnerSource::HwId;
  request.moi_engine = ConSanMoiEngine::RecordReplay;
  EXPECT_EQ(plan_moi_object_mode(request, point, facts, observation).owner_source,
            ConSanMoiOwnerSource::HwId);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsItsDynamicStackSpillPolicy) {
  const auto record_rdna =
      plan_moi_dynamic_stack_spill(ConSanMoiEngine::RecordReplay, ROCJITSU_CODE_ARCH_RDNA4);
  const auto sampled_cdna =
      plan_moi_dynamic_stack_spill(ConSanMoiEngine::Sampled, ROCJITSU_CODE_ARCH_CDNA4);
  const auto record_unsupported =
      plan_moi_dynamic_stack_spill(ConSanMoiEngine::RecordReplay, ROCJITSU_CODE_ARCH_INVALID);
  const auto inline_unsupported =
      plan_moi_dynamic_stack_spill(ConSanMoiEngine::InlineShadow, ROCJITSU_CODE_ARCH_INVALID);

  EXPECT_TRUE(record_rdna.backend_supported);
  EXPECT_TRUE(record_rdna.requires_every_owner_dynamic);
  EXPECT_TRUE(sampled_cdna.backend_supported);
  EXPECT_TRUE(sampled_cdna.requires_every_owner_dynamic);
  EXPECT_FALSE(record_unsupported.backend_supported);
  EXPECT_TRUE(record_unsupported.requires_every_owner_dynamic);
  EXPECT_TRUE(inline_unsupported.backend_supported);
  EXPECT_FALSE(inline_unsupported.requires_every_owner_dynamic);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsItsOperandOverlapSpillPolicy) {
  ConSanRequest request;
  const ConSanMoiOperatingPoint point;
  const ConSanMoiCandidate candidate;
  const auto plan = [&](ConSanResourceSiteKind kind, rj_code_arch_t arch, bool disjoint,
                        const ConSanMoiCandidate *access) {
    return plan_moi_operand_overlap_spill({request, point, access, kind, arch, disjoint});
  };

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  EXPECT_TRUE(
      plan(ConSanResourceSiteKind::Access, ROCJITSU_CODE_ARCH_RDNA4, false, &candidate).supported);
  request.moi_dynamic_access_records = true;
  EXPECT_FALSE(
      plan(ConSanResourceSiteKind::Access, ROCJITSU_CODE_ARCH_RDNA4, false, &candidate).supported);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  EXPECT_FALSE(
      plan(ConSanResourceSiteKind::Access, ROCJITSU_CODE_ARCH_RDNA4, true, &candidate).supported);

  request.moi_engine = ConSanMoiEngine::Sampled;
  EXPECT_TRUE(
      plan(ConSanResourceSiteKind::Atomic, ROCJITSU_CODE_ARCH_CDNA5, false, nullptr).supported);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsItsDispatchIdentityPolicy) {
  ConSanRequest request;

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  auto plan = plan_moi_dispatch_identity(request, {});
  EXPECT_TRUE(plan.needs_dispatch_id);
  EXPECT_EQ(plan.fallback_kind, ConSanMoiFallbackKind::RecordReplayZeroGeneration);
  EXPECT_FALSE(plan.fallback_replans_dispatch_only);

  request.moi_engine = ConSanMoiEngine::Sampled;
  request.moi_runtime_sample_stride = 8u;
  plan = plan_moi_dispatch_identity(request, {});
  EXPECT_TRUE(plan.needs_dispatch_id);
  EXPECT_EQ(plan.fallback_kind, ConSanMoiFallbackKind::SampledLiteralDispatchId);
  EXPECT_TRUE(plan.fallback_replans_dispatch_only);
  EXPECT_FALSE(plan_moi_dispatch_identity(request, {.target_uses_gfx12_cdna_execution = true})
                   .needs_dispatch_id);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  plan = plan_moi_dispatch_identity(request, {.has_access_or_atomic_consumer = true});
  EXPECT_TRUE(plan.needs_dispatch_id);
  EXPECT_FALSE(plan.fallback_kind);
  request.moi_track_atomics = false;
  EXPECT_FALSE(plan_moi_dispatch_identity(request, {.target_uses_gfx12_cdna_execution = true,
                                                    .has_access_or_atomic_consumer = true})
                   .needs_dispatch_id);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsItsScalarAbiLayout) {
  ConSanRequest request;
  ConSanMoiOperatingPoint point;
  point.moi_exec_save_sgpr = 20u;

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  auto plan = plan_moi_scalar_abi(request, point);
  EXPECT_EQ(plan.special_state, (consan_detail::MoiSpecialStateSgprs{22u, 24u}));
  ASSERT_TRUE(plan.indirect_jump);
  EXPECT_EQ(plan.indirect_jump->pc_sgpr, 20u);
  EXPECT_EQ(plan.indirect_jump->scc_save_sgpr, 24u);
  EXPECT_FALSE(plan.access_router_uses_dense_abi);

  request.moi_engine = ConSanMoiEngine::Sampled;
  plan = plan_moi_scalar_abi(request, point);
  ASSERT_TRUE(plan.special_state);
  EXPECT_EQ(plan.special_state->vcc_save_sgpr, 22u);
  ASSERT_TRUE(plan.indirect_jump);
  EXPECT_EQ(plan.indirect_jump->pc_sgpr, 20u);
  EXPECT_EQ(plan.indirect_jump->scc_save_sgpr, 24u);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  plan = plan_moi_scalar_abi(request, point);
  EXPECT_EQ(plan.special_state, (consan_detail::MoiSpecialStateSgprs{28u, 30u}));
  ASSERT_TRUE(plan.indirect_jump);
  EXPECT_EQ(plan.indirect_jump->pc_sgpr, 32u);
  EXPECT_EQ(plan.indirect_jump->scc_save_sgpr, 30u);
  EXPECT_TRUE(plan.access_router_uses_dense_abi);

  point.automatic_moi_inline_sgpr_spill = true;
  point.moi_inline_indirect_pc_sgpr = 40u;
  point.moi_inline_indirect_scc_sgpr = 42u;
  plan = plan_moi_scalar_abi(request, point);
  ASSERT_TRUE(plan.indirect_jump);
  EXPECT_EQ(plan.indirect_jump->pc_sgpr, 40u);
  EXPECT_EQ(plan.indirect_jump->scc_save_sgpr, 42u);

  point.moi_exec_save_sgpr.reset();
  plan = plan_moi_scalar_abi(request, point);
  EXPECT_FALSE(plan.special_state);
  ASSERT_TRUE(plan.indirect_jump);
  EXPECT_EQ(plan.indirect_jump->pc_sgpr, 40u);
  EXPECT_EQ(plan.indirect_jump->scc_save_sgpr, 42u);
}

TEST(ConSanMoiModePlanning, RecordReplaySelectsDenseRoutingFromNormalizedTargetFacts) {
  ConSanRequest request;
  request.moi_engine = ConSanMoiEngine::RecordReplay;
  ConSanMoiOperatingPoint point;
  const ConSanObservationPlan observation;

  MoiObjectFacts facts{.has_admitted_barrier = true,
                       .admitted_barrier_count = 33u,
                       .target_supports_dense_barrier_router = true};
  EXPECT_TRUE(plan_moi_object_mode(request, point, facts, observation).dense_barrier_router);

  facts.target_supports_dense_barrier_router = false;
  EXPECT_FALSE(plan_moi_object_mode(request, point, facts, observation).dense_barrier_router);

  facts.admitted_barrier_count = 1u;
  facts.has_stranded_admitted_barrier = true;
  facts.target_supports_dense_barrier_router = true;
  EXPECT_TRUE(plan_moi_object_mode(request, point, facts, observation).dense_barrier_router);
}

TEST(ConSanMoiModePlanning, RecordReplayDropsAutomaticStateOnlyWithoutConsumers) {
  ConSanRequest request;
  request.moi_engine = ConSanMoiEngine::RecordReplay;
  request.moi_track_atomics = true;
  request.moi_track_barriers = true;
  ConSanMoiOperatingPoint point;
  point.moi_initialize_owner_epoch = true;
  const ConSanObservationPlan observation;

  const auto empty = plan_moi_object_mode(request, point, {}, observation);
  EXPECT_EQ(empty.initialize_owner_epoch, false);
  EXPECT_FALSE(empty.track_barriers);
  ASSERT_EQ(empty.warnings.size(), 1u);

  MoiObjectFacts atomic{.has_admitted_atomic = true};
  const auto consumed = plan_moi_object_mode(request, point, atomic, observation);
  EXPECT_EQ(consumed.initialize_owner_epoch, true);
  EXPECT_TRUE(consumed.track_barriers);
  EXPECT_TRUE(consumed.atomic_or_fence_relevant);
}

TEST(ConSanMoiModePlanning, SampledRequiresAnAccessConsumerForAtomicMetadata) {
  ConSanRequest request;
  request.moi_engine = ConSanMoiEngine::Sampled;
  request.moi_track_atomics = true;
  const ConSanMoiOperatingPoint point;
  const ConSanObservationPlan observation;

  const auto no_access =
      plan_moi_object_mode(request, point, {.has_admitted_atomic = true}, observation);
  EXPECT_FALSE(no_access.track_atomics);
  EXPECT_FALSE(no_access.atomic_or_fence_relevant);
  ASSERT_EQ(no_access.warnings.size(), 1u);

  const auto access = plan_moi_object_mode(
      request, point, {.has_access_candidate = true, .has_admitted_atomic = true}, observation);
  EXPECT_TRUE(access.track_atomics);
  EXPECT_TRUE(access.atomic_or_fence_relevant);
  EXPECT_TRUE(access.warnings.empty());
}

TEST(ConSanMoiModePlanning, InlineDemandFollowsAdmittedConsumers) {
  ConSanRequest request;
  request.moi_engine = ConSanMoiEngine::InlineShadow;
  request.moi_track_atomics = true;
  request.moi_track_barriers = true;
  const ConSanMoiOperatingPoint point;
  const ConSanObservationPlan observation;

  const auto empty = plan_moi_object_mode(request, point, {}, observation);
  EXPECT_FALSE(empty.track_atomics);
  EXPECT_FALSE(empty.track_barriers);
  EXPECT_FALSE(empty.inline_access_present);
  EXPECT_FALSE(empty.inline_atomic_without_access);
  EXPECT_EQ(empty.warnings.size(), 2u);

  const auto atomic_only = plan_moi_object_mode(
      request, point, {.has_admitted_atomic = true, .has_admitted_barrier = true}, observation);
  EXPECT_TRUE(atomic_only.track_atomics);
  EXPECT_TRUE(atomic_only.track_barriers);
  EXPECT_TRUE(atomic_only.inline_atomic_without_access);

  request.moi_owner_source = ConSanMoiOwnerSource::WorkitemId;
  EXPECT_EQ(plan_moi_object_mode(request, point, {}, observation).errors.size(), 1u);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsItsProloguePublicationPolicy) {
  ConSanRequest request;
  ConSanMoiOperatingPoint point;
  const MoiObjectFacts facts{.has_access_candidate = true};
  const ConSanObservationPlan observation;

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  auto plan = plan_moi_object_mode(request, point, facts, observation);
  EXPECT_FALSE(plan.reserve_dynamic_stack_prologue_entry);
  EXPECT_TRUE(plan.prologue_requires_consumer);

  MoiObjectFacts buffered_facts = facts;
  buffered_facts.has_report_buffer = true;
  EXPECT_FALSE(
      plan_moi_object_mode(request, point, buffered_facts, observation).prologue_requires_consumer);

  request.moi_engine = ConSanMoiEngine::Sampled;
  plan = plan_moi_object_mode(request, point, facts, observation);
  EXPECT_TRUE(plan.reserve_dynamic_stack_prologue_entry);
  EXPECT_TRUE(plan.prologue_requires_consumer);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  plan = plan_moi_object_mode(request, point, facts, observation);
  EXPECT_FALSE(plan.reserve_dynamic_stack_prologue_entry);
  EXPECT_TRUE(plan.prologue_requires_consumer);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsPersistentStateDemand) {
  ConSanRequest request;
  BoundRuntimeResources resources;
  ConSanMoiOperatingPoint point;
  point.moi_initialize_owner_epoch = false;

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  request.moi_track_barriers = true;
  auto demand = plan_moi_persistent_state_demand(request, resources, point,
                                                 {.access_count = 66u, .barrier_count = 33u});
  EXPECT_TRUE(demand.needs_entry_workgroup_tuple);
  EXPECT_TRUE(demand.needs_persistent_state);
  EXPECT_TRUE(demand.prefer_compact_barriers);
  EXPECT_TRUE(demand.private_workgroup_tuple_supported);
  EXPECT_FALSE(demand.scalar_state_supported);
  EXPECT_TRUE(demand.private_state_supported);
  EXPECT_TRUE(demand.scalar_state_required_for_private_or_overflow);
  EXPECT_TRUE(demand.prefer_private_epoch_for_descriptor_growth);

  request.moi_engine = ConSanMoiEngine::Sampled;
  request.moi_track_barriers = false;
  demand = plan_moi_persistent_state_demand(request, resources, point, {});
  EXPECT_FALSE(demand.needs_persistent_state);
  EXPECT_TRUE(demand.scalar_state_supported);
  EXPECT_FALSE(demand.private_state_supported);
  EXPECT_FALSE(demand.scalar_state_required_for_private_or_overflow);
  EXPECT_TRUE(demand.prefer_private_epoch_for_descriptor_growth);

  demand = plan_moi_persistent_state_demand(request, resources, point, {.atomic_count = 1u});
  EXPECT_TRUE(demand.needs_entry_workgroup_tuple);
  EXPECT_TRUE(demand.needs_persistent_state);
  EXPECT_TRUE(demand.synchronization_requires_persistent_owner);
  EXPECT_TRUE(demand.private_workgroup_tuple_supported);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  point.automatic_moi_private_dispatch_id = true;
  demand = plan_moi_persistent_state_demand(request, resources, point,
                                            {.access_count = 1u,
                                             .has_operational_dynamic_stack_owner = true,
                                             .has_operational_dynamic_lds_owner = true});
  EXPECT_TRUE(demand.needs_workgroup_key);
  EXPECT_TRUE(demand.needs_persistent_state);
  EXPECT_TRUE(demand.private_dispatch_incompatible_with_dynamic_stack);
  EXPECT_TRUE(demand.prefer_private_epoch_for_dynamic_lds);
  EXPECT_TRUE(demand.scalar_state_supported);
  EXPECT_TRUE(demand.private_state_supported);
  EXPECT_TRUE(demand.scalar_state_required_for_private_or_overflow);
  EXPECT_FALSE(demand.prefer_private_epoch_for_descriptor_growth);
}

} // namespace
} // namespace rocjitsu

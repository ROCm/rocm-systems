// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"

namespace rocjitsu {
namespace {

using consan_moi_impl::find_moi_mode_operations;
using consan_moi_impl::moi_mode_operations;
using consan_moi_impl::MoiCdnaPersistentOverflowStrategy;
using consan_moi_impl::MoiObjectFacts;
using consan_moi_impl::MoiPersistentStateFacts;
using consan_moi_impl::plan_moi_dispatch_identity;
using consan_moi_impl::plan_moi_dynamic_stack_spill;
using consan_moi_impl::plan_moi_object_mode;
using consan_moi_impl::plan_moi_operand_overlap_spill;
using consan_moi_impl::plan_moi_persistent_state_demand;
using consan_moi_impl::plan_moi_scalar_abi;

consan_moi_impl::MoiObjectModePlan
plan_hypothetical_mode(const ConSanRequest &, const BoundRuntimeResources &,
                       const TransformPolicy &, const ConSanMoiOperatingPoint &,
                       const consan_moi_impl::MoiObjectFacts &facts,
                       const ConSanObservationPlan &) {
  consan_moi_impl::MoiObjectModePlan plan;
  plan.track_atomics = facts.has_admitted_atomic;
  plan.semantics.inline_access_present = facts.has_access_candidate;
  return plan;
}

uint16_t hypothetical_exec_save_sgpr_count(const MoiExecSaveRequirement &requirement,
                                           const consan_moi_impl::MoiExecSaveTargetFacts &target) {
  return requirement.has_report_buffer
             ? (target.direct_call_form == ConSanDirectCallForm::SCallI64 ? 3u : 2u)
             : 0u;
}

TEST(ConSanMoiModePlanning, HypotheticalModeRegistersWithoutConcreteTargetChanges) {
  enum class HypotheticalModeKey : uint8_t { FifthMode };
  consan_moi_impl::MoiModeOperations operations{};
  operations.plan = plan_hypothetical_mode;
  operations.operational_evidence = {ConSanProbeIntentKind::BarrierRecord,
                                     ConSanProbeIntentKind::SampledAtomicOrdering, false};
  operations.dynamic_stack_frame_save_sgpr_offset = 1u;
  operations.exec_save_sgpr_count = hypothetical_exec_save_sgpr_count;
  operations.prologue.one_based_owner_ids = true;
  const std::array registrations{
      consan_moi_impl::MoiModeRegistrationFor<HypotheticalModeKey>{HypotheticalModeKey::FifthMode,
                                                                   &operations},
  };

  const auto *selected =
      find_moi_mode_operations<HypotheticalModeKey>(registrations, HypotheticalModeKey::FifthMode);
  ASSERT_EQ(selected, &operations);
  const auto plan = selected->plan({}, {}, {}, {}, {.has_access_candidate = true}, {});
  EXPECT_TRUE(plan.semantics.inline_access_present);
  EXPECT_FALSE(plan.track_atomics);
  EXPECT_EQ(selected->operational_evidence.barrier, ConSanProbeIntentKind::BarrierRecord);
  EXPECT_EQ(selected->operational_evidence.atomic, ConSanProbeIntentKind::SampledAtomicOrdering);
  EXPECT_EQ(selected->exec_save_sgpr_count({.has_report_buffer = true},
                                           {.direct_call_form = ConSanDirectCallForm::SCallI64}),
            3u);
  EXPECT_TRUE(selected->prologue.one_based_owner_ids);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsItsAutomaticOwnerDefault) {
  ConSanRequest request;
  const BoundRuntimeResources resources;
  const TransformPolicy policy;
  ConSanMoiOperatingPoint point;
  const MoiObjectFacts facts{.has_access_candidate = true};
  const ConSanObservationPlan observation;

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  EXPECT_EQ(
      plan_moi_object_mode(request, resources, policy, point, facts, observation).owner_source,
      ConSanMoiOwnerSource::WorkitemId);
  request.moi_engine = ConSanMoiEngine::Sampled;
  EXPECT_EQ(
      plan_moi_object_mode(request, resources, policy, point, facts, observation).owner_source,
      ConSanMoiOwnerSource::WorkitemId);
  request.moi_engine = ConSanMoiEngine::InlineShadow;
  EXPECT_EQ(
      plan_moi_object_mode(request, resources, policy, point, facts, observation).owner_source,
      ConSanMoiOwnerSource::HwId);

  request.moi_owner_source = ConSanMoiOwnerSource::HwId;
  request.moi_engine = ConSanMoiEngine::RecordReplay;
  EXPECT_EQ(
      plan_moi_object_mode(request, resources, policy, point, facts, observation).owner_source,
      ConSanMoiOwnerSource::HwId);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsItsLegacyReportLayout) {
  ConSanRequest request;
  BoundRuntimeResources resources;
  TransformPolicy policy;
  const ConSanMoiOperatingPoint point;
  const MoiObjectFacts facts{
      .has_access_candidate = true,
      .has_admitted_atomic = true,
      .admitted_atomic_count = 4u,
      .has_admitted_barrier = true,
      .admitted_barrier_count = 3u,
  };
  const ConSanObservationPlan observation;
  resources.moi_report_buffer_size = 1u << 20u;
  policy.max_patches = 2u;

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  request.moi_track_barriers = true;
  request.moi_track_atomics = true;
  EXPECT_EQ(plan_moi_object_mode(request, resources, policy, point, facts, observation)
                .semantics.report_layout,
            consan_moi_report_buffer_layout_for_bytes(resources.moi_report_buffer_size, true, true,
                                                      true));

  request.moi_engine = ConSanMoiEngine::Sampled;
  const auto sampled = plan_moi_object_mode(request, resources, policy, point, facts, observation);
  EXPECT_EQ(
      sampled.semantics.report_layout,
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(resources.moi_report_buffer_size));
  EXPECT_EQ(sampled.semantics.reserved_barrier_island_count, 2u);
  EXPECT_EQ(sampled.semantics.reserved_atomic_island_count, 2u);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  EXPECT_EQ(
      plan_moi_object_mode(request, resources, policy, point, facts, observation)
          .semantics.report_layout,
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(resources.moi_report_buffer_size));
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

  EXPECT_EQ(moi_mode_operations(ConSanMoiEngine::RecordReplay).dynamic_stack_frame_save_sgpr_offset,
            5u);
  EXPECT_EQ(moi_mode_operations(ConSanMoiEngine::Sampled).dynamic_stack_frame_save_sgpr_offset, 8u);
  EXPECT_EQ(moi_mode_operations(ConSanMoiEngine::InlineShadow).dynamic_stack_frame_save_sgpr_offset,
            24u);
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
  EXPECT_TRUE(plan.permits_private_entry_capture);
  EXPECT_EQ(plan.fallback_kind, ConSanMoiFallbackKind::RecordReplayZeroGeneration);
  EXPECT_FALSE(plan.fallback_replans_dispatch_only);

  request.moi_engine = ConSanMoiEngine::Sampled;
  request.moi_runtime_sample_stride = 8u;
  plan = plan_moi_dispatch_identity(request, {});
  EXPECT_TRUE(plan.needs_dispatch_id);
  EXPECT_FALSE(plan.permits_private_entry_capture);
  EXPECT_EQ(plan.fallback_kind, ConSanMoiFallbackKind::SampledLiteralDispatchId);
  EXPECT_TRUE(plan.fallback_replans_dispatch_only);
  EXPECT_FALSE(plan_moi_dispatch_identity(request, {.access_reports_need_explicit_identity = false})
                   .needs_dispatch_id);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  plan = plan_moi_dispatch_identity(request, {.has_access_or_atomic_consumer = true});
  EXPECT_TRUE(plan.needs_dispatch_id);
  EXPECT_TRUE(plan.permits_private_entry_capture);
  EXPECT_FALSE(plan.fallback_kind);
  request.moi_track_atomics = false;
  EXPECT_FALSE(plan_moi_dispatch_identity(request, {.access_reports_need_explicit_identity = false,
                                                    .has_access_or_atomic_consumer = true})
                   .needs_dispatch_id);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsItsScalarAbiLayout) {
  ConSanRequest request;
  ConSanMoiOperatingPoint point;
  point.moi_exec_save_sgpr = 20u;

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  const auto &record_traits = moi_mode_operations(request.moi_engine).transient_scalar_placement;
  EXPECT_EQ(record_traits.spill_layout, ConSanMoiScalarSpillLayout::Compact);
  EXPECT_FALSE(record_traits.requires_capability_target);
  EXPECT_TRUE(record_traits.branch_only_spill_preserves_indirect_state);
  EXPECT_EQ(record_traits.compact_spill_scalar_count, 0u);
  auto plan = plan_moi_scalar_abi(request, point);
  EXPECT_EQ(plan.special_state, (consan_detail::MoiSpecialStateSgprs{22u, 24u}));
  ASSERT_TRUE(plan.indirect_jump);
  EXPECT_EQ(plan.indirect_jump->pc_sgpr, 20u);
  EXPECT_EQ(plan.indirect_jump->scc_save_sgpr, 24u);
  EXPECT_FALSE(plan.access_router_uses_dense_abi);

  request.moi_engine = ConSanMoiEngine::Sampled;
  const auto &sampled_traits = moi_mode_operations(request.moi_engine).transient_scalar_placement;
  EXPECT_EQ(sampled_traits.spill_layout, ConSanMoiScalarSpillLayout::Compact);
  EXPECT_TRUE(sampled_traits.requires_capability_target);
  EXPECT_FALSE(sampled_traits.branch_only_spill_preserves_indirect_state);
  EXPECT_EQ(sampled_traits.compact_spill_scalar_count, 8u);
  plan = plan_moi_scalar_abi(request, point);
  ASSERT_TRUE(plan.special_state);
  EXPECT_EQ(plan.special_state->vcc_save_sgpr, 22u);
  ASSERT_TRUE(plan.indirect_jump);
  EXPECT_EQ(plan.indirect_jump->pc_sgpr, 20u);
  EXPECT_EQ(plan.indirect_jump->scc_save_sgpr, 24u);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  const auto &inline_traits = moi_mode_operations(request.moi_engine).transient_scalar_placement;
  EXPECT_EQ(inline_traits.spill_layout, ConSanMoiScalarSpillLayout::Inline);
  EXPECT_FALSE(inline_traits.requires_capability_target);
  EXPECT_FALSE(inline_traits.branch_only_spill_preserves_indirect_state);
  EXPECT_EQ(inline_traits.compact_spill_scalar_count, 0u);
  plan = plan_moi_scalar_abi(request, point);
  EXPECT_EQ(plan.special_state, (consan_detail::MoiSpecialStateSgprs{28u, 30u}));
  ASSERT_TRUE(plan.indirect_jump);
  EXPECT_EQ(plan.indirect_jump->pc_sgpr, 32u);
  EXPECT_EQ(plan.indirect_jump->scc_save_sgpr, 30u);
  EXPECT_TRUE(plan.access_router_uses_dense_abi);

  point.automatic_moi_scalar_spill_layout = ConSanMoiScalarSpillLayout::Inline;
  point.moi_router_jump = ConSanMoiIndirectJumpSgprs{40u, 42u};
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
  const BoundRuntimeResources resources;
  const TransformPolicy policy;
  request.moi_engine = ConSanMoiEngine::RecordReplay;
  ConSanMoiOperatingPoint point;
  const ConSanObservationPlan observation;

  MoiObjectFacts facts{.has_admitted_barrier = true,
                       .admitted_barrier_count = 33u,
                       .target_supports_dense_barrier_router = true};
  EXPECT_TRUE(plan_moi_object_mode(request, resources, policy, point, facts, observation)
                  .semantics.dense_barrier_router);

  facts.target_supports_dense_barrier_router = false;
  EXPECT_FALSE(plan_moi_object_mode(request, resources, policy, point, facts, observation)
                   .semantics.dense_barrier_router);

  facts.admitted_barrier_count = 1u;
  facts.has_stranded_admitted_barrier = true;
  facts.target_supports_dense_barrier_router = true;
  EXPECT_TRUE(plan_moi_object_mode(request, resources, policy, point, facts, observation)
                  .semantics.dense_barrier_router);
}

TEST(ConSanMoiModePlanning, RecordReplayDropsAutomaticStateOnlyWithoutConsumers) {
  ConSanRequest request;
  const BoundRuntimeResources resources;
  const TransformPolicy policy;
  request.moi_engine = ConSanMoiEngine::RecordReplay;
  request.moi_track_atomics = true;
  request.moi_track_barriers = true;
  ConSanMoiOperatingPoint point;
  point.moi_initialize_owner_epoch = true;
  const ConSanObservationPlan observation;

  const auto empty = plan_moi_object_mode(request, resources, policy, point, {}, observation);
  EXPECT_EQ(empty.initialize_owner_epoch, false);
  EXPECT_FALSE(empty.track_barriers);
  ASSERT_EQ(empty.warnings.size(), 1u);

  MoiObjectFacts atomic{.has_admitted_atomic = true};
  const auto consumed =
      plan_moi_object_mode(request, resources, policy, point, atomic, observation);
  EXPECT_EQ(consumed.initialize_owner_epoch, true);
  EXPECT_TRUE(consumed.track_barriers);
  EXPECT_TRUE(consumed.warnings.empty());
}

TEST(ConSanMoiModePlanning, SampledRequiresAnAccessConsumerForAtomicMetadata) {
  ConSanRequest request;
  const BoundRuntimeResources resources;
  const TransformPolicy policy;
  request.moi_engine = ConSanMoiEngine::Sampled;
  request.moi_track_atomics = true;
  const ConSanMoiOperatingPoint point;
  const ConSanObservationPlan observation;

  const auto no_access = plan_moi_object_mode(request, resources, policy, point,
                                              {.has_admitted_atomic = true}, observation);
  EXPECT_FALSE(no_access.track_atomics);
  ASSERT_EQ(no_access.warnings.size(), 1u);

  const auto access = plan_moi_object_mode(
      request, resources, policy, point,
      {.has_access_candidate = true, .has_admitted_atomic = true, .admitted_atomic_count = 1u},
      observation);
  EXPECT_TRUE(access.track_atomics);
  EXPECT_TRUE(access.warnings.empty());
}

TEST(ConSanMoiModePlanning, InlineDemandFollowsAdmittedConsumers) {
  ConSanRequest request;
  const BoundRuntimeResources resources;
  const TransformPolicy policy;
  request.moi_engine = ConSanMoiEngine::InlineShadow;
  request.moi_track_atomics = true;
  request.moi_track_barriers = true;
  const ConSanMoiOperatingPoint point;
  const ConSanObservationPlan observation;

  const auto empty = plan_moi_object_mode(request, resources, policy, point, {}, observation);
  EXPECT_FALSE(empty.track_atomics);
  EXPECT_FALSE(empty.track_barriers);
  EXPECT_FALSE(empty.semantics.inline_access_present);
  EXPECT_FALSE(empty.inline_atomic_without_access);
  EXPECT_EQ(empty.warnings.size(), 2u);

  const auto atomic_only = plan_moi_object_mode(request, resources, policy, point,
                                                {.has_admitted_atomic = true,
                                                 .admitted_atomic_count = 1u,
                                                 .has_admitted_barrier = true,
                                                 .admitted_barrier_count = 1u},
                                                observation);
  EXPECT_TRUE(atomic_only.track_atomics);
  EXPECT_TRUE(atomic_only.track_barriers);
  EXPECT_TRUE(atomic_only.inline_atomic_without_access);

  request.moi_owner_source = ConSanMoiOwnerSource::WorkitemId;
  EXPECT_EQ(plan_moi_object_mode(request, resources, policy, point, {}, observation).errors.size(),
            1u);
}

TEST(ConSanMoiModePlanning, EachEngineOwnsItsProloguePublicationPolicy) {
  ConSanRequest request;
  const BoundRuntimeResources resources;
  const TransformPolicy policy;
  ConSanMoiOperatingPoint point;
  const MoiObjectFacts facts{.has_access_candidate = true};
  const ConSanObservationPlan observation;

  request.moi_engine = ConSanMoiEngine::RecordReplay;
  auto plan = plan_moi_object_mode(request, resources, policy, point, facts, observation);
  EXPECT_FALSE(plan.reserve_dynamic_stack_prologue_entry);
  EXPECT_TRUE(plan.prologue_requires_consumer);

  MoiObjectFacts buffered_facts = facts;
  buffered_facts.has_report_buffer = true;
  EXPECT_FALSE(plan_moi_object_mode(request, resources, policy, point, buffered_facts, observation)
                   .prologue_requires_consumer);
  const auto &record_policy = moi_mode_operations(request.moi_engine).prologue;
  EXPECT_FALSE(record_policy.skip_unobserved_barrier_only_initialization);
  EXPECT_TRUE(record_policy.backup_compact_spill_for_runtime_sampling);
  EXPECT_FALSE(record_policy.one_based_owner_ids);
  EXPECT_FALSE(record_policy.persistent_state_requires_in_place_entry);

  request.moi_engine = ConSanMoiEngine::Sampled;
  plan = plan_moi_object_mode(request, resources, policy, point, facts, observation);
  EXPECT_TRUE(plan.reserve_dynamic_stack_prologue_entry);
  EXPECT_TRUE(plan.prologue_requires_consumer);
  const auto &sampled_policy = moi_mode_operations(request.moi_engine).prologue;
  EXPECT_FALSE(sampled_policy.skip_unobserved_barrier_only_initialization);
  EXPECT_FALSE(sampled_policy.backup_compact_spill_for_runtime_sampling);
  EXPECT_FALSE(sampled_policy.one_based_owner_ids);
  EXPECT_FALSE(sampled_policy.persistent_state_requires_in_place_entry);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  plan = plan_moi_object_mode(request, resources, policy, point, facts, observation);
  EXPECT_FALSE(plan.reserve_dynamic_stack_prologue_entry);
  EXPECT_TRUE(plan.prologue_requires_consumer);
  const auto &inline_policy = moi_mode_operations(request.moi_engine).prologue;
  EXPECT_TRUE(inline_policy.skip_unobserved_barrier_only_initialization);
  EXPECT_FALSE(inline_policy.backup_compact_spill_for_runtime_sampling);
  EXPECT_TRUE(inline_policy.one_based_owner_ids);
  EXPECT_TRUE(inline_policy.persistent_state_requires_in_place_entry);
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
  EXPECT_EQ(demand.cdna_overflow_strategy, MoiCdnaPersistentOverflowStrategy::ExactWorkgroupState);

  request.moi_engine = ConSanMoiEngine::Sampled;
  request.moi_track_barriers = false;
  demand = plan_moi_persistent_state_demand(request, resources, point, {});
  EXPECT_FALSE(demand.needs_persistent_state);
  EXPECT_TRUE(demand.scalar_state_supported);
  EXPECT_FALSE(demand.private_state_supported);
  EXPECT_FALSE(demand.scalar_state_required_for_private_or_overflow);
  EXPECT_TRUE(demand.prefer_private_epoch_for_descriptor_growth);
  EXPECT_EQ(demand.cdna_overflow_strategy, MoiCdnaPersistentOverflowStrategy::OwnerSnapshot);

  demand = plan_moi_persistent_state_demand(request, resources, point, {.atomic_count = 1u});
  EXPECT_TRUE(demand.needs_entry_workgroup_tuple);
  EXPECT_TRUE(demand.needs_persistent_state);
  EXPECT_TRUE(demand.synchronization_requires_persistent_owner);
  EXPECT_TRUE(demand.private_workgroup_tuple_supported);

  request.moi_engine = ConSanMoiEngine::InlineShadow;
  point.moi_dispatch_identity.set_private_fallback(true);
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
  EXPECT_EQ(demand.cdna_overflow_strategy,
            MoiCdnaPersistentOverflowStrategy::ResidentWavePrivateState);
}

} // namespace
} // namespace rocjitsu

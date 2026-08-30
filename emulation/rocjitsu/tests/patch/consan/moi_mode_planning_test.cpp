// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"

namespace rocjitsu {
namespace {

using consan_moi_impl::MoiObjectFacts;
using consan_moi_impl::MoiPersistentStateFacts;
using consan_moi_impl::plan_moi_object_mode;
using consan_moi_impl::plan_moi_persistent_state_demand;

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

  request.moi_engine = ConSanMoiEngine::Sampled;
  request.moi_track_barriers = false;
  demand = plan_moi_persistent_state_demand(request, resources, point, {});
  EXPECT_FALSE(demand.needs_persistent_state);

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
}

} // namespace
} // namespace rocjitsu

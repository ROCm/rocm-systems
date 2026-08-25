// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include <functional>

namespace rocjitsu {
namespace {

const std::vector<uint8_t> &barrier_policy_bytes() {
  static const std::vector<uint8_t> bytes(256, 0);
  return bytes;
}

ConSanSyncEvent
make_barrier_event(uint64_t offset,
                   ConSanSyncOperation operation = ConSanSyncOperation::BarrierFull,
                   ConSanBarrierSite::Scope scope = ConSanBarrierSite::Scope::Workgroup,
                   std::string container = "barrier_kernel") {
  ConSanSyncEvent event;
  event.semantic_id = {
      .physical =
          {
              .code_object = make_consan_code_object_id(barrier_policy_bytes()),
              .original_text_offset = offset,
          },
      .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
      .member_ordinal = 0,
      .range_ordinal = 0,
  };
  event.kind = ConSanSyncEventKind::Barrier;
  event.operation = operation;
  event.memory_role =
      operation == ConSanSyncOperation::BarrierSignal ? ConSanSyncMemoryRole::Release
      : operation == ConSanSyncOperation::BarrierWait ? ConSanSyncMemoryRole::Acquire
                                                      : ConSanSyncMemoryRole::AcquireRelease;
  event.confidence = ConSanSemanticConfidence::Exact;
  event.memory_role_confidence = ConSanSemanticConfidence::Exact;
  event.identity = container + "|barrier=" + std::to_string(offset);
  event.code_object_fingerprint = event.semantic_id.physical.code_object.fingerprint;
  event.container_name = std::move(container);
  event.in_kernel = true;
  event.text_offset = offset;
  event.file_offset = offset;
  event.size = sizeof(uint32_t);
  event.mnemonic = "s_barrier";
  event.barrier_id = 0;
  event.barrier_operand_source = ConSanBarrierSite::OperandSource::Immediate;
  event.barrier_scope = scope;
  event.execution_owners.push_back({});
  return event;
}

ConSanSyncSequence make_barrier_sequence(std::span<const ConSanSyncEvent> events) {
  ConSanSyncSequence sequence;
  sequence.kind = ConSanSyncSequenceKind::Barrier;
  sequence.operation = ConSanSyncOperation::BarrierFull;
  sequence.memory_role = ConSanSyncMemoryRole::AcquireRelease;
  sequence.confidence = ConSanSemanticConfidence::Exact;
  sequence.memory_role_confidence = ConSanSemanticConfidence::Exact;
  sequence.identity = "barrier-sequence";
  sequence.container_name = events.front().container_name;
  sequence.in_kernel = events.front().in_kernel;
  sequence.begin_text_offset = events.front().text_offset;
  sequence.end_text_offset = events.back().text_offset + events.back().size;
  sequence.basic_block_index = 0;
  sequence.barrier_id = 0;
  sequence.barrier_operand_source = ConSanBarrierSite::OperandSource::Immediate;
  sequence.barrier_scope = events.front().barrier_scope;
  sequence.execution_owners.push_back({});
  for (const ConSanSyncEvent &event : events) {
    SemanticSiteId member = event.semantic_id;
    member.domain = ConSanSemanticSiteDomain::SynchronizationSequenceMember;
    sequence.member_semantic_ids.push_back(std::move(member));
    sequence.member_event_identities.push_back(event.identity);
  }
  return sequence;
}

ProgramInventory
build_barrier_inventory(std::vector<ConSanSyncEvent> events,
                        std::vector<ConSanSyncSequence> sequences,
                        rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                        rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201) {
  ProgramInventoryBuilder builder(barrier_policy_bytes());
  builder.set_code_object_facts(true, 0, arch, target);
  SynchronizationInventoryBuildView synchronization = builder.synchronization();
  synchronization.sync_events = std::move(events);
  synchronization.sync_sequences = std::move(sequences);
  return builder.view();
}

ProgramInventory
one_full_barrier_inventory(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                           rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201) {
  std::vector events{make_barrier_event(32)};
  std::vector sequences{make_barrier_sequence(events)};
  return build_barrier_inventory(std::move(events), std::move(sequences), arch, target);
}

ConSanBarrierPolicyRequest barrier_request(ConSanCapabilityEngine engine) {
  return {
      .engine = engine,
      .tracking_enabled = true,
      .container_filter = {},
  };
}

TEST(ConSanBarrierPolicy, InvalidEngineFailsValidationButEmptyInventoryIsAValidEmptyPlan) {
  const ConSanBarrierPolicyResult invalid = plan_consan_barrier_observation(
      one_full_barrier_inventory(), barrier_request(ConSanCapabilityEngine::Count));
  EXPECT_FALSE(invalid.valid());
  EXPECT_FALSE(invalid.plan.valid());

  const ConSanBarrierPolicyResult empty = plan_consan_barrier_observation(
      ProgramInventory{}, barrier_request(ConSanCapabilityEngine::RecordReplay));
  EXPECT_TRUE(empty.valid());
  EXPECT_TRUE(empty.plan.barrier_site_decisions.empty());
  EXPECT_TRUE(empty.plan.probe_intents.empty());
}

TEST(ConSanBarrierPolicy, AllEnginesExpressTheirBarrierContract) {
  const ProgramInventory inventory = one_full_barrier_inventory();
  const ConSanBarrierPolicyResult supercollider = plan_consan_barrier_observation(
      inventory, barrier_request(ConSanCapabilityEngine::SuperCollider));
  ASSERT_TRUE(supercollider.valid());
  ASSERT_EQ(supercollider.plan.barrier_site_decisions.size(), 1u);
  EXPECT_EQ(supercollider.plan.barrier_site_decisions.front().kind,
            ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(supercollider.plan.barrier_site_decisions.front().reason,
            ConSanBarrierPolicyReason::EngineMutationOnly);
  EXPECT_TRUE(supercollider.plan.probe_intents.empty());

  constexpr std::array expected = {
      std::pair{ConSanCapabilityEngine::RecordReplay, ConSanProbeIntentKind::BarrierRecord},
      std::pair{ConSanCapabilityEngine::Sampled, ConSanProbeIntentKind::SampledBarrierEpoch},
      std::pair{ConSanCapabilityEngine::InlineShadow, ConSanProbeIntentKind::ExactBarrierEpoch},
  };
  for (const auto &[engine, intent_kind] : expected) {
    SCOPED_TRACE(consan_capability_engine_name(engine));
    const ConSanBarrierPolicyResult policy =
        plan_consan_barrier_observation(inventory, barrier_request(engine));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.barrier_site_decisions.size(), 1u);
    ASSERT_EQ(policy.plan.probe_intents.size(), 1u);
    EXPECT_EQ(policy.plan.barrier_site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);
    EXPECT_EQ(policy.plan.barrier_site_decisions.front().reason, ConSanBarrierPolicyReason::None);
    EXPECT_EQ(policy.plan.probe_intents.front().kind, intent_kind);
    EXPECT_EQ(policy.plan.probe_intents.front().position, ConSanProbePosition::After);
    EXPECT_EQ(policy.plan.probe_intents.front().requirement, ConSanProbeRequirement::Required);
  }
}

TEST(ConSanBarrierPolicy, PairedSequenceHasPerEventRecordsButOneEpochIntent) {
  std::vector events{
      make_barrier_event(32, ConSanSyncOperation::BarrierSignal),
      make_barrier_event(48, ConSanSyncOperation::BarrierWait),
  };
  std::vector sequences{make_barrier_sequence(events)};
  const ProgramInventory inventory = build_barrier_inventory(events, sequences);

  const ConSanBarrierPolicyResult record = plan_consan_barrier_observation(
      inventory, barrier_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(record.valid());
  ASSERT_EQ(record.plan.barrier_site_decisions.size(), 2u);
  ASSERT_EQ(record.plan.probe_intents.size(), 2u);
  EXPECT_NE(record.plan.barrier_site_decisions[0].intent_ids,
            record.plan.barrier_site_decisions[1].intent_ids);

  for (ConSanCapabilityEngine engine :
       {ConSanCapabilityEngine::Sampled, ConSanCapabilityEngine::InlineShadow}) {
    SCOPED_TRACE(consan_capability_engine_name(engine));
    const ConSanBarrierPolicyResult epoch =
        plan_consan_barrier_observation(inventory, barrier_request(engine));
    ASSERT_TRUE(epoch.valid());
    ASSERT_EQ(epoch.plan.barrier_site_decisions.size(), 2u);
    ASSERT_EQ(epoch.plan.probe_intents.size(), 1u);
    EXPECT_EQ(epoch.plan.barrier_site_decisions[0].intent_ids,
              epoch.plan.barrier_site_decisions[1].intent_ids);
    EXPECT_EQ(epoch.plan.probe_intents.front().physical_site.original_text_offset, 48u);
    EXPECT_EQ(epoch.plan.probe_intents.front().covered_semantic_sites.size(), 2u);
  }
}

TEST(ConSanBarrierPolicy, InlineSignalWithoutACompletionIsExplicitlyNotApplicable) {
  std::vector events{make_barrier_event(32, ConSanSyncOperation::BarrierSignal)};
  ConSanSyncSequence sequence = make_barrier_sequence(events);
  sequence.operation = ConSanSyncOperation::BarrierSignal;
  sequence.memory_role = ConSanSyncMemoryRole::Release;
  const ConSanBarrierPolicyResult policy = plan_consan_barrier_observation(
      build_barrier_inventory(std::move(events), {std::move(sequence)}),
      barrier_request(ConSanCapabilityEngine::InlineShadow));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.barrier_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.barrier_site_decisions.front().kind, ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(policy.plan.barrier_site_decisions.front().reason,
            ConSanBarrierPolicyReason::UnqualifiedSyncSequence);
  EXPECT_TRUE(policy.plan.probe_intents.empty());
}

TEST(ConSanBarrierPolicy, DisabledFilteredAndRuntimeEventsRemainExplicitlyOutsideContract) {
  const ProgramInventory inventory = one_full_barrier_inventory();
  ConSanBarrierPolicyRequest disabled = barrier_request(ConSanCapabilityEngine::RecordReplay);
  disabled.tracking_enabled = false;
  EXPECT_EQ(plan_consan_barrier_observation(inventory, disabled)
                .plan.barrier_site_decisions.front()
                .reason,
            ConSanBarrierPolicyReason::TrackingDisabled);

  ConSanBarrierPolicyRequest filtered = barrier_request(ConSanCapabilityEngine::RecordReplay);
  filtered.container_filter = "different";
  EXPECT_EQ(plan_consan_barrier_observation(inventory, filtered)
                .plan.barrier_site_decisions.front()
                .reason,
            ConSanBarrierPolicyReason::ContainerFilterExcluded);

  std::vector runtime_events{make_barrier_event(32, ConSanSyncOperation::BarrierFull,
                                                ConSanBarrierSite::Scope::Workgroup,
                                                "__amd_rocclr_runtime")};
  const ProgramInventory runtime =
      build_barrier_inventory(runtime_events, {make_barrier_sequence(runtime_events)});
  EXPECT_EQ(plan_consan_barrier_observation(runtime,
                                            barrier_request(ConSanCapabilityEngine::RecordReplay))
                .plan.barrier_site_decisions.front()
                .reason,
            ConSanBarrierPolicyReason::RuntimeKernelExcluded);
}

TEST(ConSanBarrierPolicy, ClusterScopeUsesTheTargetCapabilityContract) {
  std::vector events{
      make_barrier_event(32, ConSanSyncOperation::BarrierFull, ConSanBarrierSite::Scope::Cluster),
  };
  std::vector sequences{make_barrier_sequence(events)};
  const ConSanBarrierPolicyResult gfx1201 =
      plan_consan_barrier_observation(build_barrier_inventory(events, sequences),
                                      barrier_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(gfx1201.valid());
  EXPECT_EQ(gfx1201.plan.barrier_site_decisions.front().kind,
            ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(gfx1201.plan.barrier_site_decisions.front().reason,
            ConSanBarrierPolicyReason::TargetCapabilityUnavailable);

  const ConSanBarrierPolicyResult gfx1250 = plan_consan_barrier_observation(
      build_barrier_inventory(std::move(events), std::move(sequences), ROCJITSU_CODE_ARCH_CDNA5,
                              ROCJITSU_CODE_TARGET_GFX1250),
      barrier_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(gfx1250.valid());
  EXPECT_EQ(gfx1250.plan.barrier_site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);
}

TEST(ConSanBarrierPolicy, InvalidEncodingAndRedundantFullBarrierHaveTypedReasons) {
  std::vector invalid_events{make_barrier_event(32)};
  invalid_events.front().size = 8;
  std::vector invalid_sequences{make_barrier_sequence(invalid_events)};
  const ConSanBarrierPolicyResult invalid = plan_consan_barrier_observation(
      build_barrier_inventory(std::move(invalid_events), std::move(invalid_sequences)),
      barrier_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(invalid.valid());
  EXPECT_EQ(invalid.plan.barrier_site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(invalid.plan.barrier_site_decisions.front().reason,
            ConSanBarrierPolicyReason::InvalidBarrierEncoding);

  std::vector adjacent_events{make_barrier_event(32), make_barrier_event(36)};
  std::vector adjacent_sequences{
      make_barrier_sequence(std::span<const ConSanSyncEvent>(adjacent_events).first(1))};
  adjacent_sequences.push_back(
      make_barrier_sequence(std::span<const ConSanSyncEvent>(adjacent_events).subspan(1)));
  const ConSanBarrierPolicyResult adjacent = plan_consan_barrier_observation(
      build_barrier_inventory(std::move(adjacent_events), std::move(adjacent_sequences)),
      barrier_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(adjacent.valid());
  ASSERT_EQ(adjacent.plan.barrier_site_decisions.size(), 2u);
  EXPECT_EQ(adjacent.plan.barrier_site_decisions[0].kind, ConSanSiteDecisionKind::Admitted);
  EXPECT_EQ(adjacent.plan.barrier_site_decisions[1].reason,
            ConSanBarrierPolicyReason::RedundantAdjacentFullBarrier);
}

TEST(ConSanBarrierPolicy, SampledQualificationRejectsEveryRequiredSemanticFact) {
  using Mutation = std::function<void(ConSanSyncSequence &)>;
  const std::vector<std::pair<std::string_view, Mutation>> cases = {
      {"kind", [](auto &s) { s.kind = ConSanSyncSequenceKind::Fence; }},
      {"operation", [](auto &s) { s.operation = ConSanSyncOperation::BarrierWait; }},
      {"role", [](auto &s) { s.memory_role = ConSanSyncMemoryRole::Acquire; }},
      {"confidence", [](auto &s) { s.confidence = ConSanSemanticConfidence::Unsupported; }},
      {"role-confidence",
       [](auto &s) { s.memory_role_confidence = ConSanSemanticConfidence::Unsupported; }},
      {"block", [](auto &s) { s.basic_block_index.reset(); }},
      {"clause", [](auto &s) { s.inside_scalar_clause = true; }},
      {"owner", [](auto &s) { s.execution_owners.clear(); }},
      {"static-id",
       [](auto &s) { s.barrier_operand_source = ConSanBarrierSite::OperandSource::DynamicM0; }},
      {"id", [](auto &s) { s.barrier_id.reset(); }},
      {"scope", [](auto &s) { s.barrier_scope = ConSanBarrierSite::Scope::Unknown; }},
      {"members",
       [](auto &s) {
         s.member_semantic_ids.clear();
         s.member_event_identities.clear();
       }},
      {"bounds", [](auto &s) { s.end_text_offset = s.begin_text_offset; }},
  };
  for (const auto &[name, mutate] : cases) {
    SCOPED_TRACE(name);
    std::vector events{make_barrier_event(32)};
    std::vector sequences{make_barrier_sequence(events)};
    mutate(sequences.front());
    const ConSanBarrierPolicyResult policy = plan_consan_barrier_observation(
        build_barrier_inventory(std::move(events), std::move(sequences)),
        barrier_request(ConSanCapabilityEngine::Sampled));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.barrier_site_decisions.size(), 1u);
    EXPECT_EQ(policy.plan.barrier_site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
    EXPECT_EQ(policy.plan.barrier_site_decisions.front().reason,
              ConSanBarrierPolicyReason::UnqualifiedSyncSequence);
    EXPECT_TRUE(policy.plan.probe_intents.empty());
  }
}

TEST(ConSanBarrierPolicy, AmbiguousAndIncompleteSequencesFailClosedWithDistinctReasons) {
  std::vector events{make_barrier_event(32), make_barrier_event(48)};
  std::vector sequences{
      make_barrier_sequence(std::span<const ConSanSyncEvent>(events).first(1)),
      make_barrier_sequence(events),
  };
  const ConSanBarrierPolicyResult ambiguous = plan_consan_barrier_observation(
      build_barrier_inventory(events, sequences), barrier_request(ConSanCapabilityEngine::Sampled));
  ASSERT_TRUE(ambiguous.valid());
  EXPECT_EQ(ambiguous.plan.barrier_site_decisions.front().reason,
            ConSanBarrierPolicyReason::AmbiguousSequenceMembership);

  std::vector incomplete_events{make_barrier_event(32)};
  std::vector incomplete_sequences{make_barrier_sequence(incomplete_events)};
  incomplete_sequences.front().end_text_offset = 40;
  const ConSanBarrierPolicyResult incomplete = plan_consan_barrier_observation(
      build_barrier_inventory(std::move(incomplete_events), std::move(incomplete_sequences)),
      barrier_request(ConSanCapabilityEngine::Sampled));
  ASSERT_TRUE(incomplete.valid());
  EXPECT_EQ(incomplete.plan.barrier_site_decisions.front().reason,
            ConSanBarrierPolicyReason::MissingCompletingEvent);
}

TEST(ConSanBarrierPolicy, IdenticalAliasesCoalesceAndConflictingAliasesAreFatal) {
  ConSanSyncEvent first = make_barrier_event(32);
  ConSanSyncEvent second = first;
  second.container_name = "shared_alias";
  second.identity = "shared_alias|barrier=32";
  const ConSanBarrierPolicyResult coalesced =
      plan_consan_barrier_observation(build_barrier_inventory({first, second}, {}),
                                      barrier_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(coalesced.valid());
  ASSERT_EQ(coalesced.plan.barrier_site_decisions.size(), 1u);
  EXPECT_EQ(coalesced.plan.barrier_site_decisions.front().source_containers,
            (std::vector<std::string>{"barrier_kernel", "shared_alias"}));
  EXPECT_EQ(coalesced.plan.probe_intents.size(), 1u);

  second.size = 8;
  const ConSanBarrierPolicyResult conflicting =
      plan_consan_barrier_observation(build_barrier_inventory({first, second}, {}),
                                      barrier_request(ConSanCapabilityEngine::RecordReplay));
  EXPECT_FALSE(conflicting.valid());
  EXPECT_EQ(conflicting.errors,
            (std::vector{ConSanBarrierPolicyReason::ConflictingPhysicalAliases}));
  EXPECT_EQ(conflicting.plan.barrier_site_decisions.front().reason,
            ConSanBarrierPolicyReason::ConflictingPhysicalAliases);
}

TEST(ConSanBarrierPolicy, PolicyIsDeterministicAndDoesNotMutateInventory) {
  const ProgramInventory inventory = one_full_barrier_inventory();
  const SynchronizationInventoryView view = inventory.synchronization_view();
  const std::vector<ConSanSyncEvent> events(view.sync_events.begin(), view.sync_events.end());
  const std::vector<ConSanSyncSequence> sequences(view.sync_sequences.begin(),
                                                  view.sync_sequences.end());
  const ConSanBarrierPolicyRequest request = barrier_request(ConSanCapabilityEngine::InlineShadow);
  EXPECT_EQ(plan_consan_barrier_observation(inventory, request),
            plan_consan_barrier_observation(inventory, request));
  ASSERT_EQ(view.sync_events.size(), events.size());
  ASSERT_EQ(view.sync_sequences.size(), sequences.size());
  EXPECT_EQ(view.sync_events.front().identity, events.front().identity);
  EXPECT_EQ(view.sync_events.front().semantic_id, events.front().semantic_id);
  EXPECT_EQ(view.sync_events.front().operation, events.front().operation);
  EXPECT_EQ(view.sync_sequences.front().identity, sequences.front().identity);
  EXPECT_EQ(view.sync_sequences.front().member_semantic_ids, sequences.front().member_semantic_ids);
  EXPECT_EQ(view.sync_sequences.front().operation, sequences.front().operation);
}

} // namespace
} // namespace rocjitsu

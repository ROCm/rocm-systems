// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include <concepts>
#include <set>

namespace rocjitsu {
namespace {

template <typename Enum, size_t N, typename NameFunction>
void expect_observation_enum_contract(const std::array<Enum, N> &values, Enum count,
                                      NameFunction name, std::string_view invalid_name) {
  EXPECT_EQ(values.size(), static_cast<size_t>(count));
  std::set<std::string_view> names;
  for (size_t ordinal = 0; ordinal < values.size(); ++ordinal) {
    EXPECT_EQ(static_cast<size_t>(values[ordinal]), ordinal);
    EXPECT_NE(name(values[ordinal]), invalid_name);
    EXPECT_TRUE(names.insert(name(values[ordinal])).second);
  }
  EXPECT_EQ(name(count), invalid_name);
  EXPECT_EQ(name(static_cast<Enum>(255)), invalid_name);
}

struct AccessInventoryInput {
  std::vector<uint8_t> bytes = std::vector<uint8_t>(256, 0);
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201;
  std::vector<ConSanKernelInfo> kernels;
  std::vector<ConSanFunctionInfo> functions;
  std::vector<ConSanAccessInventorySite> accesses;
};

template <typename Container>
void stage_policy_access(AccessInventoryInput &input, const Container &container,
                         ConSanAccessInventorySite site) {
  site.container = consan_program_container_ref(container);
  input.accesses.push_back(std::move(site));
}

ConSanAccessInventorySite make_policy_lds_site(std::string mnemonic = "ds_store_b32",
                                               uint64_t text_offset = 32,
                                               uint64_t file_offset = 32) {
  ConSanAccessInventorySite site;
  site.origin = ConSanAccessOrigin::NativeLds;
  site.kind = ConSanLdsAccessKind::Write;
  site.physical_id.original_text_offset = text_offset;
  site.file_offset = file_offset;
  site.instruction_size = 8;
  site.decoded_width_bits = 32;
  site.operands.address_vgpr = 3;
  site.operands.data_vgpr = 4;
  site.mnemonic = std::move(mnemonic);
  return site;
}

ConSanAccessInventorySite make_policy_flat_site(ConSanFlatAddressSpaceHint hint,
                                                uint64_t text_offset = 64) {
  ConSanAccessInventorySite site;
  site.origin = ConSanAccessOrigin::Flat;
  site.kind = ConSanLdsAccessKind::Write;
  site.physical_id.original_text_offset = text_offset;
  site.file_offset = text_offset;
  site.instruction_size = 12;
  site.decoded_width_bits = 32;
  site.operands.address_vgpr = 5;
  site.operands.data_vgpr = 6;
  site.operands.raw_saddr = 0;
  site.operands.raw_scale_offset = true;
  site.operands.raw_ioffset = 0;
  site.flat_address_space_hint = hint;
  site.mnemonic = "flat_store_b32";
  return site;
}

ConSanKernelInfo make_policy_kernel(std::string name = "policy_kernel") {
  ConSanKernelInfo kernel;
  kernel.name = std::move(name);
  kernel.descriptor_file_offset = 192;
  kernel.entry_text_offset = 0;
  return kernel;
}

ProgramInventory build_policy_inventory(AccessInventoryInput input) {
  ProgramInventoryBuilder builder(input.bytes);
  builder.set_code_object_facts(true, 0, input.arch, input.target);
  builder.kernels() = std::move(input.kernels);
  builder.functions() = std::move(input.functions);
  builder.access_sites() = std::move(input.accesses);
  builder.publish_decoded_accesses(input.bytes);
  return builder.view();
}

ConSanAccessPolicyRequest policy_request(ConSanCapabilityEngine engine) {
  return {
      .engine = engine,
      .native_lds_enabled = true,
      .group_flat_enabled = true,
      .flat_provenance_mode = ConSanFlatProvenanceMode::Likely,
      .container_filter = {},
      .reserved_for_synchronization = {},
  };
}

ProgramInventory one_native_access_inventory(std::string mnemonic = "ds_store_b32") {
  AccessInventoryInput input;
  ConSanKernelInfo kernel = make_policy_kernel();
  stage_policy_access(input, kernel, make_policy_lds_site(std::move(mnemonic)));
  input.kernels.push_back(std::move(kernel));
  return build_policy_inventory(std::move(input));
}

TEST(ConSanObservationPlan, EnumContractsAreExhaustiveNamedAndRejectInvalidValues) {
  expect_observation_enum_contract(kConSanSiteDecisionKinds, ConSanSiteDecisionKind::Count,
                                   consan_site_decision_kind_name, "invalid-site-decision-kind");
  expect_observation_enum_contract(kConSanAccessPolicyReasons, ConSanAccessPolicyReason::Count,
                                   consan_access_policy_reason_name,
                                   "invalid-access-policy-reason");
  expect_observation_enum_contract(kConSanBarrierPolicyReasons, ConSanBarrierPolicyReason::Count,
                                   consan_barrier_policy_reason_name,
                                   "invalid-barrier-policy-reason");
  expect_observation_enum_contract(kConSanAtomicPolicyReasons, ConSanAtomicPolicyReason::Count,
                                   consan_atomic_policy_reason_name,
                                   "invalid-atomic-policy-reason");
  expect_observation_enum_contract(kConSanFencePolicyReasons, ConSanFencePolicyReason::Count,
                                   consan_fence_policy_reason_name, "invalid-fence-policy-reason");
  expect_observation_enum_contract(kConSanProbeIntentKinds, ConSanProbeIntentKind::Count,
                                   consan_probe_intent_kind_name, "invalid-probe-intent-kind");
  expect_observation_enum_contract(kConSanProbePositions, ConSanProbePosition::Count,
                                   consan_probe_position_name, "invalid-probe-position");
  expect_observation_enum_contract(
      kConSanDynamicResultRequirements, ConSanDynamicResultRequirement::Count,
      consan_dynamic_result_requirement_name, "invalid-dynamic-result-requirement");
  expect_observation_enum_contract(kConSanLoweringOutcomeKinds, ConSanLoweringOutcomeKind::Count,
                                   consan_lowering_outcome_kind_name,
                                   "invalid-lowering-outcome-kind");
}

TEST(ConSanObservationPlan, SynchronizationAssociationIdentityHasAnExplicitInvalidDefault) {
  EXPECT_FALSE(ConSanSynchronizationAssociationId{}.valid());
  EXPECT_TRUE(ConSanSynchronizationAssociationId{"sequence"}.valid());
  EXPECT_EQ(ConSanSynchronizationAssociationId{"sequence"},
            ConSanSynchronizationAssociationId{"sequence"});
  EXPECT_NE(ConSanSynchronizationAssociationId{"sequence"},
            ConSanSynchronizationAssociationId{"other-sequence"});
  EXPECT_LT(ConSanSynchronizationAssociationId{"sequence-a"},
            ConSanSynchronizationAssociationId{"sequence-b"});
}

TEST(ConSanObservationPlan, IntentIdentifiersAreExplicitlyInvalidAndPlanLocal) {
  EXPECT_FALSE(ConSanProbeIntentId{}.valid());
  EXPECT_TRUE(ConSanProbeIntentId{0}.valid());
  EXPECT_NE(ConSanProbeIntentId{0}, ConSanProbeIntentId{1});

  const ConSanAccessPolicyResult policy = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.probe_intents.size(), 1u);
  EXPECT_EQ(policy.plan.intent({0}), &policy.plan.probe_intents.front());
  EXPECT_EQ(policy.plan.intent({1}), nullptr);
  EXPECT_EQ(policy.plan.intent({}), nullptr);
}

TEST(ConSanObservationPlan, PlanValidationRejectsEveryBrokenTypedRelationship) {
  const ConSanAccessPolicyResult policy = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());

  ConSanObservationPlan broken = policy.plan;
  broken.engine = ConSanCapabilityEngine::Count;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().id = {7};
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().kind = ConSanProbeIntentKind::Count;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().covered_semantic_sites.clear();
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.site_decisions.front().reason = ConSanAccessPolicyReason::UnsupportedMnemonic;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.site_decisions.front().intent_ids = {{99}};
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.site_decisions.front().source_containers.clear();
  EXPECT_FALSE(broken.valid());
}

TEST(ConSanObservationPlan, BarrierDecisionValidationRejectsEveryBrokenTypedRelationship) {
  ProgramInventoryBuilder builder(std::array<uint8_t, 4>{});
  builder.set_code_object_facts(true, 0, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
  ConSanSyncEvent event;
  event.semantic_id = {
      .physical = {.code_object = builder.view().code_object_id(), .original_text_offset = 8},
      .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
  };
  event.kind = ConSanSyncEventKind::Barrier;
  event.operation = ConSanSyncOperation::BarrierFull;
  event.memory_role = ConSanSyncMemoryRole::AcquireRelease;
  event.confidence = ConSanSemanticConfidence::Exact;
  event.memory_role_confidence = ConSanSemanticConfidence::Exact;
  event.identity = "barrier";
  event.container_name = "kernel";
  event.text_offset = 8;
  event.file_offset = 0;
  event.size = 4;
  event.barrier_id = 0;
  event.barrier_operand_source = ConSanBarrierSite::OperandSource::Immediate;
  event.barrier_scope = ConSanBarrierSite::Scope::Workgroup;
  event.execution_owners.push_back({});
  ConSanSyncSequence sequence;
  sequence.kind = ConSanSyncSequenceKind::Barrier;
  sequence.operation = ConSanSyncOperation::BarrierFull;
  sequence.memory_role = ConSanSyncMemoryRole::AcquireRelease;
  sequence.confidence = ConSanSemanticConfidence::Exact;
  sequence.memory_role_confidence = ConSanSemanticConfidence::Exact;
  sequence.identity = "sequence";
  sequence.container_name = "kernel";
  sequence.begin_text_offset = 8;
  sequence.end_text_offset = 12;
  sequence.basic_block_index = 0;
  SemanticSiteId member = event.semantic_id;
  member.domain = ConSanSemanticSiteDomain::SynchronizationSequenceMember;
  sequence.member_semantic_ids.push_back(member);
  sequence.member_event_identities.push_back(event.identity);
  sequence.barrier_id = 0;
  sequence.barrier_operand_source = ConSanBarrierSite::OperandSource::Immediate;
  sequence.barrier_scope = ConSanBarrierSite::Scope::Workgroup;
  sequence.execution_owners.push_back({});
  SynchronizationInventoryBuildView synchronization = builder.synchronization();
  synchronization.sync_events.push_back(std::move(event));
  synchronization.sync_sequences.push_back(std::move(sequence));
  const ConSanBarrierPolicyResult policy = plan_consan_barrier_observation(
      builder.view(), {.engine = ConSanCapabilityEngine::RecordReplay,
                       .tracking_enabled = true,
                       .container_filter = {}});
  ASSERT_TRUE(policy.valid());

  ConSanObservationPlan broken = policy.plan;
  broken.barrier_site_decisions.front().reason = ConSanBarrierPolicyReason::InvalidBarrierEncoding;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.barrier_site_decisions.front().semantic_site.domain =
      ConSanSemanticSiteDomain::SynchronizationSequenceMember;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.barrier_site_decisions.front().intent_ids = {{99}};
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.barrier_site_decisions.front().source_containers.clear();
  EXPECT_FALSE(broken.valid());
}

TEST(ConSanObservationPlan, AppendRebasesBothDecisionFamiliesAndIsTransactional) {
  const ConSanAccessPolicyResult access = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(access.valid());
  ConSanObservationPlan combined = access.plan;
  ConSanObservationPlan fragment = access.plan;
  fragment.site_decisions.clear();
  fragment.probe_intents.front().covered_semantic_sites.front().domain =
      ConSanSemanticSiteDomain::SynchronizationEvent;
  fragment.probe_intents.front().kind = ConSanProbeIntentKind::BarrierRecord;
  fragment.probe_intents.front().position = ConSanProbePosition::After;
  fragment.barrier_site_decisions.push_back({
      .engine = ConSanCapabilityEngine::RecordReplay,
      .semantic_site = fragment.probe_intents.front().covered_semantic_sites.front(),
      .kind = ConSanSiteDecisionKind::Admitted,
      .reason = ConSanBarrierPolicyReason::None,
      .intent_ids = {{0}},
      .source_containers = {"kernel"},
  });
  ASSERT_TRUE(fragment.valid());
  ASSERT_TRUE(combined.append(fragment));
  ASSERT_TRUE(combined.valid());
  ASSERT_EQ(combined.probe_intents.size(), 2u);
  ASSERT_EQ(combined.barrier_site_decisions.size(), 1u);
  EXPECT_EQ(combined.probe_intents[1].id, ConSanProbeIntentId{1});
  EXPECT_EQ(combined.barrier_site_decisions.front().intent_ids,
            (std::vector{ConSanProbeIntentId{1}}));

  const ConSanObservationPlan before = combined;
  fragment.engine = ConSanCapabilityEngine::Sampled;
  EXPECT_FALSE(combined.append(fragment));
  EXPECT_EQ(combined, before);
  fragment = access.plan;
  fragment.probe_intents.front().id = {99};
  EXPECT_FALSE(combined.append(fragment));
  EXPECT_EQ(combined, before);
}

TEST(ConSanObservationPlan, CoverageLedgerSeparatesPolicyFromLoweringAndCopiesPlan) {
  const ConSanAccessPolicyResult policy = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ConSanCoverageLedger ledger(policy.plan);
  ASSERT_TRUE(std::ranges::equal(ledger.site_decisions(), policy.plan.site_decisions));
  ASSERT_EQ(ledger.intent_entries().size(), 1u);
  EXPECT_EQ(ledger.intent_entries().front().lowering, ConSanLoweringOutcomeKind::Pending);
  EXPECT_FALSE(ledger.all_intents_instrumented());

  EXPECT_FALSE(ledger.set_lowering_outcome({}, ConSanLoweringOutcomeKind::Instrumented));
  EXPECT_FALSE(ledger.set_lowering_outcome({7}, ConSanLoweringOutcomeKind::Instrumented));
  EXPECT_FALSE(ledger.set_lowering_outcome({0}, ConSanLoweringOutcomeKind::Count));
  EXPECT_TRUE(ledger.set_lowering_outcome({0}, ConSanLoweringOutcomeKind::Instrumented, "placed"));
  EXPECT_TRUE(ledger.all_intents_instrumented());
  ASSERT_NE(ledger.intent_entry({0}), nullptr);
  EXPECT_EQ(ledger.intent_entry({0})->detail, "placed");

  ConSanCoverageLedger copied = ledger;
  EXPECT_TRUE(copied.set_lowering_outcome({0}, ConSanLoweringOutcomeKind::ResourceRejected));
  EXPECT_EQ(ledger.intent_entry({0})->lowering, ConSanLoweringOutcomeKind::Instrumented);
  EXPECT_EQ(copied.intent_entry({0})->lowering, ConSanLoweringOutcomeKind::ResourceRejected);
}

ConSanRuntimeStaticMapping
record_replay_runtime_mapping_for(const ConSanObservationPlan &plan,
                                  std::span<const ConSanProbeIntentId> intent_ids) {
  assert(!intent_ids.empty());
  const ConSanProbeIntent *first = plan.intent(intent_ids.front());
  assert(first != nullptr);
  ConSanStaticAccessAttribution access{
      .intent_ids = std::vector(intent_ids.begin(), intent_ids.end()),
      .original_site = first->physical_site,
      .original_semantic_sites = {},
      .execution_owner_descriptor_file_offsets = {},
      .owner_provenance_complete = false,
  };
  for (ConSanProbeIntentId id : intent_ids) {
    const ConSanProbeIntent *intent = plan.intent(id);
    assert(intent != nullptr);
    assert(intent->kind == ConSanProbeIntentKind::AccessRecord);
    assert(intent->physical_site == access.original_site);
    for (const SemanticSiteId &site : intent->covered_semantic_sites) {
      if (std::ranges::find(access.original_semantic_sites, site) ==
          access.original_semantic_sites.end()) {
        access.original_semantic_sites.push_back(site);
      }
    }
  }
  ConSanRuntimeStaticMapping mapping;
  mapping.record_replay_accesses.push_back({.access = std::move(access)});
  return mapping;
}

TEST(ConSanObservationPlan, CommittedLoweringBindsSeveralIntentsToOneLocation) {
  const ConSanAccessPolicyResult policy = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ConSanObservationPlan plan = policy.plan;
  ASSERT_TRUE(plan.append(policy.plan));
  ASSERT_EQ(plan.probe_intents.size(), 2u);
  ASSERT_EQ(plan.probe_intents[0].physical_site, plan.probe_intents[1].physical_site);

  ConSanTransformArtifacts result;
  result.observation_plan = plan;
  result.coverage_ledger = ConSanCoverageLedger(plan);
  ConSanPatchInfo unrelated_patch;
  unrelated_patch.kind = ConSanPatchKind::InlineNopRewrite;
  unrelated_patch.anchor_offset = plan.probe_intents[0].physical_site.original_text_offset;
  result.patches.push_back(std::move(unrelated_patch));
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering, ConSanLoweringOutcomeKind::Pending)
      << "patch telemetry must not publish semantic coverage";
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering, ConSanLoweringOutcomeKind::Pending);

  const std::array intent_ids = {ConSanProbeIntentId{0}, ConSanProbeIntentId{1}};
  const std::array locations = {ConSanCommittedLoweringLocation{
      .original_site = plan.probe_intents[0].physical_site,
      .emitted_text_offset = 0x200,
      .emitted_size = 16,
      .relocated_guest_text_offset = 0x208,
  }};
  auto commit = make_consan_committed_lowering(plan, intent_ids, locations,
                                               ConSanLoweringOutcomeKind::Instrumented, "coalesced",
                                               record_replay_runtime_mapping_for(plan, intent_ids));
  ASSERT_TRUE(commit);
  ASSERT_TRUE(result.publish_lowering_commit(std::move(*commit)));
  ASSERT_EQ(result.committed_lowerings.size(), 1u);
  EXPECT_EQ(result.committed_lowerings.front().intent_ids,
            (std::vector{ConSanProbeIntentId{0}, ConSanProbeIntentId{1}}));
  EXPECT_EQ(result.committed_lowerings.front().original_physical_sites.size(), 1u);
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering,
            ConSanLoweringOutcomeKind::Instrumented);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering,
            ConSanLoweringOutcomeKind::Instrumented);
}

TEST(ConSanObservationPlan, CommittedLoweringRetainsEveryLocationInASequence) {
  const ConSanAccessPolicyResult policy = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  const PhysicalSiteId original = policy.plan.probe_intents.front().physical_site;
  const std::array intent_ids = {ConSanProbeIntentId{0}};
  const std::array locations = {
      ConSanCommittedLoweringLocation{
          .original_site = original,
          .emitted_text_offset = 0x200,
          .emitted_size = 8,
          .relocated_guest_text_offset = std::nullopt,
      },
      ConSanCommittedLoweringLocation{
          .original_site = original,
          .emitted_text_offset = 0x300,
          .emitted_size = 12,
          .relocated_guest_text_offset = 0x308,
      },
  };
  const auto commit = make_consan_committed_lowering(
      policy.plan, intent_ids, locations, ConSanLoweringOutcomeKind::Instrumented, {},
      record_replay_runtime_mapping_for(policy.plan, intent_ids));
  ASSERT_TRUE(commit);
  EXPECT_EQ(commit->locations, (std::vector(locations.begin(), locations.end())));
}

TEST(ConSanObservationPlan, CommittedLoweringPublishesTypedRuntimeMappingTransactionally) {
  const ConSanAccessPolicyResult policy = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  const ConSanProbeIntent &intent = policy.plan.probe_intents.front();
  const std::array intent_ids = {intent.id};
  const std::array locations = {ConSanCommittedLoweringLocation{
      .original_site = intent.physical_site,
      .emitted_text_offset = 0x200,
      .emitted_size = 16,
      .relocated_guest_text_offset = 0x208,
  }};
  ConSanRuntimeStaticMapping mapping;
  mapping.record_replay_accesses.push_back({
      .access =
          {
              .intent_ids = {intent.id},
              .original_site = intent.physical_site,
              .original_semantic_sites = intent.covered_semantic_sites,
              .execution_owner_descriptor_file_offsets = {192u},
              .owner_provenance_complete = true,
          },
  });
  const ConSanRuntimeStaticMapping expected_mapping = mapping;
  EXPECT_FALSE(make_consan_committed_lowering(policy.plan, intent_ids, locations,
                                              ConSanLoweringOutcomeKind::Instrumented));
  auto commit = make_consan_committed_lowering(policy.plan, intent_ids, locations,
                                               ConSanLoweringOutcomeKind::Instrumented, "mapped",
                                               std::move(mapping));
  ASSERT_TRUE(commit);

  ConSanTransformArtifacts result;
  result.observation_plan = policy.plan;
  result.coverage_ledger = ConSanCoverageLedger(policy.plan);
  ASSERT_TRUE(result.publish_lowering_commit(std::move(*commit)));
  EXPECT_EQ(result.runtime_static_mapping, expected_mapping);
  EXPECT_TRUE(result.runtime_static_mapping_matches_commits());
  ASSERT_EQ(result.committed_lowerings.size(), 1u);
  EXPECT_EQ(result.committed_lowerings.front().runtime_mapping, expected_mapping);

  result.runtime_static_mapping = {};
  EXPECT_FALSE(result.runtime_static_mapping_matches_commits());
  result.runtime_static_mapping = expected_mapping;

  ConSanRuntimeStaticMapping malformed = expected_mapping;
  malformed.record_replay_accesses.front().access.original_site.original_text_offset += 4u;
  EXPECT_FALSE(make_consan_committed_lowering(policy.plan, intent_ids, locations,
                                              ConSanLoweringOutcomeKind::Instrumented, {},
                                              std::move(malformed)));
  EXPECT_FALSE(make_consan_committed_lowering(
      policy.plan, intent_ids, std::span<const ConSanCommittedLoweringLocation>{},
      ConSanLoweringOutcomeKind::PlacementRejected, {}, expected_mapping));
}

TEST(ConSanObservationPlan, CommittedLoweringRejectsExactlyItsBoundIntents) {
  const ConSanAccessPolicyResult policy = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ConSanObservationPlan plan = policy.plan;
  ASSERT_TRUE(plan.append(policy.plan));
  ConSanTransformArtifacts result;
  result.observation_plan = plan;
  result.coverage_ledger = ConSanCoverageLedger(plan);

  const std::array rejected_id = {ConSanProbeIntentId{1}};
  auto rejection = make_consan_committed_lowering(
      plan, rejected_id, std::span<const ConSanCommittedLoweringLocation>{},
      ConSanLoweringOutcomeKind::ResourceRejected, "no scratch registers",
      ConSanRuntimeStaticMapping{}, ConSanRegisterPlanReason::NoLegalWindow);
  ASSERT_TRUE(rejection);
  ASSERT_TRUE(result.publish_lowering_commit(std::move(*rejection)));
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering, ConSanLoweringOutcomeKind::Pending);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering,
            ConSanLoweringOutcomeKind::ResourceRejected);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->resource_rejection_reason,
            ConSanRegisterPlanReason::NoLegalWindow);

  EXPECT_FALSE(make_consan_committed_lowering(
      plan, rejected_id, std::span<const ConSanCommittedLoweringLocation>{},
      ConSanLoweringOutcomeKind::PlacementRejected, "wrong domain", ConSanRuntimeStaticMapping{},
      ConSanRegisterPlanReason::NoLegalWindow));

  const std::array both_ids = {ConSanProbeIntentId{0}, ConSanProbeIntentId{1}};
  auto stale = make_consan_committed_lowering(plan, both_ids,
                                              std::span<const ConSanCommittedLoweringLocation>{},
                                              ConSanLoweringOutcomeKind::PlacementRejected);
  ASSERT_TRUE(stale);
  EXPECT_FALSE(result.publish_lowering_commit(std::move(*stale)));
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering, ConSanLoweringOutcomeKind::Pending)
      << "a mixed stale transaction must not partially update the ledger";
  EXPECT_EQ(result.committed_lowerings.size(), 1u);
}

TEST(ConSanObservationPlan, CommittedLoweringBatchPublicationIsTransactional) {
  const ConSanAccessPolicyResult policy = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ConSanObservationPlan plan = policy.plan;
  ASSERT_TRUE(plan.append(policy.plan));
  ConSanTransformArtifacts result;
  result.observation_plan = plan;
  result.coverage_ledger = ConSanCoverageLedger(plan);

  const std::array first_id = {ConSanProbeIntentId{0}};
  const std::array second_id = {ConSanProbeIntentId{1}};
  auto first = make_consan_committed_lowering(plan, first_id,
                                              std::span<const ConSanCommittedLoweringLocation>{},
                                              ConSanLoweringOutcomeKind::ResourceRejected, "first");
  auto second = make_consan_committed_lowering(
      plan, second_id, std::span<const ConSanCommittedLoweringLocation>{},
      ConSanLoweringOutcomeKind::PlacementRejected, "second");
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  second->intent_ids.assign(first_id.begin(), first_id.end());

  std::vector commits = {std::move(*first), std::move(*second)};
  EXPECT_FALSE(result.publish_lowering_commits(std::move(commits)));
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering, ConSanLoweringOutcomeKind::Pending);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering, ConSanLoweringOutcomeKind::Pending);
  EXPECT_TRUE(result.committed_lowerings.empty());
}

TEST(ConSanObservationPlan, DiscardedImageRetractsOnlyInstrumentedLoweringCommits) {
  const ConSanAccessPolicyResult policy = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ConSanObservationPlan plan = policy.plan;
  ASSERT_TRUE(plan.append(policy.plan));
  ConSanTransformArtifacts result;
  result.observation_plan = plan;
  result.coverage_ledger = ConSanCoverageLedger(plan);

  const std::array rejected_id = {ConSanProbeIntentId{0}};
  auto rejection = make_consan_committed_lowering(
      plan, rejected_id, std::span<const ConSanCommittedLoweringLocation>{},
      ConSanLoweringOutcomeKind::ResourceRejected, "retained planning failure");
  const std::array instrumented_id = {ConSanProbeIntentId{1}};
  const std::array location = {ConSanCommittedLoweringLocation{
      .original_site = plan.probe_intents[1].physical_site,
      .emitted_text_offset = 0x200,
      .emitted_size = 16,
      .relocated_guest_text_offset = 0x208,
  }};
  auto instrumentation = make_consan_committed_lowering(
      plan, instrumented_id, location, ConSanLoweringOutcomeKind::Instrumented, {},
      record_replay_runtime_mapping_for(plan, instrumented_id));
  ASSERT_TRUE(rejection);
  ASSERT_TRUE(instrumentation);
  std::vector commits = {std::move(*rejection), std::move(*instrumentation)};
  ASSERT_TRUE(result.publish_lowering_commits(std::move(commits)));
  result.replacement.push_back(0u);
  result.patches.emplace_back();

  result.discard_candidate_modification();

  EXPECT_TRUE(result.replacement.empty());
  EXPECT_TRUE(result.patches.empty());
  ASSERT_EQ(result.committed_lowerings.size(), 1u);
  EXPECT_EQ(result.committed_lowerings.front().outcome,
            ConSanLoweringOutcomeKind::ResourceRejected);
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering,
            ConSanLoweringOutcomeKind::ResourceRejected);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering, ConSanLoweringOutcomeKind::Pending);
}

TEST(ConSanObservationPlan, CoverageLedgerOwnsBarrierDecisionsAlongsideAccessDecisions) {
  const ConSanAccessPolicyResult access = plan_consan_access_observation(
      one_native_access_inventory(), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(access.valid());
  ConSanObservationPlan plan = access.plan;
  ConSanObservationPlan barrier_fragment{
      .engine = ConSanCapabilityEngine::RecordReplay,
      .site_decisions = {},
      .barrier_site_decisions = {{
          .engine = ConSanCapabilityEngine::RecordReplay,
          .semantic_site =
              {
                  .physical = plan.probe_intents.front().physical_site,
                  .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
                  .member_ordinal = 0,
                  .range_ordinal = 0,
              },
          .kind = ConSanSiteDecisionKind::Admitted,
          .reason = ConSanBarrierPolicyReason::None,
          .intent_ids = {{0}},
          .source_containers = {"kernel"},
      }},
      .atomic_site_decisions = {},
      .fence_site_decisions = {},
      .probe_intents = {{
          .id = {0},
          .engine = ConSanCapabilityEngine::RecordReplay,
          .physical_site = plan.probe_intents.front().physical_site,
          .covered_semantic_sites = {{
              .physical = plan.probe_intents.front().physical_site,
              .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
              .member_ordinal = 0,
              .range_ordinal = 0,
          }},
          .kind = ConSanProbeIntentKind::BarrierRecord,
          .position = ConSanProbePosition::After,
          .synchronization_association = std::nullopt,
          .dynamic_result = ConSanDynamicResultRequirement::None,
      }},
  };
  ASSERT_TRUE(barrier_fragment.valid());
  ASSERT_TRUE(plan.append(barrier_fragment));
  ConSanCoverageLedger ledger(plan);
  EXPECT_TRUE(std::ranges::equal(ledger.site_decisions(), plan.site_decisions));
  EXPECT_TRUE(std::ranges::equal(ledger.barrier_site_decisions(), plan.barrier_site_decisions));
  ASSERT_EQ(ledger.intent_entries().size(), 2u);
  EXPECT_TRUE(ledger.set_lowering_outcome({1}, ConSanLoweringOutcomeKind::Instrumented));
  EXPECT_EQ(ledger.intent_entry({1})->intent.kind, ConSanProbeIntentKind::BarrierRecord);
}

TEST(ConSanAccessPolicy, AllFourEnginesMapOneAccessToTheirOwnEvidenceIntent) {
  constexpr std::array expected = {
      std::pair{ConSanCapabilityEngine::SuperCollider,
                ConSanProbeIntentKind::RedundantAccessObservation},
      std::pair{ConSanCapabilityEngine::RecordReplay, ConSanProbeIntentKind::AccessRecord},
      std::pair{ConSanCapabilityEngine::Sampled, ConSanProbeIntentKind::SampledAccess},
      std::pair{ConSanCapabilityEngine::InlineShadow, ConSanProbeIntentKind::ExactShadowAccess},
  };
  const ProgramInventory inventory = one_native_access_inventory();
  for (const auto &[engine, expected_kind] : expected) {
    SCOPED_TRACE(consan_capability_engine_name(engine));
    const ConSanAccessPolicyResult policy =
        plan_consan_access_observation(inventory, policy_request(engine));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.site_decisions.size(), 1u);
    ASSERT_EQ(policy.plan.probe_intents.size(), 1u);
    EXPECT_EQ(policy.plan.site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);
    EXPECT_EQ(policy.plan.site_decisions.front().reason, ConSanAccessPolicyReason::None);
    EXPECT_EQ(policy.plan.probe_intents.front().kind, expected_kind);
    EXPECT_EQ(policy.plan.probe_intents.front().position, ConSanProbePosition::Before);
  }
}

TEST(ConSanAccessPolicy, TwoRangeAccessHasTwoDecisionsAndOnePhysicalIntent) {
  AccessInventoryInput input;
  input.bytes[32] = 2;
  input.bytes[33] = 5;
  ConSanKernelInfo kernel = make_policy_kernel();
  ConSanAccessInventorySite site = make_policy_lds_site("ds_store_2addr_b32");
  site.decoded_width_bits = 64;
  stage_policy_access(input, kernel, std::move(site));
  input.kernels.push_back(std::move(kernel));
  const ConSanAccessPolicyResult policy =
      plan_consan_access_observation(build_policy_inventory(std::move(input)),
                                     policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.site_decisions.size(), 2u);
  ASSERT_EQ(policy.plan.probe_intents.size(), 1u);
  EXPECT_EQ(policy.plan.site_decisions[0].intent_ids, policy.plan.site_decisions[1].intent_ids);
  EXPECT_EQ(policy.plan.site_decisions[0].semantic_site.range_ordinal, 0u);
  EXPECT_EQ(policy.plan.site_decisions[1].semantic_site.range_ordinal, 1u);
  EXPECT_EQ(policy.plan.probe_intents.front().covered_semantic_sites.size(), 2u);
}

TEST(ConSanAccessPolicy, FamilySwitchesAndFlatProvenanceAreExplicitNotApplicableDecisions) {
  AccessInventoryInput input;
  ConSanKernelInfo kernel = make_policy_kernel();
  stage_policy_access(input, kernel, make_policy_lds_site());
  stage_policy_access(input, kernel, make_policy_flat_site(ConSanFlatAddressSpaceHint::MaybeGroup));
  stage_policy_access(input, kernel, make_policy_flat_site(ConSanFlatAddressSpaceHint::Global, 80));
  input.kernels.push_back(std::move(kernel));
  const ProgramInventory inventory = build_policy_inventory(std::move(input));

  ConSanAccessPolicyRequest request = policy_request(ConSanCapabilityEngine::RecordReplay);
  request.native_lds_enabled = false;
  request.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  const ConSanAccessPolicyResult strict = plan_consan_access_observation(inventory, request);
  ASSERT_TRUE(strict.valid());
  ASSERT_EQ(strict.plan.site_decisions.size(), 3u);
  EXPECT_EQ(strict.plan.site_decisions[0].reason, ConSanAccessPolicyReason::AccessFamilyDisabled);
  EXPECT_EQ(strict.plan.site_decisions[1].reason,
            ConSanAccessPolicyReason::FlatProvenancePolicyExcluded);
  EXPECT_EQ(strict.plan.site_decisions[2].reason, ConSanAccessPolicyReason::NonGroupAddressSpace);
  EXPECT_TRUE(strict.plan.probe_intents.empty());

  request.native_lds_enabled = true;
  request.flat_provenance_mode = ConSanFlatProvenanceMode::Likely;
  const ConSanAccessPolicyResult likely = plan_consan_access_observation(inventory, request);
  ASSERT_TRUE(likely.valid());
  EXPECT_EQ(std::ranges::count(likely.plan.site_decisions, ConSanSiteDecisionKind::Admitted,
                               &ConSanSiteDecision::kind),
            2u);
  EXPECT_EQ(likely.plan.probe_intents.size(), 2u);
}

TEST(ConSanAccessPolicy, InventoryLimitationsBecomeTypedUnsupportedDecisions) {
  AccessInventoryInput input;
  ConSanKernelInfo kernel = make_policy_kernel();
  ConSanAccessInventorySite invalid_size = make_policy_lds_site("ds_store_b32", 16, 16);
  invalid_size.instruction_size = 0;
  stage_policy_access(input, kernel, invalid_size);
  ConSanAccessInventorySite invalid_width = make_policy_lds_site("ds_store_b32", 32, 32);
  invalid_width.decoded_width_bits = 0;
  stage_policy_access(input, kernel, invalid_width);
  ConSanAccessInventorySite missing_address = make_policy_lds_site("ds_store_b32", 48, 48);
  missing_address.operands.address_vgpr.reset();
  stage_policy_access(input, kernel, missing_address);
  ConSanAccessInventorySite unavailable_ranges =
      make_policy_lds_site("ds_store_2addr_b32", 64, 255);
  unavailable_ranges.decoded_width_bits = 64;
  stage_policy_access(input, kernel, unavailable_ranges);
  input.kernels.push_back(std::move(kernel));

  const ConSanAccessPolicyResult policy =
      plan_consan_access_observation(build_policy_inventory(std::move(input)),
                                     policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.site_decisions.size(), 4u);
  EXPECT_EQ(policy.plan.site_decisions[0].reason, ConSanAccessPolicyReason::InvalidInstructionSize);
  EXPECT_EQ(policy.plan.site_decisions[1].reason, ConSanAccessPolicyReason::InvalidAccessWidth);
  EXPECT_EQ(policy.plan.site_decisions[2].reason, ConSanAccessPolicyReason::MissingAddressOperand);
  EXPECT_EQ(policy.plan.site_decisions[3].reason,
            ConSanAccessPolicyReason::RangeEncodingUnavailable);
  EXPECT_TRUE(std::ranges::all_of(policy.plan.site_decisions, [](const auto &decision) {
    return decision.kind == ConSanSiteDecisionKind::Unsupported && decision.intent_ids.empty();
  }));
}

TEST(ConSanAccessPolicy, IdenticalAliasesCoalesceButConflictingAliasesFailClosed) {
  AccessInventoryInput input;
  ConSanFunctionInfo function;
  function.name = "function_alias";
  function.entry_text_offset = 0;
  ConSanAccessInventorySite first_access = make_policy_lds_site();
  stage_policy_access(input, function, std::move(first_access));
  input.functions.push_back(std::move(function));
  ConSanFunctionInfo second_function;
  second_function.name = "second_function_alias";
  second_function.entry_text_offset = 0;
  ConSanAccessInventorySite second_access = make_policy_lds_site();
  stage_policy_access(input, second_function, std::move(second_access));
  input.functions.push_back(std::move(second_function));
  const ConSanAccessPolicyResult coalesced = plan_consan_access_observation(
      build_policy_inventory(input), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(coalesced.valid());
  ASSERT_EQ(coalesced.plan.site_decisions.size(), 1u);
  ASSERT_EQ(coalesced.plan.probe_intents.size(), 1u);
  EXPECT_EQ(coalesced.plan.site_decisions.front().source_containers,
            (std::vector<std::string>{"function_alias", "second_function_alias"}));

  input.accesses.back().decoded_width_bits = 64;
  const ConSanAccessPolicyResult conflicting =
      plan_consan_access_observation(build_policy_inventory(std::move(input)),
                                     policy_request(ConSanCapabilityEngine::RecordReplay));
  EXPECT_FALSE(conflicting.valid());
  EXPECT_EQ(conflicting.errors,
            (std::vector{ConSanAccessPolicyReason::ConflictingPhysicalAliases}));
  ASSERT_EQ(conflicting.plan.site_decisions.size(), 1u);
  EXPECT_EQ(conflicting.plan.site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(conflicting.plan.site_decisions.front().reason,
            ConSanAccessPolicyReason::ConflictingPhysicalAliases);
  EXPECT_TRUE(conflicting.plan.probe_intents.empty());
}

TEST(ConSanAccessPolicy, FilterAndSynchronizationReservationRemainVisibleOutsideDenominator) {
  const ProgramInventory inventory = one_native_access_inventory();
  ConSanAccessPolicyRequest filtered = policy_request(ConSanCapabilityEngine::Sampled);
  filtered.container_filter = "different_kernel";
  const ConSanAccessPolicyResult filter_result =
      plan_consan_access_observation(inventory, filtered);
  ASSERT_TRUE(filter_result.valid());
  EXPECT_EQ(filter_result.plan.site_decisions.front().reason,
            ConSanAccessPolicyReason::ContainerFilterExcluded);

  const PhysicalSiteId reserved = inventory.access_sites().front().physical_id;
  ConSanAccessPolicyRequest synchronization = policy_request(ConSanCapabilityEngine::Sampled);
  synchronization.reserved_for_synchronization = std::span(&reserved, 1);
  const ConSanAccessPolicyResult sync_result =
      plan_consan_access_observation(inventory, synchronization);
  ASSERT_TRUE(sync_result.valid());
  EXPECT_EQ(sync_result.plan.site_decisions.front().reason,
            ConSanAccessPolicyReason::ReservedForSynchronizationPolicy);
}

TEST(ConSanAccessPolicy, UnsupportedMnemonicAndTargetCapabilityFailAtPolicyBoundary) {
  const ConSanAccessPolicyResult mnemonic =
      plan_consan_access_observation(one_native_access_inventory("ds_unknown_b32"),
                                     policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(mnemonic.valid());
  EXPECT_EQ(mnemonic.plan.site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(mnemonic.plan.site_decisions.front().reason,
            ConSanAccessPolicyReason::UnsupportedMnemonic);

  AccessInventoryInput unknown_target;
  unknown_target.arch = ROCJITSU_CODE_ARCH_INVALID;
  unknown_target.target = ROCJITSU_CODE_TARGET_INVALID;
  ConSanKernelInfo kernel = make_policy_kernel();
  stage_policy_access(unknown_target, kernel, make_policy_lds_site());
  unknown_target.kernels.push_back(std::move(kernel));
  const ConSanAccessPolicyResult target =
      plan_consan_access_observation(build_policy_inventory(std::move(unknown_target)),
                                     policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(target.valid());
  EXPECT_EQ(target.plan.site_decisions.front().kind, ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(target.plan.site_decisions.front().reason,
            ConSanAccessPolicyReason::TargetCapabilityUnavailable);
}

TEST(ConSanAccessPolicy, RelaxedLdsAtomicAccessUsesTargetCapabilityAndExactMnemonic) {
  AccessInventoryInput input;
  input.arch = ROCJITSU_CODE_ARCH_CDNA4;
  input.target = ROCJITSU_CODE_TARGET_GFX950;
  ConSanKernelInfo kernel = make_policy_kernel();
  ConSanAccessInventorySite atomic = make_policy_lds_site("ds_add_u32");
  atomic.kind = ConSanLdsAccessKind::Atomic;
  stage_policy_access(input, kernel, atomic);
  input.kernels.push_back(std::move(kernel));
  const ProgramInventory inventory = build_policy_inventory(std::move(input));

  const ConSanAccessPolicyResult moi = plan_consan_access_observation(
      inventory, policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(moi.valid());
  EXPECT_EQ(moi.plan.site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);
  const ConSanAccessPolicyResult supercollider = plan_consan_access_observation(
      inventory, policy_request(ConSanCapabilityEngine::SuperCollider));
  ASSERT_TRUE(supercollider.valid());
  EXPECT_EQ(supercollider.plan.site_decisions.front().kind, ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(supercollider.plan.site_decisions.front().reason,
            ConSanAccessPolicyReason::OperationKindExcluded);
}

TEST(ConSanObservationPolicy, OneAuthorityAssemblesPlanAndInitialLedgerForEveryEngine) {
  const ProgramInventory inventory = one_native_access_inventory();
  for (const ConSanCapabilityEngine engine : kConSanCapabilityEngines) {
    const bool supercollider = engine == ConSanCapabilityEngine::SuperCollider;
    const ConSanObservationProduct product = assemble_consan_observation_product(
        inventory, {.engine = engine,
                    .native_lds_enabled = true,
                    .group_flat_enabled = true,
                    .flat_provenance_mode = ConSanFlatProvenanceMode::Likely,
                    .barrier_tracking_enabled = true,
                    .include_atomic_fence_policy = !supercollider,
                    .atomic_fence_tracking_enabled = !supercollider,
                    .container_filter = {},
                    .reserved_for_synchronization = {}});
    SCOPED_TRACE(consan_capability_engine_name(engine));
    ASSERT_TRUE(product.valid());
    EXPECT_EQ(product.plan.engine, engine);
    EXPECT_EQ(product.initial_coverage, ConSanCoverageLedger(product.plan));
    EXPECT_TRUE(product.barrier_fragment_appended);
    EXPECT_EQ(product.atomic_fence_fragment_required, !supercollider);
    EXPECT_EQ(product.atomic_fence_fragment_appended, !supercollider);
    ASSERT_EQ(product.plan.probe_intents.size(), 1u);
    ASSERT_EQ(product.initial_coverage.intent_entries().size(), 1u);
    EXPECT_EQ(product.initial_coverage.intent_entries().front().intent,
              product.plan.probe_intents.front());
  }
}

TEST(ConSanObservationPolicy, TypedRequestAssemblyMatchesTheExplicitPolicyContractForEveryEngine) {
  const ProgramInventory inventory = one_native_access_inventory();
  for (const ConSanCapabilityEngine engine : kConSanCapabilityEngines) {
    const bool supercollider = engine == ConSanCapabilityEngine::SuperCollider;
    ConSanRequest request;
    request.flavor = supercollider ? ConSanFlavor::SuperCollider : ConSanFlavor::Moi;
    request.probe_lds_check_trap = true;
    request.probe_flat_check_trap = true;
    request.moi_track_barriers = true;
    request.moi_track_atomics = true;
    switch (engine) {
    case ConSanCapabilityEngine::SuperCollider:
      break;
    case ConSanCapabilityEngine::RecordReplay:
      request.moi_engine = ConSanMoiEngine::RecordReplay;
      break;
    case ConSanCapabilityEngine::Sampled:
      request.moi_engine = ConSanMoiEngine::Sampled;
      break;
    case ConSanCapabilityEngine::InlineShadow:
      request.moi_engine = ConSanMoiEngine::InlineShadow;
      break;
    case ConSanCapabilityEngine::Count:
      FAIL() << "sentinel engine is not iterable";
      continue;
    }

    const ConSanObservationProduct typed =
        assemble_consan_observation_product(inventory, request, ConSanDebugOverrides{});
    const ConSanObservationProduct explicit_product = assemble_consan_observation_product(
        inventory, {.engine = engine,
                    .native_lds_enabled = true,
                    .group_flat_enabled = true,
                    .flat_provenance_mode = ConSanFlatProvenanceMode::Likely,
                    .barrier_tracking_enabled = true,
                    .include_atomic_fence_policy = !supercollider,
                    .atomic_fence_tracking_enabled = !supercollider,
                    .container_filter = {},
                    .reserved_for_synchronization = {}});
    SCOPED_TRACE(consan_capability_engine_name(engine));
    ASSERT_TRUE(typed.valid());
    EXPECT_EQ(typed, explicit_product);
  }
}

TEST(ConSanObservationPolicy, ConflictingAliasesFailInTheAssembledProduct) {
  AccessInventoryInput input;
  ConSanFunctionInfo first;
  first.name = "first_alias";
  first.entry_text_offset = 0;
  stage_policy_access(input, first, make_policy_lds_site());
  input.functions.push_back(std::move(first));
  ConSanFunctionInfo second;
  second.name = "second_alias";
  second.entry_text_offset = 0;
  ConSanAccessInventorySite conflicting = make_policy_lds_site();
  conflicting.decoded_width_bits = 64;
  stage_policy_access(input, second, std::move(conflicting));
  input.functions.push_back(std::move(second));

  const ConSanObservationProduct product =
      assemble_consan_observation_product(build_policy_inventory(std::move(input)),
                                          {.engine = ConSanCapabilityEngine::RecordReplay,
                                           .native_lds_enabled = true,
                                           .group_flat_enabled = true,
                                           .flat_provenance_mode = ConSanFlatProvenanceMode::Likely,
                                           .barrier_tracking_enabled = true,
                                           .include_atomic_fence_policy = true,
                                           .atomic_fence_tracking_enabled = true,
                                           .container_filter = {},
                                           .reserved_for_synchronization = {}});
  EXPECT_FALSE(product.valid());
  EXPECT_EQ(product.access_errors,
            (std::vector{ConSanAccessPolicyReason::ConflictingPhysicalAliases}));
  ASSERT_EQ(product.diagnostics.size(), 1u);
  EXPECT_EQ(product.diagnostics.front(),
            "ConSan MOI physical access at original text offset 32 was decoded inconsistently "
            "through aliases 'first_alias', 'second_alias'");
  EXPECT_EQ(product.initial_coverage, ConSanCoverageLedger(product.plan));
}

TEST(ConSanAccessPolicy, PolicyIsDeterministicAndDoesNotMutatePublishedInventory) {
  const ProgramInventory inventory = one_native_access_inventory();
  const std::vector<ConSanAccessInventorySite> before(inventory.access_sites().begin(),
                                                      inventory.access_sites().end());
  const ConSanAccessPolicyRequest request = policy_request(ConSanCapabilityEngine::InlineShadow);
  const ConSanAccessPolicyResult first = plan_consan_access_observation(inventory, request);
  const ConSanAccessPolicyResult second = plan_consan_access_observation(inventory, request);
  EXPECT_EQ(first, second);
  EXPECT_TRUE(first.valid());
  EXPECT_TRUE(std::ranges::equal(inventory.access_sites(), before));
}

} // namespace
} // namespace rocjitsu

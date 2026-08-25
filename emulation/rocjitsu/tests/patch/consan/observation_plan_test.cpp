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
};

ConSanLdsSite make_policy_lds_site(std::string mnemonic = "ds_store_b32", uint64_t text_offset = 32,
                                   uint64_t file_offset = 32) {
  ConSanLdsSite site;
  site.kind = ConSanLdsAccessKind::Write;
  site.supported_mvp = true;
  site.text_offset = text_offset;
  site.file_offset = file_offset;
  site.size = 8;
  site.width_bits = 32;
  site.addr_vgpr = 3;
  site.data_vgpr = 4;
  site.mnemonic = std::move(mnemonic);
  return site;
}

ConSanFlatSite make_policy_flat_site(ConSanFlatAddressSpaceHint hint, uint64_t text_offset = 64) {
  ConSanFlatSite site;
  site.kind = ConSanLdsAccessKind::Write;
  site.text_offset = text_offset;
  site.file_offset = text_offset;
  site.size = 12;
  site.width_bits = 32;
  site.addr_vgpr = 5;
  site.data_vgpr = 6;
  site.raw_saddr = 0;
  site.raw_scale_offset = true;
  site.raw_ioffset = 0;
  site.address_space_hint = hint;
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
  builder.rebuild_access_inventory(input.bytes);
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
  kernel.lds_sites.push_back(make_policy_lds_site(std::move(mnemonic)));
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
  expect_observation_enum_contract(kConSanLaneMaskPolicies, ConSanLaneMaskPolicy::Count,
                                   consan_lane_mask_policy_name, "invalid-lane-mask-policy");
  expect_observation_enum_contract(kConSanProbeRequirements, ConSanProbeRequirement::Count,
                                   consan_probe_requirement_name, "invalid-probe-requirement");
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
  EXPECT_FALSE(ledger.all_required_intents_instrumented());

  EXPECT_FALSE(ledger.set_lowering_outcome({}, ConSanLoweringOutcomeKind::Instrumented));
  EXPECT_FALSE(ledger.set_lowering_outcome({7}, ConSanLoweringOutcomeKind::Instrumented));
  EXPECT_FALSE(ledger.set_lowering_outcome({0}, ConSanLoweringOutcomeKind::Count));
  EXPECT_TRUE(ledger.set_lowering_outcome({0}, ConSanLoweringOutcomeKind::Instrumented, "placed"));
  EXPECT_TRUE(ledger.all_required_intents_instrumented());
  ASSERT_NE(ledger.intent_entry({0}), nullptr);
  EXPECT_EQ(ledger.intent_entry({0})->detail, "placed");

  ConSanCoverageLedger copied = ledger;
  EXPECT_TRUE(copied.set_lowering_outcome({0}, ConSanLoweringOutcomeKind::ResourceRejected));
  EXPECT_EQ(ledger.intent_entry({0})->lowering, ConSanLoweringOutcomeKind::Instrumented);
  EXPECT_EQ(copied.intent_entry({0})->lowering, ConSanLoweringOutcomeKind::ResourceRejected);
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
          .lane_mask = ConSanLaneMaskPolicy::ActiveExecutionMask,
          .requirement = ConSanProbeRequirement::Required,
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

TEST(ConSanObservationPlan, PhysicalOutcomeAdapterUpdatesOnlyMatchingCoalescedIntents) {
  AccessInventoryInput input;
  ConSanKernelInfo kernel = make_policy_kernel();
  kernel.lds_sites.push_back(make_policy_lds_site("ds_store_b32", 32, 32));
  kernel.lds_sites.push_back(make_policy_lds_site("ds_store_b32", 48, 48));
  input.kernels.push_back(std::move(kernel));
  const ConSanAccessPolicyResult policy =
      plan_consan_access_observation(build_policy_inventory(std::move(input)),
                                     policy_request(ConSanCapabilityEngine::InlineShadow));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.probe_intents.size(), 2u);
  ConSanCoverageLedger ledger(policy.plan);
  EXPECT_EQ(ledger.set_physical_lowering_outcome(policy.plan.probe_intents[1].physical_site,
                                                 ConSanLoweringOutcomeKind::PlacementRejected,
                                                 "no route"),
            1u);
  EXPECT_EQ(ledger.intent_entries()[0].lowering, ConSanLoweringOutcomeKind::Pending);
  EXPECT_EQ(ledger.intent_entries()[1].lowering, ConSanLoweringOutcomeKind::PlacementRejected);
  EXPECT_EQ(ledger.set_physical_lowering_outcome({}, ConSanLoweringOutcomeKind::Instrumented), 0u);
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
    EXPECT_EQ(policy.plan.probe_intents.front().lane_mask,
              ConSanLaneMaskPolicy::ActiveExecutionMask);
    EXPECT_EQ(policy.plan.probe_intents.front().requirement, ConSanProbeRequirement::Required);
  }
}

TEST(ConSanAccessPolicy, TwoRangeAccessHasTwoDecisionsAndOnePhysicalIntent) {
  AccessInventoryInput input;
  input.bytes[32] = 2;
  input.bytes[33] = 5;
  ConSanKernelInfo kernel = make_policy_kernel();
  ConSanLdsSite site = make_policy_lds_site("ds_store_2addr_b32");
  site.width_bits = 64;
  kernel.lds_sites.push_back(std::move(site));
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
  kernel.lds_sites.push_back(make_policy_lds_site());
  kernel.flat_sites.push_back(make_policy_flat_site(ConSanFlatAddressSpaceHint::MaybeGroup));
  kernel.flat_sites.push_back(make_policy_flat_site(ConSanFlatAddressSpaceHint::Global, 80));
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
  ConSanLdsSite invalid_size = make_policy_lds_site("ds_store_b32", 16, 16);
  invalid_size.size = 0;
  kernel.lds_sites.push_back(invalid_size);
  ConSanLdsSite invalid_width = make_policy_lds_site("ds_store_b32", 32, 32);
  invalid_width.width_bits = 0;
  kernel.lds_sites.push_back(invalid_width);
  ConSanLdsSite missing_address = make_policy_lds_site("ds_store_b32", 48, 48);
  missing_address.addr_vgpr.reset();
  kernel.lds_sites.push_back(missing_address);
  ConSanLdsSite unavailable_ranges = make_policy_lds_site("ds_store_2addr_b32", 64, 255);
  unavailable_ranges.width_bits = 64;
  kernel.lds_sites.push_back(unavailable_ranges);
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
  function.lds_sites.push_back(make_policy_lds_site());
  input.functions.push_back(std::move(function));
  ConSanFunctionInfo second_function;
  second_function.name = "second_function_alias";
  second_function.entry_text_offset = 0;
  second_function.lds_sites.push_back(make_policy_lds_site());
  input.functions.push_back(std::move(second_function));
  const ConSanAccessPolicyResult coalesced = plan_consan_access_observation(
      build_policy_inventory(input), policy_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(coalesced.valid());
  ASSERT_EQ(coalesced.plan.site_decisions.size(), 1u);
  ASSERT_EQ(coalesced.plan.probe_intents.size(), 1u);
  EXPECT_EQ(coalesced.plan.site_decisions.front().source_containers,
            (std::vector<std::string>{"function_alias", "second_function_alias"}));

  input.functions.back().lds_sites.front().width_bits = 64;
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
  kernel.lds_sites.push_back(make_policy_lds_site());
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
  ConSanLdsSite atomic = make_policy_lds_site("ds_add_u32");
  atomic.kind = ConSanLdsAccessKind::Atomic;
  kernel.lds_sites.push_back(atomic);
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

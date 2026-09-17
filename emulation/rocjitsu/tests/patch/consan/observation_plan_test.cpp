// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"

#include <concepts>
#include <set>

namespace rocjitsu::consan {
namespace {

template <typename Values, typename Enum, typename NameFunction>
void expect_observation_enum_contract(const Values &values, Enum count, NameFunction name,
                                      std::string_view invalid_name) {
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
  struct ContainerSelector {
    ProgramContainerKind kind = ProgramContainerKind::Function;
    std::string name;
  };

  std::vector<uint8_t> bytes = std::vector<uint8_t>(256, 0);
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201;
  std::vector<ProgramContainer> kernels;
  std::vector<ProgramContainer> functions;
  std::vector<ProgramSite> accesses;
  std::vector<ContainerSelector> access_containers;
};

template <typename Container>
void stage_policy_access(AccessInventoryInput &input, const Container &container,
                         ProgramSite site) {
  input.accesses.push_back(std::move(site));
  input.access_containers.push_back({.kind = container.kind, .name = container.name});
}

ProgramSite make_policy_lds_site(std::string mnemonic = "ds_store_b32", uint64_t text_offset = 32,
                                 uint64_t file_offset = 32) {
  ProgramSite site;
  site.origin = AccessOrigin::NativeLds;
  site.kind = LdsAccessKind::Write;
  site.physical_id.original_text_offset = text_offset;
  site.decoded_site().text_offset = text_offset;
  site.decoded_site().file_offset = file_offset;
  site.decoded_site().size = 8;
  site.decoded_width_bits = 32;
  site.operands.address_vgpr = 3;
  site.operands.data_vgpr = 4;
  site.decoded_site().mnemonic = std::move(mnemonic);
  return site;
}

ProgramSite make_policy_flat_site(FlatAddressSpaceHint hint, uint64_t text_offset = 64) {
  ProgramSite site;
  site.origin = AccessOrigin::Flat;
  site.kind = LdsAccessKind::Write;
  site.physical_id.original_text_offset = text_offset;
  site.decoded_site().text_offset = text_offset;
  site.decoded_site().file_offset = text_offset;
  site.decoded_site().size = 12;
  site.decoded_width_bits = 32;
  site.operands.address_vgpr = 5;
  site.operands.data_vgpr = 6;
  site.operands.raw_saddr = 0;
  site.operands.raw_scale_offset = true;
  site.operands.raw_ioffset = 0;
  site.flat_address_space_hint = hint;
  site.decoded_site().mnemonic = "flat_store_b32";
  return site;
}

ProgramContainer make_policy_kernel(std::string name = "policy_kernel") {
  ProgramContainer kernel{ProgramContainerKind::Kernel};
  kernel.name = std::move(name);
  kernel.descriptor_file_offset = 192;
  kernel.entry_text_offset = 0;
  return kernel;
}

ProgramInventory build_policy_inventory(AccessInventoryInput input) {
  ProgramInventoryBuilder builder(input.bytes);
  builder.set_code_object_facts(true, 0, input.arch, input.target);
  for (ProgramContainer &kernel : input.kernels)
    builder.add_kernel(std::move(kernel));
  for (ProgramContainer &function : input.functions)
    builder.add_function(std::move(function));
  EXPECT_EQ(input.accesses.size(), input.access_containers.size());
  for (size_t index = 0; index < input.accesses.size(); ++index) {
    ProgramSite &access = input.accesses[index];
    const AccessInventoryInput::ContainerSelector &selector = input.access_containers[index];
    const std::span<const ProgramContainer> containers =
        selector.kind == ProgramContainerKind::Kernel ? builder.kernels() : builder.functions();
    const auto container = std::ranges::find(containers, selector.name, &ProgramContainer::name);
    EXPECT_NE(container, containers.end());
    if (container != containers.end())
      access.container = container->id;
    builder.add_access_site(std::move(access));
  }
  builder.publish_decoded_accesses(input.bytes);
  return builder.view();
}

AccessPolicyRequest policy_request(Mode mode) {
  return {
      .mode = mode,
      .native_lds_enabled = true,
      .group_flat_enabled = true,
      .flat_provenance_mode = FlatProvenanceMode::Likely,
      .container_filter = {},
      .kernel_name_allowlist = {},
      .reserved_for_synchronization = {},
  };
}

ProgramInventory one_native_access_inventory(std::string mnemonic = "ds_store_b32") {
  AccessInventoryInput input;
  ProgramContainer kernel = make_policy_kernel();
  stage_policy_access(input, kernel, make_policy_lds_site(std::move(mnemonic)));
  input.kernels.push_back(std::move(kernel));
  return build_policy_inventory(std::move(input));
}

ObservationPlan one_barrier_observation_plan(PhysicalSiteId physical_site) {
  const SemanticSiteId semantic_site{
      .physical = physical_site,
      .domain = SemanticSiteDomain::SynchronizationEvent,
      .member_ordinal = 0,
      .range_ordinal = 0,
  };
  return {
      .mode = Mode::Default,
      .site_decisions = {},
      .barrier_site_decisions = {{
          .semantic_site = semantic_site,
          .kind = SiteDecisionKind::Admitted,
          .reason = BarrierPolicyReason::None,
      }},
      .atomic_site_decisions = {},
      .fence_site_decisions = {},
      .probe_intents = {{
          .id = {0},
          .mode = Mode::Default,
          .source_site = {0},
          .physical_site = physical_site,
          .covered_semantic_sites = {semantic_site},
          .kind = ProbeIntentKind::BarrierEpoch,
          .position = ProbePosition::After,
          .synchronization_association = std::nullopt,
          .dynamic_result = DynamicResultRequirement::None,
          .atomic_lowering_form = std::nullopt,
      }},
  };
}

TEST(ObservationPlan, EnumContractsAreExhaustiveNamedAndRejectInvalidValues) {
  expect_observation_enum_contract(kAtomicPolicyReasons, AtomicPolicyReason::Count,
                                   atomic_policy_reason_name, "invalid-atomic-policy-reason");
}

TEST(ObservationPlan, SynchronizationAssociationIdentityHasAnExplicitInvalidDefault) {
  EXPECT_FALSE(SynchronizationAssociationId{}.valid());
  EXPECT_TRUE(SynchronizationAssociationId{"sequence"}.valid());
  EXPECT_EQ(SynchronizationAssociationId{"sequence"}, SynchronizationAssociationId{"sequence"});
  EXPECT_NE(SynchronizationAssociationId{"sequence"},
            SynchronizationAssociationId{"other-sequence"});
  EXPECT_LT(SynchronizationAssociationId{"sequence-a"}, SynchronizationAssociationId{"sequence-b"});
}

TEST(ObservationPlan, IntentIdentifiersAreExplicitlyInvalidAndPlanLocal) {
  EXPECT_FALSE(ProbeIntentId{}.valid());
  EXPECT_TRUE(ProbeIntentId{0}.valid());
  EXPECT_NE(ProbeIntentId{0}, ProbeIntentId{1});

  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.probe_intents.size(), 1u);
  EXPECT_EQ(policy.plan.intent({0}), &policy.plan.probe_intents.front());
  EXPECT_EQ(policy.plan.intent({1}), nullptr);
  EXPECT_EQ(policy.plan.intent({}), nullptr);
}

TEST(ObservationPlan, PlanValidationRejectsEveryBrokenTypedRelationship) {
  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());

  ObservationPlan broken = policy.plan;
  broken.mode = Mode::None;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().id = {7};
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().source_site = {};
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().kind = ProbeIntentKind::Count;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().covered_semantic_sites.clear();
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.site_decisions.front().reason = AccessPolicyReason::UnsupportedMnemonic;
  EXPECT_FALSE(broken.valid());
}

TEST(ObservationPlan, BarrierDecisionValidationRejectsEveryBrokenTypedRelationship) {
  ProgramInventoryBuilder builder(std::array<uint8_t, 4>{});
  builder.set_code_object_facts(true, 0, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
  ProgramContainer kernel{ProgramContainerKind::Kernel};
  kernel.name = "kernel";
  builder.add_kernel(std::move(kernel));
  SyncEvent event;
  event.semantic_id = {
      .physical = {.code_object = builder.view().code_object_id(), .original_text_offset = 8},
      .domain = SemanticSiteDomain::SynchronizationEvent,
  };
  event.kind = SyncKind::Barrier;
  event.operation = SyncOperation::BarrierFull;
  event.memory_role = SyncMemoryRole::AcquireRelease;
  event.confidence = SemanticConfidence::Exact;
  event.memory_role_confidence = SemanticConfidence::Exact;
  event.identity = "barrier";
  event.source_site = {0};
  BarrierSite source;
  source.operation = BarrierSite::Operation::Full;
  source.text_offset = 8;
  source.file_offset = 0;
  source.size = 4;
  source.barrier_id = 0;
  source.operand_source = BarrierSite::OperandSource::Immediate;
  source.scope = BarrierSite::Scope::Workgroup;
  source.mnemonic = "s_barrier";
  builder.add_semantic_site(make_program_site(builder.kernels().front().id, std::move(source)));
  builder.program_sites().back().execution_owners.push_back({});
  SyncSequence sequence;
  sequence.kind = SyncKind::Barrier;
  sequence.operation = SyncOperation::BarrierFull;
  sequence.memory_role = SyncMemoryRole::AcquireRelease;
  sequence.confidence = SemanticConfidence::Exact;
  sequence.memory_role_confidence = SemanticConfidence::Exact;
  sequence.identity = "sequence";
  sequence.begin_text_offset = 8;
  sequence.end_text_offset = 12;
  sequence.basic_block_index = 0;
  sequence.member_event_ids.push_back({0});
  sequence.barrier_id = 0;
  sequence.barrier_operand_source = BarrierSite::OperandSource::Immediate;
  sequence.barrier_scope = BarrierSite::Scope::Workgroup;
  SynchronizationInventoryBuildView synchronization = builder.synchronization();
  synchronization.sync_events.push_back(std::move(event));
  synchronization.sync_sequences.push_back(std::move(sequence));
  const BarrierPolicyResult policy =
      plan_barrier_observation(builder.view(), {.mode = Mode::Default,
                                                .tracking_enabled = true,
                                                .container_filter = {},
                                                .kernel_name_allowlist = {}});
  ASSERT_TRUE(policy.valid());

  ObservationPlan broken = policy.plan;
  broken.barrier_site_decisions.front().reason = BarrierPolicyReason::InvalidBarrierEncoding;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.barrier_site_decisions.front().semantic_site.domain = SemanticSiteDomain::Count;
  EXPECT_FALSE(broken.valid());
}

TEST(ObservationPlan, AppendRebasesIntentsAndOwnsDecisionsTransactionally) {
  const AccessPolicyResult access =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(access.valid());
  ObservationPlan combined = access.plan;
  ObservationPlan fragment = access.plan;
  fragment.site_decisions.clear();
  fragment.probe_intents.front().covered_semantic_sites.front().domain =
      SemanticSiteDomain::SynchronizationEvent;
  fragment.probe_intents.front().kind = ProbeIntentKind::BarrierEpoch;
  fragment.probe_intents.front().position = ProbePosition::After;
  fragment.barrier_site_decisions.push_back({
      .semantic_site = fragment.probe_intents.front().covered_semantic_sites.front(),
      .kind = SiteDecisionKind::Admitted,
      .reason = BarrierPolicyReason::None,
  });
  ASSERT_TRUE(fragment.valid());
  ASSERT_TRUE(combined.append(fragment));
  ASSERT_TRUE(combined.valid());
  ASSERT_EQ(combined.probe_intents.size(), 2u);
  ASSERT_EQ(combined.barrier_site_decisions.size(), 1u);
  EXPECT_EQ(combined.probe_intents[1].id, ProbeIntentId{1});

  const ObservationPlan before = combined;
  fragment.mode = Mode::SuperCollider;
  EXPECT_FALSE(combined.append(fragment));
  EXPECT_EQ(combined, before);
  fragment = access.plan;
  fragment.probe_intents.front().id = {99};
  EXPECT_FALSE(combined.append(fragment));
  EXPECT_EQ(combined, before);
}

TEST(ObservationPlan, CoverageLedgerSolelyOwnsPlanAndJoinedLoweringState) {
  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  CoverageLedger ledger(policy.plan);
  EXPECT_EQ(ledger.observation_plan(), policy.plan);
  ASSERT_TRUE(std::ranges::equal(ledger.site_decisions(), policy.plan.site_decisions));
  ASSERT_EQ(ledger.intent_entries().size(), 1u);
  EXPECT_EQ(ledger.intent_entries().front().intent_id, policy.plan.probe_intents.front().id);
  EXPECT_EQ(ledger.intent({0}), &ledger.observation_plan().probe_intents.front());
  EXPECT_EQ(ledger.intent_entries().front().lowering, LoweringOutcomeKind::Pending);
  EXPECT_FALSE(all_intents_instrumented(ledger));

  CoverageLedger copied = ledger;
  EXPECT_FALSE(publish_test_lowering_outcome(ledger, {}, LoweringOutcomeKind::Instrumented));
  EXPECT_FALSE(publish_test_lowering_outcome(ledger, {7}, LoweringOutcomeKind::Instrumented));
  EXPECT_FALSE(publish_test_lowering_outcome(ledger, {0}, LoweringOutcomeKind::Count));
  EXPECT_TRUE(
      publish_test_lowering_outcome(ledger, {0}, LoweringOutcomeKind::Instrumented, "placed"));
  EXPECT_TRUE(all_intents_instrumented(ledger));
  ASSERT_NE(ledger.intent_entry({0}), nullptr);
  EXPECT_EQ(ledger.intent_entry({0})->detail, "placed");

  EXPECT_TRUE(publish_test_lowering_outcome(copied, {0}, LoweringOutcomeKind::ResourceRejected));
  EXPECT_EQ(ledger.intent_entry({0})->lowering, LoweringOutcomeKind::Instrumented);
  EXPECT_EQ(copied.intent_entry({0})->lowering, LoweringOutcomeKind::ResourceRejected);
}

StaticAccessMappings runtime_mapping_for(const ObservationPlan &plan,
                                         std::span<const ProbeIntentId> intent_ids) {
  assert(!intent_ids.empty());
  const ProbeIntent *first = plan.intent(intent_ids.front());
  assert(first != nullptr);
  StaticAccessAttribution access{
      .intent_ids = std::vector(intent_ids.begin(), intent_ids.end()),
      .original_site = first->physical_site,
      .original_semantic_sites = {},
      .execution_owner_kernel_ids = {},
      .owner_provenance_complete = false,
  };
  for (ProbeIntentId id : intent_ids) {
    const ProbeIntent *intent = plan.intent(id);
    assert(intent != nullptr);
    assert(intent->kind == ProbeIntentKind::Access);
    assert(intent->physical_site == access.original_site);
    for (const SemanticSiteId &site : intent->covered_semantic_sites) {
      if (std::ranges::find(access.original_semantic_sites, site) ==
          access.original_semantic_sites.end()) {
        access.original_semantic_sites.push_back(site);
      }
    }
  }
  const auto range_count = static_cast<uint32_t>(access.original_semantic_sites.size());
  return StaticAccessMappings{{.access = std::move(access),
                               .range_count = range_count,
                               .bank_count = 1u,
                               .relocated_guest_text_offset = std::nullopt,
                               .scratch_vgpr = std::nullopt}};
}

TEST(ObservationPlan, CommittedLoweringBindsSeveralIntentsToOneLocation) {
  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ObservationPlan plan = policy.plan;
  ASSERT_TRUE(plan.append(policy.plan));
  ASSERT_EQ(plan.probe_intents.size(), 2u);
  ASSERT_EQ(plan.probe_intents[0].physical_site, plan.probe_intents[1].physical_site);

  TransformArtifacts result;
  result.coverage_ledger = CoverageLedger(plan);
  PatchInfo unrelated_patch;
  unrelated_patch.kind = PatchKind::InlineBarrierNopRewrite;
  unrelated_patch.anchor_offset = plan.probe_intents[0].physical_site.original_text_offset;
  result.patches.push_back(std::move(unrelated_patch));
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering, LoweringOutcomeKind::Pending)
      << "patch telemetry must not publish semantic coverage";
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering, LoweringOutcomeKind::Pending);

  const std::array intent_ids = {ProbeIntentId{0}, ProbeIntentId{1}};
  const std::array locations = {CommittedLoweringLocation{
      .original_site = plan.probe_intents[0].physical_site,
      .emitted_text_offset = 0x200,
      .emitted_size = 16,
      .relocated_guest_text_offset = 0x208,
  }};
  auto commit =
      make_committed_lowering(plan, intent_ids, locations, LoweringOutcomeKind::Instrumented,
                              "coalesced", runtime_mapping_for(plan, intent_ids));
  ASSERT_TRUE(commit);
  ASSERT_TRUE(result.coverage_ledger.publish_lowering_commit(std::move(*commit)));
  ASSERT_EQ(result.coverage_ledger.lowering_commits().size(), 1u);
  EXPECT_EQ(result.coverage_ledger.lowering_commits().front().intent_ids,
            (std::vector{ProbeIntentId{0}, ProbeIntentId{1}}));
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering, LoweringOutcomeKind::Instrumented);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering, LoweringOutcomeKind::Instrumented);
}

TEST(ObservationPlan, CommittedLoweringRetainsEveryLocationInASequence) {
  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  const PhysicalSiteId original = policy.plan.probe_intents.front().physical_site;
  const std::array intent_ids = {ProbeIntentId{0}};
  const std::array locations = {
      CommittedLoweringLocation{
          .original_site = original,
          .emitted_text_offset = 0x200,
          .emitted_size = 8,
          .relocated_guest_text_offset = std::nullopt,
      },
      CommittedLoweringLocation{
          .original_site = original,
          .emitted_text_offset = 0x300,
          .emitted_size = 12,
          .relocated_guest_text_offset = 0x308,
      },
  };
  const auto commit =
      make_committed_lowering(policy.plan, intent_ids, locations, LoweringOutcomeKind::Instrumented,
                              {}, runtime_mapping_for(policy.plan, intent_ids));
  ASSERT_TRUE(commit);
  EXPECT_EQ(commit->locations, (std::vector(locations.begin(), locations.end())));
}

TEST(ObservationPlan, InstrumentedPatchGeometryBuildsOneLocationPairPerOriginalSite) {
  const AccessPolicyResult access =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(access.valid());
  ObservationPlan plan =
      one_barrier_observation_plan(access.plan.probe_intents.front().physical_site);
  const ObservationPlan duplicate = plan;
  ASSERT_TRUE(plan.append(duplicate));
  ASSERT_EQ(plan.probe_intents.size(), 2u);

  const std::array intent_ids = {ProbeIntentId{0}, ProbeIntentId{1}};
  const CommittedPatchGeometry patch{
      .anchor_offset = 0x200,
      .trampoline_offset = 0x300,
      .original_size = 4,
      .trampoline_size = 12,
      .relocated_guest_instruction_offset = 0x308,
  };
  const auto commit = make_instrumented_patch_lowering(plan, intent_ids, patch);
  ASSERT_TRUE(commit);
  ASSERT_EQ(commit->locations.size(), 2u);
  EXPECT_EQ(commit->locations[0], (CommittedLoweringLocation{
                                      .original_site = plan.probe_intents.front().physical_site,
                                      .emitted_text_offset = 0x200,
                                      .emitted_size = 4,
                                      .relocated_guest_text_offset = std::nullopt,
                                  }));
  EXPECT_EQ(commit->locations[1], (CommittedLoweringLocation{
                                      .original_site = plan.probe_intents.front().physical_site,
                                      .emitted_text_offset = 0x300,
                                      .emitted_size = 12,
                                      .relocated_guest_text_offset = 0x308,
                                  }));
  const std::array stale_id = {ProbeIntentId{2}};
  EXPECT_FALSE(make_instrumented_patch_lowering(plan, stale_id, patch));
}

TEST(ObservationPlan, CommittedLoweringPublishesTypedRuntimeMappingTransactionally) {
  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  const ProbeIntent &intent = policy.plan.probe_intents.front();
  const std::array intent_ids = {intent.id};
  const std::array locations = {CommittedLoweringLocation{
      .original_site = intent.physical_site,
      .emitted_text_offset = 0x200,
      .emitted_size = 16,
      .relocated_guest_text_offset = 0x208,
  }};
  StaticAccessMappings mapping = StaticAccessMappings{{
      .access =
          {
              .intent_ids = {intent.id},
              .original_site = intent.physical_site,
              .original_semantic_sites = intent.covered_semantic_sites,
              .execution_owner_kernel_ids = {{0u}},
              .owner_provenance_complete = true,
          },
      .range_count = static_cast<uint32_t>(intent.covered_semantic_sites.size()),
      .bank_count = 1u,
      .relocated_guest_text_offset = std::nullopt,
      .scratch_vgpr = std::nullopt,
  }};
  const StaticAccessMappings expected_mapping = mapping;
  EXPECT_FALSE(make_committed_lowering(policy.plan, intent_ids, locations,
                                       LoweringOutcomeKind::Instrumented));
  auto commit =
      make_committed_lowering(policy.plan, intent_ids, locations, LoweringOutcomeKind::Instrumented,
                              "mapped", std::move(mapping));
  ASSERT_TRUE(commit);

  TransformArtifacts result;
  result.coverage_ledger = CoverageLedger(policy.plan);
  CommittedLowering malformed_commit = *commit;

  malformed_commit.runtime_mapping.front().access.original_site.original_text_offset += 4u;
  CoverageLedger malformed_ledger(policy.plan);
  EXPECT_FALSE(malformed_ledger.publish_lowering_commit(std::move(malformed_commit)))
      << "the ledger must reject a runtime projection outside its bound intents";

  ASSERT_TRUE(result.coverage_ledger.publish_lowering_commit(std::move(*commit)));
  EXPECT_EQ(result.coverage_ledger.runtime_static_mapping(), expected_mapping);
  ASSERT_EQ(result.coverage_ledger.lowering_commits().size(), 1u);
  EXPECT_EQ(result.coverage_ledger.lowering_commits().front().runtime_mapping, expected_mapping);
  EXPECT_EQ(result.coverage_ledger.observation_plan(), policy.plan);

  StaticAccessMappings malformed = expected_mapping;

  malformed.front().access.original_site.original_text_offset += 4u;
  EXPECT_FALSE(make_committed_lowering(policy.plan, intent_ids, locations,
                                       LoweringOutcomeKind::Instrumented, {},
                                       std::move(malformed)));
  malformed = expected_mapping;
  malformed.front().access.execution_owner_kernel_ids.push_back({0u});
  EXPECT_FALSE(make_committed_lowering(policy.plan, intent_ids, locations,
                                       LoweringOutcomeKind::Instrumented, {}, std::move(malformed)))
      << "runtime attribution must reject duplicate semantic owner handles";
  malformed = expected_mapping;
  malformed.front().access.execution_owner_kernel_ids.front() = {};
  EXPECT_FALSE(make_committed_lowering(policy.plan, intent_ids, locations,
                                       LoweringOutcomeKind::Instrumented, {}, std::move(malformed)))
      << "runtime attribution must reject invalid semantic owner handles";
  EXPECT_FALSE(
      make_committed_lowering(policy.plan, intent_ids, std::span<const CommittedLoweringLocation>{},
                              LoweringOutcomeKind::PlacementRejected, {}, expected_mapping));
}

TEST(ObservationPlan, CommittedLoweringRejectsExactlyItsBoundIntents) {
  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ObservationPlan plan = policy.plan;
  ASSERT_TRUE(plan.append(policy.plan));
  TransformArtifacts result;
  result.coverage_ledger = CoverageLedger(plan);

  const std::array rejected_id = {ProbeIntentId{1}};
  ASSERT_TRUE(result.coverage_ledger.publish_lowering_rejection(
      rejected_id, LoweringOutcomeKind::ResourceRejected, "no scratch registers",
      RegisterPlanReason::NoLegalWindow));
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering, LoweringOutcomeKind::Pending);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering,
            LoweringOutcomeKind::ResourceRejected);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->resource_rejection_reason,
            RegisterPlanReason::NoLegalWindow);

  CoverageLedger invalid_reason(plan);
  EXPECT_FALSE(
      invalid_reason.publish_lowering_rejection(rejected_id, LoweringOutcomeKind::PlacementRejected,
                                                "wrong domain", RegisterPlanReason::NoLegalWindow));

  const std::array both_ids = {ProbeIntentId{0}, ProbeIntentId{1}};
  auto stale = make_committed_lowering(plan, both_ids, std::span<const CommittedLoweringLocation>{},
                                       LoweringOutcomeKind::PlacementRejected);
  ASSERT_TRUE(stale);
  EXPECT_FALSE(result.coverage_ledger.publish_lowering_commit(std::move(*stale)));
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering, LoweringOutcomeKind::Pending)
      << "a mixed stale transaction must not partially update the ledger";
  EXPECT_EQ(result.coverage_ledger.lowering_commits().size(), 1u);
}

TEST(ObservationPlan, PendingIntentQueryJoinsKindSiteAndCurrentLoweringState) {
  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::SuperCollider));
  ASSERT_TRUE(policy.valid());
  ObservationPlan plan = policy.plan;
  ASSERT_TRUE(plan.append(policy.plan));
  ASSERT_EQ(plan.probe_intents.size(), 2u);
  const PhysicalSiteId site = plan.probe_intents.front().physical_site;
  CoverageLedger ledger(std::move(plan));

  EXPECT_EQ(ledger.pending_intent_ids(ProbeIntentKind::RedundantAccessObservation, site),
            (std::vector<ProbeIntentId>{{0}, {1}}));
  ASSERT_NE(ledger.intent({0}), nullptr);
  EXPECT_EQ(ledger.intent_ids_covering(ledger.intent({0})->covered_semantic_sites.front()),
            (std::vector<ProbeIntentId>{{0}, {1}}));
  EXPECT_TRUE(ledger.pending_intent_ids(ProbeIntentKind::BarrierEpoch, site).empty());
  PhysicalSiteId other_site = site;
  other_site.original_text_offset += 4u;
  EXPECT_TRUE(
      ledger.pending_intent_ids(ProbeIntentKind::RedundantAccessObservation, other_site).empty());

  const std::array rejected = {ProbeIntentId{1}};
  ASSERT_TRUE(ledger.publish_lowering_rejection(rejected, LoweringOutcomeKind::PlacementRejected,
                                                "test rejection"));
  EXPECT_EQ(ledger.pending_intent_ids(ProbeIntentKind::RedundantAccessObservation, site),
            (std::vector<ProbeIntentId>{{0}}));
}

TEST(ObservationPlan, PendingIntentQueriesScaleWithAddressedSites) {
  ObservationPlan plan =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::SuperCollider))
          .plan;
  ASSERT_TRUE(plan.valid());
  ASSERT_EQ(plan.probe_intents.size(), 1u);

  constexpr uint32_t kSiteCount = 16384;
  const ProbeIntent prototype = plan.probe_intents.front();
  plan.probe_intents.reserve(kSiteCount);
  for (uint32_t ordinal = 1; ordinal < kSiteCount; ++ordinal) {
    ProbeIntent intent = prototype;
    intent.id = {ordinal};
    intent.source_site = {ordinal};
    intent.physical_site.original_text_offset += 4u * ordinal;
    for (SemanticSiteId &semantic_site : intent.covered_semantic_sites) {
      semantic_site.physical = intent.physical_site;
      semantic_site.member_ordinal = ordinal;
    }
    plan.probe_intents.push_back(std::move(intent));
  }
  ASSERT_TRUE(plan.valid());

  CoverageLedger ledger(std::move(plan));
  for (uint32_t ordinal = 0; ordinal < kSiteCount; ++ordinal) {
    const ProbeIntentId expected{ordinal};
    const ProbeIntent *intent = ledger.intent(expected);
    ASSERT_NE(intent, nullptr);
    EXPECT_EQ(ledger.pending_intent_ids(ProbeIntentKind::RedundantAccessObservation,
                                        intent->physical_site),
              (std::vector<ProbeIntentId>{expected}));
    EXPECT_EQ(ledger.intent_ids_covering(intent->covered_semantic_sites.front()),
              (std::vector<ProbeIntentId>{expected}));
  }
}

TEST(ObservationPlan, CommittedLoweringBatchPublicationIsTransactional) {
  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ObservationPlan plan = policy.plan;
  ASSERT_TRUE(plan.append(policy.plan));
  TransformArtifacts result;
  result.coverage_ledger = CoverageLedger(plan);

  const std::array first_id = {ProbeIntentId{0}};
  const std::array second_id = {ProbeIntentId{1}};
  auto first = make_committed_lowering(plan, first_id, std::span<const CommittedLoweringLocation>{},
                                       LoweringOutcomeKind::ResourceRejected, "first");
  auto second =
      make_committed_lowering(plan, second_id, std::span<const CommittedLoweringLocation>{},
                              LoweringOutcomeKind::PlacementRejected, "second");
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  second->intent_ids.assign(first_id.begin(), first_id.end());

  std::vector commits = {std::move(*first), std::move(*second)};
  EXPECT_FALSE(result.coverage_ledger.publish_lowering_commits(std::move(commits)));
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering, LoweringOutcomeKind::Pending);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering, LoweringOutcomeKind::Pending);
  EXPECT_TRUE(result.coverage_ledger.lowering_commits().empty());
}

TEST(ObservationPlan, CoalescingPublicationOwnsMultiLocationSyncTransactions) {
  const AccessPolicyResult access =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(access.valid());
  const ObservationPlan plan =
      one_barrier_observation_plan(access.plan.probe_intents.front().physical_site);
  ASSERT_TRUE(plan.valid());

  const std::array intent_ids = {ProbeIntentId{0}};
  const std::array first_location = {CommittedLoweringLocation{
      .original_site = plan.probe_intents.front().physical_site,
      .emitted_text_offset = 0x200,
      .emitted_size = 8,
      .relocated_guest_text_offset = std::nullopt,
  }};
  const std::array second_location = {CommittedLoweringLocation{
      .original_site = plan.probe_intents.front().physical_site,
      .emitted_text_offset = 0x300,
      .emitted_size = 12,
      .relocated_guest_text_offset = 0x308,
  }};
  auto first = make_committed_lowering(plan, intent_ids, first_location,
                                       LoweringOutcomeKind::Instrumented, "first");
  auto second = make_committed_lowering(plan, intent_ids, second_location,
                                        LoweringOutcomeKind::Instrumented, "second");
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);

  CoverageLedger ledger(plan);
  ASSERT_TRUE(ledger.publish_coalescing_instrumented_commits({*first}));
  ASSERT_EQ(ledger.lowering_commits().size(), 1u);
  const std::vector before_failed_batch(ledger.lowering_commits().begin(),
                                        ledger.lowering_commits().end());

  EXPECT_FALSE(ledger.publish_lowering_commit(*second))
      << "ordinary publication must not silently reopen an accepted intent";
  CommittedLowering malformed = *second;
  malformed.locations.front().emitted_size = 0;
  EXPECT_FALSE(ledger.publish_coalescing_instrumented_commits({*second, malformed}));
  EXPECT_EQ(std::vector(ledger.lowering_commits().begin(), ledger.lowering_commits().end()),
            before_failed_batch)
      << "a malformed later transaction must roll back the entire coalescing batch";

  ASSERT_TRUE(ledger.publish_coalescing_instrumented_commits({*second}));
  ASSERT_EQ(ledger.lowering_commits().size(), 1u);
  EXPECT_EQ(ledger.lowering_commits().front().intent_ids, (std::vector{ProbeIntentId{0}}));
  EXPECT_EQ(ledger.lowering_commits().front().locations,
            (std::vector{second_location.front(), first_location.front()}));
  EXPECT_EQ(ledger.intent_entry({0})->lowering, LoweringOutcomeKind::Instrumented);
  EXPECT_EQ(ledger.observation_plan(), plan);

  ASSERT_TRUE(ledger.publish_replacing_instrumented_commits({*first}));
  ASSERT_EQ(ledger.lowering_commits().size(), 1u);
  EXPECT_EQ(ledger.lowering_commits().front().locations, (std::vector{first_location.front()}));
  EXPECT_EQ(ledger.intent_entry({0})->lowering, LoweringOutcomeKind::Instrumented);
}

TEST(ObservationPlan, DiscardedImageRetractsOnlyInstrumentedLoweringCommits) {
  const AccessPolicyResult policy =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ObservationPlan plan = policy.plan;
  ASSERT_TRUE(plan.append(policy.plan));
  TransformArtifacts result;
  result.coverage_ledger = CoverageLedger(plan);

  const std::array rejected_id = {ProbeIntentId{0}};
  auto rejection =
      make_committed_lowering(plan, rejected_id, std::span<const CommittedLoweringLocation>{},
                              LoweringOutcomeKind::ResourceRejected, "retained planning failure");
  const std::array instrumented_id = {ProbeIntentId{1}};
  const std::array location = {CommittedLoweringLocation{
      .original_site = plan.probe_intents[1].physical_site,
      .emitted_text_offset = 0x200,
      .emitted_size = 16,
      .relocated_guest_text_offset = 0x208,
  }};
  auto instrumentation =
      make_committed_lowering(plan, instrumented_id, location, LoweringOutcomeKind::Instrumented,
                              {}, runtime_mapping_for(plan, instrumented_id));
  ASSERT_TRUE(rejection);
  ASSERT_TRUE(instrumentation);
  std::vector commits = {std::move(*rejection), std::move(*instrumentation)};
  ASSERT_TRUE(result.coverage_ledger.publish_lowering_commits(std::move(commits)));
  result.replacement.push_back(0u);
  result.patches.emplace_back();

  result.discard_candidate_modification();

  EXPECT_TRUE(result.replacement.empty());
  EXPECT_TRUE(result.patches.empty());
  ASSERT_EQ(result.coverage_ledger.lowering_commits().size(), 1u);
  EXPECT_EQ(result.coverage_ledger.lowering_commits().front().outcome,
            LoweringOutcomeKind::ResourceRejected);
  EXPECT_EQ(result.coverage_ledger.intent_entry({0})->lowering,
            LoweringOutcomeKind::ResourceRejected);
  EXPECT_EQ(result.coverage_ledger.intent_entry({1})->lowering, LoweringOutcomeKind::Pending);
}

TEST(ObservationPlan, CoverageLedgerOwnsBarrierDecisionsAlongsideAccessDecisions) {
  const AccessPolicyResult access =
      plan_access_observation(one_native_access_inventory(), policy_request(Mode::Default));
  ASSERT_TRUE(access.valid());
  ObservationPlan plan = access.plan;
  ObservationPlan barrier_fragment =
      one_barrier_observation_plan(plan.probe_intents.front().physical_site);
  ASSERT_TRUE(barrier_fragment.valid());
  ASSERT_TRUE(plan.append(barrier_fragment));
  CoverageLedger ledger(plan);
  EXPECT_TRUE(std::ranges::equal(ledger.site_decisions(), plan.site_decisions));
  EXPECT_TRUE(std::ranges::equal(ledger.barrier_site_decisions(), plan.barrier_site_decisions));
  ASSERT_EQ(ledger.intent_entries().size(), 2u);
  EXPECT_TRUE(publish_test_lowering_outcome(ledger, {1}, LoweringOutcomeKind::Instrumented));
  ASSERT_NE(ledger.intent({1}), nullptr);
  EXPECT_EQ(ledger.intent({1})->kind, ProbeIntentKind::BarrierEpoch);
}

TEST(ConSanAccessPolicy, BothModesMapOneAccessToTheirOwnEvidenceIntent) {
  constexpr std::array expected = {
      std::pair{Mode::SuperCollider, ProbeIntentKind::RedundantAccessObservation},
      std::pair{Mode::Default, ProbeIntentKind::Access},
  };
  const ProgramInventory inventory = one_native_access_inventory();
  for (const auto &[mode, expected_kind] : expected) {
    SCOPED_TRACE(mode_label(mode));
    const AccessPolicyResult policy = plan_access_observation(inventory, policy_request(mode));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.site_decisions.size(), 1u);
    ASSERT_EQ(policy.plan.probe_intents.size(), 1u);
    EXPECT_EQ(policy.plan.site_decisions.front().kind, SiteDecisionKind::Admitted);
    EXPECT_EQ(policy.plan.site_decisions.front().reason, AccessPolicyReason::None);
    EXPECT_EQ(policy.plan.probe_intents.front().kind, expected_kind);
    EXPECT_EQ(policy.plan.probe_intents.front().position, ProbePosition::Before);
  }
}

TEST(ConSanAccessPolicy, TwoRangeAccessHasTwoDecisionsAndOnePhysicalIntent) {
  AccessInventoryInput input;
  input.bytes[32] = 2;
  input.bytes[33] = 5;
  ProgramContainer kernel = make_policy_kernel();
  ProgramSite site = make_policy_lds_site("ds_store_2addr_b32");
  site.decoded_width_bits = 64;
  stage_policy_access(input, kernel, std::move(site));
  input.kernels.push_back(std::move(kernel));
  const AccessPolicyResult policy = plan_access_observation(
      build_policy_inventory(std::move(input)), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.site_decisions.size(), 2u);
  ASSERT_EQ(policy.plan.probe_intents.size(), 1u);
  EXPECT_EQ(policy.plan.site_decisions[0].semantic_site.range_ordinal, 0u);
  EXPECT_EQ(policy.plan.site_decisions[1].semantic_site.range_ordinal, 1u);
  EXPECT_EQ(policy.plan.probe_intents.front().covered_semantic_sites.size(), 2u);
}

TEST(ConSanAccessPolicy, FamilySwitchesAndFlatProvenanceAreExplicitNotApplicableDecisions) {
  AccessInventoryInput input;
  ProgramContainer kernel = make_policy_kernel();
  stage_policy_access(input, kernel, make_policy_lds_site());
  stage_policy_access(input, kernel, make_policy_flat_site(FlatAddressSpaceHint::MaybeGroup));
  stage_policy_access(input, kernel, make_policy_flat_site(FlatAddressSpaceHint::Global, 80));
  input.kernels.push_back(std::move(kernel));
  const ProgramInventory inventory = build_policy_inventory(std::move(input));

  AccessPolicyRequest request = policy_request(Mode::Default);
  request.native_lds_enabled = false;
  request.flat_provenance_mode = FlatProvenanceMode::Strict;
  const AccessPolicyResult strict = plan_access_observation(inventory, request);
  ASSERT_TRUE(strict.valid());
  ASSERT_EQ(strict.plan.site_decisions.size(), 3u);
  EXPECT_EQ(strict.plan.site_decisions[0].reason, AccessPolicyReason::AccessFamilyDisabled);
  EXPECT_EQ(strict.plan.site_decisions[1].reason, AccessPolicyReason::FlatProvenancePolicyExcluded);
  EXPECT_EQ(strict.plan.site_decisions[2].reason, AccessPolicyReason::NonGroupAddressSpace);
  EXPECT_TRUE(strict.plan.probe_intents.empty());

  request.native_lds_enabled = true;
  request.flat_provenance_mode = FlatProvenanceMode::Likely;
  const AccessPolicyResult likely = plan_access_observation(inventory, request);
  ASSERT_TRUE(likely.valid());
  EXPECT_EQ(std::ranges::count(likely.plan.site_decisions, SiteDecisionKind::Admitted,
                               &SiteDecision::kind),
            2u);
  EXPECT_EQ(likely.plan.probe_intents.size(), 2u);
}

TEST(ConSanAccessPolicy, InventoryLimitationsBecomeTypedUnsupportedDecisions) {
  AccessInventoryInput input;
  ProgramContainer kernel = make_policy_kernel();
  ProgramSite invalid_size = make_policy_lds_site("ds_store_b32", 16, 16);
  invalid_size.decoded_site().size = 0;
  stage_policy_access(input, kernel, invalid_size);
  ProgramSite invalid_width = make_policy_lds_site("ds_store_b32", 32, 32);
  invalid_width.decoded_width_bits = 0;
  stage_policy_access(input, kernel, invalid_width);
  ProgramSite missing_address = make_policy_lds_site("ds_store_b32", 48, 48);
  missing_address.operands.address_vgpr.reset();
  stage_policy_access(input, kernel, missing_address);
  ProgramSite unavailable_ranges = make_policy_lds_site("ds_store_2addr_b32", 64, 255);
  unavailable_ranges.decoded_width_bits = 64;
  stage_policy_access(input, kernel, unavailable_ranges);
  input.kernels.push_back(std::move(kernel));

  const AccessPolicyResult policy = plan_access_observation(
      build_policy_inventory(std::move(input)), policy_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.site_decisions.size(), 4u);
  EXPECT_EQ(policy.plan.site_decisions[0].reason, AccessPolicyReason::InvalidInstructionSize);
  EXPECT_EQ(policy.plan.site_decisions[1].reason, AccessPolicyReason::InvalidAccessWidth);
  EXPECT_EQ(policy.plan.site_decisions[2].reason, AccessPolicyReason::MissingAddressOperand);
  EXPECT_EQ(policy.plan.site_decisions[3].reason, AccessPolicyReason::RangeEncodingUnavailable);
  EXPECT_TRUE(std::ranges::all_of(policy.plan.site_decisions, [](const auto &decision) {
    return decision.kind == SiteDecisionKind::Unsupported;
  }));
}

TEST(ConSanAccessPolicy, IdenticalAliasesCoalesceButConflictingAliasesFailClosed) {
  AccessInventoryInput input;
  ProgramContainer function{ProgramContainerKind::Function};
  function.name = "function_alias";
  function.entry_text_offset = 0;
  ProgramSite first_access = make_policy_lds_site();
  stage_policy_access(input, function, std::move(first_access));
  input.functions.push_back(std::move(function));
  ProgramContainer second_function{ProgramContainerKind::Function};
  second_function.name = "second_function_alias";
  second_function.entry_text_offset = 0;
  ProgramSite second_access = make_policy_lds_site();
  stage_policy_access(input, second_function, std::move(second_access));
  input.functions.push_back(std::move(second_function));
  const ProgramInventory coalesced_inventory = build_policy_inventory(input);
  const AccessPolicyResult coalesced =
      plan_access_observation(coalesced_inventory, policy_request(Mode::Default));
  ASSERT_TRUE(coalesced.valid());
  ASSERT_EQ(coalesced.plan.site_decisions.size(), 1u);
  ASSERT_EQ(coalesced.plan.probe_intents.size(), 1u);
  EXPECT_EQ(coalesced_inventory.source_container_names(
                coalesced.plan.site_decisions.front().semantic_site.physical),
            (std::vector<std::string>{"function_alias", "second_function_alias"}));

  input.accesses.back().decoded_width_bits = 64;
  const AccessPolicyResult conflicting = plan_access_observation(
      build_policy_inventory(std::move(input)), policy_request(Mode::Default));
  EXPECT_FALSE(conflicting.valid());
  EXPECT_EQ(conflicting.errors, (std::vector{AccessPolicyReason::ConflictingPhysicalAliases}));
  ASSERT_EQ(conflicting.plan.site_decisions.size(), 1u);
  EXPECT_EQ(conflicting.plan.site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(conflicting.plan.site_decisions.front().reason,
            AccessPolicyReason::ConflictingPhysicalAliases);
  EXPECT_TRUE(conflicting.plan.probe_intents.empty());
}

TEST(ConSanAccessPolicy, FilterAndSynchronizationReservationRemainVisibleOutsideDenominator) {
  const ProgramInventory inventory = one_native_access_inventory();
  AccessPolicyRequest filtered = policy_request(Mode::Default);
  filtered.container_filter = "different_kernel";
  const AccessPolicyResult filter_result = plan_access_observation(inventory, filtered);
  ASSERT_TRUE(filter_result.valid());
  EXPECT_EQ(filter_result.plan.site_decisions.front().reason,
            AccessPolicyReason::ContainerFilterExcluded);

  const PhysicalSiteId reserved = inventory.access_sites().front().physical_id;
  AccessPolicyRequest synchronization = policy_request(Mode::Default);
  synchronization.reserved_for_synchronization = std::span(&reserved, 1);
  const AccessPolicyResult sync_result = plan_access_observation(inventory, synchronization);
  ASSERT_TRUE(sync_result.valid());
  EXPECT_EQ(sync_result.plan.site_decisions.front().reason,
            AccessPolicyReason::ReservedForSynchronizationPolicy);
}

TEST(ConSanAccessPolicy, UnsupportedMnemonicAndTargetCapabilityFailAtPolicyBoundary) {
  const AccessPolicyResult mnemonic = plan_access_observation(
      one_native_access_inventory("ds_unknown_b32"), policy_request(Mode::Default));
  ASSERT_TRUE(mnemonic.valid());
  EXPECT_EQ(mnemonic.plan.site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(mnemonic.plan.site_decisions.front().reason, AccessPolicyReason::UnsupportedMnemonic);

  AccessInventoryInput unknown_target;
  unknown_target.arch = ROCJITSU_CODE_ARCH_INVALID;
  unknown_target.target = ROCJITSU_CODE_TARGET_INVALID;
  ProgramContainer kernel = make_policy_kernel();
  stage_policy_access(unknown_target, kernel, make_policy_lds_site());
  unknown_target.kernels.push_back(std::move(kernel));
  const AccessPolicyResult target = plan_access_observation(
      build_policy_inventory(std::move(unknown_target)), policy_request(Mode::Default));
  ASSERT_TRUE(target.valid());
  EXPECT_EQ(target.plan.site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(target.plan.site_decisions.front().reason,
            AccessPolicyReason::TargetCapabilityUnavailable);
}

TEST(ConSanAccessPolicy, RelaxedLdsAtomicAccessUsesTargetCapabilityAndExactMnemonic) {
  AccessInventoryInput input;
  input.arch = ROCJITSU_CODE_ARCH_CDNA4;
  input.target = ROCJITSU_CODE_TARGET_GFX950;
  ProgramContainer kernel = make_policy_kernel();
  ProgramSite atomic = make_policy_lds_site("ds_add_u32");
  atomic.kind = LdsAccessKind::Atomic;
  stage_policy_access(input, kernel, atomic);
  input.kernels.push_back(std::move(kernel));
  const ProgramInventory inventory = build_policy_inventory(std::move(input));

  const AccessPolicyResult moi = plan_access_observation(inventory, policy_request(Mode::Default));
  ASSERT_TRUE(moi.valid());
  EXPECT_EQ(moi.plan.site_decisions.front().kind, SiteDecisionKind::Admitted);
  const AccessPolicyResult supercollider =
      plan_access_observation(inventory, policy_request(Mode::SuperCollider));
  ASSERT_TRUE(supercollider.valid());
  EXPECT_EQ(supercollider.plan.site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(supercollider.plan.site_decisions.front().reason,
            AccessPolicyReason::OperationKindExcluded);
}

TEST(ConSanObservationPolicy, OneAuthorityAssemblesPlanAndInitialLedgerForEveryMode) {
  const ProgramInventory inventory = one_native_access_inventory();
  for (const Mode mode : kEnabledModes) {
    const bool supercollider = mode == Mode::SuperCollider;
    const ObservationProduct product =
        assemble_observation_product(inventory, {.mode = mode,
                                                 .native_lds_enabled = true,
                                                 .group_flat_enabled = true,
                                                 .flat_provenance_mode = FlatProvenanceMode::Likely,
                                                 .barrier_tracking_enabled = true,
                                                 .include_atomic_fence_policy = !supercollider,
                                                 .atomic_fence_tracking_enabled = !supercollider,
                                                 .container_filter = {},
                                                 .kernel_name_allowlist = {},
                                                 .reserved_for_synchronization = {}});
    SCOPED_TRACE(mode_label(mode));
    ASSERT_TRUE(product.valid());
    EXPECT_EQ(product.plan().mode, mode);
    EXPECT_EQ(product.initial_coverage, CoverageLedger(product.plan()));
    EXPECT_TRUE(product.barrier_fragment_appended);
    EXPECT_EQ(product.atomic_fence_fragment_required, !supercollider);
    EXPECT_EQ(product.atomic_fence_fragment_appended, !supercollider);
    ASSERT_EQ(product.plan().probe_intents.size(), 1u);
    ASSERT_EQ(product.initial_coverage.intent_entries().size(), 1u);
    EXPECT_EQ(product.initial_coverage.intent_entries().front().intent_id,
              product.plan().probe_intents.front().id);
  }
}

TEST(ConSanObservationPolicy, TypedRequestAssemblyMatchesTheExplicitPolicyContractForEveryMode) {
  const ProgramInventory inventory = one_native_access_inventory();
  for (const Mode mode : kEnabledModes) {
    const bool supercollider = mode == Mode::SuperCollider;
    Request request;
    request.mode = supercollider ? Mode::SuperCollider : Mode::Default;
    request.probe_lds_check_trap = true;
    request.probe_flat_check_trap = true;
    request.track_barriers = true;
    request.track_atomics = true;
    switch (mode) {
    case Mode::SuperCollider:
      break;
    case Mode::Default:
      break;
    case Mode::None:
      FAIL() << "sentinel mode is not iterable";
      continue;
    }

    const ObservationProduct typed =
        assemble_observation_product(inventory, request, DebugOverrides{});
    const ObservationProduct explicit_product =
        assemble_observation_product(inventory, {.mode = mode,
                                                 .native_lds_enabled = true,
                                                 .group_flat_enabled = true,
                                                 .flat_provenance_mode = FlatProvenanceMode::Likely,
                                                 .barrier_tracking_enabled = true,
                                                 .include_atomic_fence_policy = !supercollider,
                                                 .atomic_fence_tracking_enabled = !supercollider,
                                                 .container_filter = {},
                                                 .kernel_name_allowlist = {},
                                                 .reserved_for_synchronization = {}});
    SCOPED_TRACE(mode_label(mode));
    ASSERT_TRUE(typed.valid());
    EXPECT_EQ(typed, explicit_product);
  }
}

TEST(ConSanObservationPolicy, ConflictingAliasesFailInTheAssembledProduct) {
  AccessInventoryInput input;
  ProgramContainer first;
  first.name = "first_alias";
  first.entry_text_offset = 0;
  stage_policy_access(input, first, make_policy_lds_site());
  input.functions.push_back(std::move(first));
  ProgramContainer second;
  second.name = "second_alias";
  second.entry_text_offset = 0;
  ProgramSite conflicting = make_policy_lds_site();
  conflicting.decoded_width_bits = 64;
  stage_policy_access(input, second, std::move(conflicting));
  input.functions.push_back(std::move(second));

  const ObservationProduct product = assemble_observation_product(
      build_policy_inventory(std::move(input)), {.mode = Mode::Default,
                                                 .native_lds_enabled = true,
                                                 .group_flat_enabled = true,
                                                 .flat_provenance_mode = FlatProvenanceMode::Likely,
                                                 .barrier_tracking_enabled = true,
                                                 .include_atomic_fence_policy = true,
                                                 .atomic_fence_tracking_enabled = true,
                                                 .container_filter = {},
                                                 .kernel_name_allowlist = {},
                                                 .reserved_for_synchronization = {}});
  EXPECT_FALSE(product.valid());
  EXPECT_EQ(product.access_errors, (std::vector{AccessPolicyReason::ConflictingPhysicalAliases}));
  ASSERT_EQ(product.diagnostics.size(), 1u);
  EXPECT_EQ(product.diagnostics.front(),
            "ConSan physical access at original text offset 32 was decoded inconsistently "
            "through aliases 'first_alias', 'second_alias'");
  EXPECT_EQ(product.initial_coverage, CoverageLedger(product.plan()));
}

TEST(ConSanAccessPolicy, PolicyIsDeterministicAndDoesNotMutatePublishedInventory) {
  const ProgramInventory inventory = one_native_access_inventory();
  const std::vector<ProgramSite> before(inventory.access_sites().begin(),
                                        inventory.access_sites().end());
  const AccessPolicyRequest request = policy_request(Mode::Default);
  const AccessPolicyResult first = plan_access_observation(inventory, request);
  const AccessPolicyResult second = plan_access_observation(inventory, request);
  EXPECT_EQ(first, second);
  EXPECT_TRUE(first.valid());
  EXPECT_TRUE(std::ranges::equal(inventory.access_sites(), before));
}

} // namespace
} // namespace rocjitsu::consan

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "transform_result_test_access.h"

#include "rocjitsu/code/patch/consan/consan_pipeline.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace rocjitsu {
namespace {

[[nodiscard]] RuntimeCapabilities complete_runtime_capabilities() {
  return {
      .backend = ConSanRuntimeBackend::PhysicalHsa,
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .max_report_allocation_bytes = 512u * 1024u * 1024u,
      .max_workgroup_lds_bytes = 64u * 1024u,
      .executable_binding = true,
      .dispatch_segment_binding = true,
  };
}

[[nodiscard]] ConSanRequest moi_request(ConSanMoiEngine engine) {
  ConSanRequest request;
  request.flavor = ConSanFlavor::Moi;
  request.moi_engine = engine;
  request.moi_auto_report_buffer_size = 128u * 1024u * 1024u;
  return request;
}

[[nodiscard]] ConSanRequest supercollider_request(bool observe_lds = true) {
  ConSanRequest request;
  request.flavor = ConSanFlavor::SuperCollider;
  request.probe_lds_check_trap = observe_lds;
  return request;
}

[[nodiscard]] RuntimePolicy enabled_runtime_policy() {
  RuntimePolicy policy;
  policy.enabled = true;
  return policy;
}

[[nodiscard]] TransformResult record_replay_inventory_result() {
  return transform_consan(make_rdna4_supported_lds_code_object(),
                          moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
                          enabled_runtime_policy(), ConSanDebugOverrides{},
                          complete_runtime_capabilities(), BoundRuntimeResources{});
}

TEST(ConSanPipeline, EnumInventoriesAreOrderedNamedUniqueAndRejectInvalidValues) {
  ASSERT_EQ(kConSanPipelineStages.size(), static_cast<size_t>(ConSanPipelineStage::Count));
  ASSERT_EQ(kConSanPipelineStageStatuses.size(),
            static_cast<size_t>(ConSanPipelineStageStatus::Count));
  ASSERT_EQ(kConSanTransformIssueKinds.size(),
            static_cast<size_t>(ConSanTransformIssueKind::Count));

  std::unordered_set<std::string_view> names;
  for (size_t index = 0; index < kConSanPipelineStages.size(); ++index) {
    EXPECT_EQ(static_cast<size_t>(kConSanPipelineStages[index]), index);
    const std::string_view name = consan_pipeline_stage_name(kConSanPipelineStages[index]);
    EXPECT_FALSE(name.empty());
    EXPECT_TRUE(names.insert(name).second) << name;
  }
  names.clear();
  for (size_t index = 0; index < kConSanPipelineStageStatuses.size(); ++index) {
    EXPECT_EQ(static_cast<size_t>(kConSanPipelineStageStatuses[index]), index);
    const std::string_view name =
        consan_pipeline_stage_status_name(kConSanPipelineStageStatuses[index]);
    EXPECT_FALSE(name.empty());
    EXPECT_TRUE(names.insert(name).second) << name;
  }
  names.clear();
  for (size_t index = 0; index < kConSanTransformIssueKinds.size(); ++index) {
    EXPECT_EQ(static_cast<size_t>(kConSanTransformIssueKinds[index]), index);
    const std::string_view name =
        consan_transform_issue_kind_name(kConSanTransformIssueKinds[index]);
    EXPECT_FALSE(name.empty());
    EXPECT_TRUE(names.insert(name).second) << name;
  }

  EXPECT_EQ(consan_pipeline_stage_name(ConSanPipelineStage::Count), "invalid-pipeline-stage");
  EXPECT_EQ(consan_pipeline_stage_name(static_cast<ConSanPipelineStage>(255)),
            "invalid-pipeline-stage");
  EXPECT_EQ(consan_pipeline_stage_status_name(ConSanPipelineStageStatus::Count),
            "invalid-pipeline-stage-status");
  EXPECT_EQ(consan_pipeline_stage_status_name(static_cast<ConSanPipelineStageStatus>(255)),
            "invalid-pipeline-stage-status");
  EXPECT_EQ(consan_transform_issue_kind_name(ConSanTransformIssueKind::Count),
            "invalid-transform-issue-kind");
  EXPECT_EQ(consan_transform_issue_kind_name(static_cast<ConSanTransformIssueKind>(255)),
            "invalid-transform-issue-kind");
}

TEST(ConSanPipeline, StageRecordValidatesEnumsIdentityStatusAndContractPayload) {
  const ConSanCodeObjectId identity = make_consan_code_object_id(std::array<uint8_t, 3>{1, 2, 3});
  ConSanPipelineStageRecord record{
      .stage = ConSanPipelineStage::Configuration,
      .status = ConSanPipelineStageStatus::Completed,
      .code_object = identity,
  };
  EXPECT_TRUE(record.well_formed());

  ConSanPipelineStageRecord malformed = record;
  malformed.stage = ConSanPipelineStage::Count;
  EXPECT_FALSE(malformed.well_formed());
  malformed = record;
  malformed.status = ConSanPipelineStageStatus::Count;
  EXPECT_FALSE(malformed.well_formed());
  malformed = record;
  malformed.code_object = {};
  EXPECT_FALSE(malformed.well_formed());
  malformed = record;
  malformed.contract_issue = ConSanContractIssue::MissingFlavor;
  EXPECT_FALSE(malformed.well_formed());
  malformed.status = ConSanPipelineStageStatus::Invalid;
  EXPECT_TRUE(malformed.well_formed());
  malformed.stage = ConSanPipelineStage::ProgramInventory;
  EXPECT_FALSE(malformed.well_formed());
  malformed = record;
  malformed.contract_issue = ConSanContractIssue::Count;
  EXPECT_FALSE(malformed.well_formed());
}

TEST(ConSanPipeline, TransformIssueValidatesTypedAndDiagnosticPayloadAlternatives) {
  ConSanTransformIssue contract{
      .kind = ConSanTransformIssueKind::Contract,
      .stage = ConSanPipelineStage::Configuration,
      .contract_issue = ConSanContractIssue::MissingFlavor,
      .detail = "missing-flavor",
  };
  EXPECT_TRUE(contract.well_formed());

  ConSanTransformIssue malformed = contract;
  malformed.contract_issue = ConSanContractIssue::None;
  EXPECT_FALSE(malformed.well_formed());
  malformed = contract;
  malformed.detail.clear();
  EXPECT_FALSE(malformed.well_formed());
  malformed = contract;
  malformed.kind = ConSanTransformIssueKind::LegacyLowering;
  EXPECT_FALSE(malformed.well_formed());
  malformed.contract_issue = ConSanContractIssue::None;
  EXPECT_TRUE(malformed.well_formed());
  malformed.stage = ConSanPipelineStage::Count;
  EXPECT_FALSE(malformed.well_formed());
  malformed = contract;
  malformed.kind = ConSanTransformIssueKind::Count;
  EXPECT_FALSE(malformed.well_formed());
}

TEST(ConSanPipeline, DefaultResultAndInvalidStageLookupFailClosed) {
  const TransformResult result;
  EXPECT_FALSE(result.well_formed());
  EXPECT_EQ(result.stage(ConSanPipelineStage::Configuration), nullptr);
  EXPECT_EQ(result.stage(ConSanPipelineStage::Count), nullptr);
  EXPECT_EQ(result.stage(static_cast<ConSanPipelineStage>(255)), nullptr);
  EXPECT_EQ(result.install_action(/*fail_closed=*/false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(/*fail_closed=*/true), ConSanInstallAction::Reject);
}

TEST(ConSanPipeline, MutationTallyDistinguishesSelectionAmbiguityFromApplication) {
  ConSanMutationTally tally;
  EXPECT_FALSE(tally.has_plan());
  EXPECT_FALSE(tally.has_ambiguous_plan());
  EXPECT_FALSE(tally.has_application());

  tally.requested = 1u;
  tally.planned = 1u;
  EXPECT_TRUE(tally.has_plan());
  EXPECT_FALSE(tally.has_ambiguous_plan());
  EXPECT_FALSE(tally.has_application());

  tally.planned = 2u;
  EXPECT_TRUE(tally.has_plan());
  EXPECT_TRUE(tally.has_ambiguous_plan());
  EXPECT_FALSE(tally.has_application());

  tally.applied = 2u;
  EXPECT_TRUE(tally.has_application());
  EXPECT_EQ(tally, (ConSanMutationTally{.requested = 1u, .planned = 2u, .applied = 2u}));
}

TEST(ConSanPipeline, MutationOutcomeKeepsFaultAndPerturbationDomainsSeparate) {
  const ConSanMutationOutcome outcome = {
      .fault = {.requested = 1u, .planned = 2u, .applied = 0u},
      .perturbation = {.requested = 3u, .planned = 1u, .applied = 1u},
      .applied_fault_logical_identity = "fault-site",
  };

  EXPECT_TRUE(outcome.fault.has_ambiguous_plan());
  EXPECT_FALSE(outcome.fault.has_application());
  EXPECT_FALSE(outcome.perturbation.has_ambiguous_plan());
  EXPECT_TRUE(outcome.perturbation.has_application());
  EXPECT_EQ(outcome.applied_fault_logical_identity, "fault-site");
  EXPECT_NE(outcome, ConSanMutationOutcome{});
}

TEST(ConSanPipeline, DispatchRequirementsValidatePayloadOrderingAndPacketInterception) {
  ConSanDispatchRequirements requirements;
  EXPECT_TRUE(requirements.well_formed());
  EXPECT_FALSE(requirements.requires_packet_interception());

  requirements.kernels = {{
      .kernel_name = "kernel_a",
      .required_private_bytes = 64u,
      .dynamic_private_addend = 16u,
      .required_group_bytes = 128u,
      .has_instrumented_probe = true,
  }};
  EXPECT_TRUE(requirements.kernels.front().has_segment_requirement());
  EXPECT_TRUE(requirements.kernels.front().well_formed());
  EXPECT_TRUE(requirements.well_formed());
  EXPECT_TRUE(requirements.requires_packet_interception());

  ConSanDispatchRequirements malformed = requirements;
  malformed.kernels.front().kernel_name.clear();
  EXPECT_FALSE(malformed.well_formed());
  malformed = requirements;
  malformed.kernels.front().dynamic_private_addend = 65u;
  EXPECT_FALSE(malformed.well_formed());
  malformed = requirements;
  malformed.kernels = {{.kernel_name = "kernel_b", .has_instrumented_probe = true},
                       {.kernel_name = "kernel_a", .has_instrumented_probe = true}};
  EXPECT_FALSE(malformed.well_formed());
  malformed.kernels[1].kernel_name = "kernel_b";
  EXPECT_FALSE(malformed.well_formed());

  const ConSanKernelDispatchRequirement attribution_only = {
      .kernel_name = "kernel_c",
      .has_instrumented_probe = true,
  };
  EXPECT_FALSE(attribution_only.has_segment_requirement());
  EXPECT_TRUE(attribution_only.well_formed());
  EXPECT_NE(attribution_only, ConSanKernelDispatchRequirement{});
}

TEST(ConSanPipeline, PublicationJoinsTypedCoverageAndSegmentGrowthOncePerKernel) {
  constexpr std::array<uint8_t, 24> bytes{};
  ProgramInventoryBuilder inventory_builder(bytes);
  inventory_builder.set_code_object_facts(true, 0u, ROCJITSU_CODE_ARCH_CDNA4,
                                          ROCJITSU_CODE_TARGET_GFX950);
  ConSanKernelInfo kernel_a;
  kernel_a.name = "kernel_a";
  kernel_a.descriptor_file_offset = 64u;
  kernel_a.entry_text_offset = 0u;
  kernel_a.code_size = 8u;
  kernel_a.has_text_range = true;
  ConSanLdsSite shared_access;
  shared_access.kind = ConSanLdsAccessKind::Read;
  shared_access.supported_mvp = true;
  shared_access.text_offset = 0u;
  shared_access.file_offset = 0u;
  shared_access.size = sizeof(uint32_t);
  shared_access.width_bits = 32u;
  shared_access.mnemonic = "ds_read_b32";
  shared_access.addr_vgpr = 0u;
  shared_access.owner_descriptor_file_offsets = {64u, 128u};
  kernel_a.lds_sites.push_back(shared_access);
  inventory_builder.kernels().push_back(kernel_a);
  ConSanKernelInfo kernel_b;
  kernel_b.name = "kernel_b";
  kernel_b.descriptor_file_offset = 128u;
  kernel_b.entry_text_offset = 8u;
  kernel_b.code_size = 8u;
  kernel_b.has_text_range = true;
  inventory_builder.kernels().push_back(kernel_b);
  ConSanKernelInfo kernel_c;
  kernel_c.name = "kernel_c";
  kernel_c.descriptor_file_offset = 192u;
  kernel_c.entry_text_offset = 16u;
  kernel_c.code_size = 8u;
  kernel_c.has_text_range = true;
  inventory_builder.kernels().push_back(kernel_c);
  inventory_builder.rebuild_access_inventory(bytes);
  ConSanSyncEvent barrier;
  barrier.semantic_id = {
      .physical = {.code_object = inventory_builder.view().code_object_id(),
                   .original_text_offset = 16u},
      .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
  };
  barrier.kind = ConSanSyncEventKind::Barrier;
  barrier.operation = ConSanSyncOperation::BarrierFull;
  barrier.container_name = "kernel_c";
  barrier.in_kernel = true;
  barrier.text_offset = 16u;
  barrier.file_offset = 16u;
  barrier.size = sizeof(uint32_t);
  barrier.execution_owners.push_back({.descriptor_file_offset = 192u});
  inventory_builder.synchronization().sync_events.push_back(barrier);
  const ProgramInventory inventory = inventory_builder.view();
  ASSERT_EQ(inventory.access_sites().size(), 1u);
  ASSERT_EQ(inventory.access_sites().front().ranges.size(), 1u);

  ConSanObservationPlan plan;
  plan.engine = ConSanCapabilityEngine::RecordReplay;
  const SemanticSiteId semantic = inventory.access_sites().front().ranges.front().id;
  plan.probe_intents.push_back({
      .id = {0u},
      .engine = ConSanCapabilityEngine::RecordReplay,
      .physical_site = semantic.physical,
      .covered_semantic_sites = {semantic},
      .kind = ConSanProbeIntentKind::AccessRecord,
      .position = ConSanProbePosition::Before,
      .lane_mask = ConSanLaneMaskPolicy::ActiveExecutionMask,
      .requirement = ConSanProbeRequirement::Required,
      .synchronization_association = std::nullopt,
      .dynamic_result = ConSanDynamicResultRequirement::None,
  });
  plan.site_decisions.push_back({
      .engine = ConSanCapabilityEngine::RecordReplay,
      .semantic_site = semantic,
      .kind = ConSanSiteDecisionKind::Admitted,
      .reason = ConSanAccessPolicyReason::None,
      .intent_ids = {{0u}},
      .source_containers = {"shared_access"},
  });
  plan.probe_intents.push_back({
      .id = {1u},
      .engine = ConSanCapabilityEngine::RecordReplay,
      .physical_site = barrier.semantic_id.physical,
      .covered_semantic_sites = {barrier.semantic_id},
      .kind = ConSanProbeIntentKind::BarrierRecord,
      .position = ConSanProbePosition::Before,
      .lane_mask = ConSanLaneMaskPolicy::ActiveExecutionMask,
      .requirement = ConSanProbeRequirement::Required,
      .synchronization_association = std::nullopt,
      .dynamic_result = ConSanDynamicResultRequirement::None,
  });
  plan.barrier_site_decisions.push_back({
      .engine = ConSanCapabilityEngine::RecordReplay,
      .semantic_site = barrier.semantic_id,
      .kind = ConSanSiteDecisionKind::Admitted,
      .reason = ConSanBarrierPolicyReason::None,
      .intent_ids = {{1u}},
      .source_containers = {"kernel_c"},
  });
  ASSERT_TRUE(plan.valid());
  ConSanCoverageLedger coverage(plan);
  ASSERT_TRUE(coverage.set_lowering_outcome({0u}, ConSanLoweringOutcomeKind::Instrumented));
  ASSERT_TRUE(coverage.set_lowering_outcome({1u}, ConSanLoweringOutcomeKind::Instrumented));

  ConSanResult mechanism;
  mechanism.program_inventory = inventory;
  mechanism.observation_plan = plan;
  mechanism.coverage_ledger = coverage;
  mechanism.outcome = ConSanTransformOutcome::ModifiedValid;
  mechanism.modified = true;
  mechanism.final_validation_passed = true;
  mechanism.elf_bytes = {0x7f, 'E', 'L', 'F'};
  mechanism.fault_sites.emplace_back().identity = "published-fault-site";
  mechanism.barrier_move_destinations.emplace_back().identity = "published-destination";
  mechanism.fault_plans.emplace_back().primary_identity = "published-fault-plan";
  mechanism.resource_plans.emplace_back().candidate_index = 7u;
  ConSanPatchInfo shared_segments;
  shared_segments.kind = ConSanPatchKind::InlineNopRewrite;
  shared_segments.required_private_segment_size = 40u;
  shared_segments.dynamic_private_segment_addend = 8u;
  shared_segments.required_group_segment_size = 100u;
  shared_segments.owner_descriptor_file_offsets = {64u, 128u};
  mechanism.patches.push_back(shared_segments);
  ConSanPatchInfo kernel_a_segments = shared_segments;
  kernel_a_segments.required_private_segment_size = 64u;
  kernel_a_segments.dynamic_private_segment_addend = 16u;
  kernel_a_segments.required_group_segment_size = 80u;
  kernel_a_segments.owner_descriptor_file_offsets = {64u};
  mechanism.patches.push_back(kernel_a_segments);
  ConSanPatchInfo legacy_kernel_b_segments = shared_segments;
  legacy_kernel_b_segments.anchor_offset = 12u;
  legacy_kernel_b_segments.required_private_segment_size = 56u;
  legacy_kernel_b_segments.dynamic_private_segment_addend = 0u;
  legacy_kernel_b_segments.required_group_segment_size = 120u;
  legacy_kernel_b_segments.owner_descriptor_file_offsets.clear();
  mechanism.patches.push_back(legacy_kernel_b_segments);

  const TransformResult published = TransformResultTestAccess::publish(
      bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
      enabled_runtime_policy(), ConSanDebugOverrides{}, MutationRequest{},
      complete_runtime_capabilities(), BoundRuntimeResources{}, std::move(mechanism));

  ASSERT_TRUE(published.well_formed()) << testing::PrintToString(published.issues);
  ASSERT_EQ(published.fault_sites.size(), 1u);
  EXPECT_EQ(published.fault_sites.front().identity, "published-fault-site");
  ASSERT_EQ(published.barrier_move_destinations.size(), 1u);
  EXPECT_EQ(published.barrier_move_destinations.front().identity, "published-destination");
  ASSERT_EQ(published.fault_plans.size(), 1u);
  EXPECT_EQ(published.fault_plans.front().primary_identity, "published-fault-plan");
  ASSERT_EQ(published.resource_plans.size(), 1u);
  EXPECT_EQ(published.resource_plans.front().candidate_index, 7u);
  ASSERT_EQ(published.patches.size(), 3u);
  ASSERT_EQ(published.dispatch_requirements.kernels.size(), 3u);
  EXPECT_EQ(published.dispatch_requirements.kernels[0], (ConSanKernelDispatchRequirement{
                                                            .kernel_name = "kernel_a",
                                                            .required_private_bytes = 64u,
                                                            .dynamic_private_addend = 16u,
                                                            .required_group_bytes = 100u,
                                                            .has_instrumented_probe = true,
                                                        }));
  EXPECT_EQ(published.dispatch_requirements.kernels[1], (ConSanKernelDispatchRequirement{
                                                            .kernel_name = "kernel_b",
                                                            .required_private_bytes = 56u,
                                                            .dynamic_private_addend = 8u,
                                                            .required_group_bytes = 120u,
                                                            .has_instrumented_probe = true,
                                                        }));
  EXPECT_EQ(published.dispatch_requirements.kernels[2], (ConSanKernelDispatchRequirement{
                                                            .kernel_name = "kernel_c",
                                                            .has_instrumented_probe = true,
                                                        }));
  EXPECT_TRUE(published.dispatch_requirements.requires_packet_interception());
}

TEST(ConSanPipeline, InvalidConfigurationStopsBeforeLegacyLoweringWithTypedIssue) {
  ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  request.moi_sample_stride = 0;
  constexpr std::array<uint8_t, 4> bytes = {0x7f, 'E', 'L', 'F'};
  TransformResult result =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), {});

  ASSERT_TRUE(result.well_formed());
  EXPECT_EQ(result.configuration_issue, ConSanContractIssue::InvalidSampleStride);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  ASSERT_EQ(result.issues.size(), 1u);
  EXPECT_EQ(result.issues.front().kind, ConSanTransformIssueKind::Contract);
  EXPECT_EQ(result.issues.front().stage, ConSanPipelineStage::Configuration);
  EXPECT_EQ(result.issues.front().contract_issue, ConSanContractIssue::InvalidSampleStride);
  EXPECT_EQ(result.stage(ConSanPipelineStage::Configuration)->status,
            ConSanPipelineStageStatus::Invalid);

  EXPECT_TRUE(result.program_inventory.empty());
  EXPECT_FALSE(result.program_inventory.code_object_parsed());
  EXPECT_EQ(result.issues.front().detail, "invalid-sample-stride");
}

TEST(ConSanPipeline, MissingRuntimeBackendStopsAtCapabilityBoundary) {
  TransformResult result = transform_consan(
      make_rdna4_supported_lds_code_object(), moi_request(ConSanMoiEngine::RecordReplay),
      TransformPolicy{}, enabled_runtime_policy(), ConSanDebugOverrides{}, RuntimeCapabilities{},
      BoundRuntimeResources{});

  ASSERT_TRUE(result.well_formed());
  EXPECT_EQ(result.configuration_issue, ConSanContractIssue::None);
  EXPECT_EQ(result.stage(ConSanPipelineStage::Configuration)->status,
            ConSanPipelineStageStatus::Completed);
  EXPECT_EQ(result.stage(ConSanPipelineStage::TargetAndRuntimeCapabilities)->status,
            ConSanPipelineStageStatus::Invalid);
  EXPECT_EQ(result.stage(ConSanPipelineStage::TargetAndRuntimeCapabilities)->contract_issue,
            ConSanContractIssue::MissingRuntimeBackend);
  ASSERT_EQ(result.issues.size(), 1u);
  EXPECT_EQ(result.issues.front().contract_issue, ConSanContractIssue::MissingRuntimeBackend);
}

TEST(ConSanPipeline, InvalidCodeObjectRetainsOneIdentityAcrossEveryStage) {
  constexpr std::array<uint8_t, 4> bytes = {0x7f, 'E', 'L', 'F'};
  TransformResult result = transform_consan(
      bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
      enabled_runtime_policy(), ConSanDebugOverrides{}, complete_runtime_capabilities(), {});

  ASSERT_TRUE(result.well_formed());
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_TRUE(std::ranges::all_of(result.stages, [&](const ConSanPipelineStageRecord &stage) {
    return stage.code_object == result.code_object;
  }));
  EXPECT_EQ(result.stage(ConSanPipelineStage::ProgramInventory)->status,
            ConSanPipelineStageStatus::Completed);
  EXPECT_EQ(result.stage(ConSanPipelineStage::ObservationPlan)->status,
            ConSanPipelineStageStatus::Invalid);
  EXPECT_EQ(result.stage(ConSanPipelineStage::FinalValidation)->status,
            ConSanPipelineStageStatus::Invalid);
  EXPECT_FALSE(result.issues.empty());
}

TEST(ConSanPipeline, EveryEnginePublishesItsTypedEvidenceContractBeforeBinding) {
  struct ModeCase {
    ConSanRequest request;
    ConSanEvidenceSchema schema;
  };
  const std::array<ModeCase, 4> modes = {{
      {moi_request(ConSanMoiEngine::RecordReplay), ConSanEvidenceSchema::RecordReplay},
      {moi_request(ConSanMoiEngine::Sampled), ConSanEvidenceSchema::Sampled},
      {moi_request(ConSanMoiEngine::InlineShadow), ConSanEvidenceSchema::InlineShadow},
      {supercollider_request(), ConSanEvidenceSchema::SuperCollider},
  }};
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();

  for (const ModeCase &mode : modes) {
    SCOPED_TRACE(consan_evidence_schema_name(mode.schema));
    TransformResult result = transform_consan(
        bytes, mode.request, TransformPolicy{}, enabled_runtime_policy(), ConSanDebugOverrides{},
        complete_runtime_capabilities(), BoundRuntimeResources{});
    ASSERT_TRUE(result.well_formed())
        << consan_evidence_schema_name(mode.schema) << testing::PrintToString(result.issues);
    ASSERT_TRUE(result.evidence_requirements)
        << consan_evidence_schema_name(mode.schema) << testing::PrintToString(result.issues);
    EXPECT_EQ(consan_evidence_requirements_schema(*result.evidence_requirements), mode.schema);
    EXPECT_EQ(result.stage(ConSanPipelineStage::ProgramInventory)->status,
              ConSanPipelineStageStatus::Completed);
    EXPECT_EQ(result.stage(ConSanPipelineStage::ObservationPlan)->status,
              ConSanPipelineStageStatus::Completed);
    EXPECT_EQ(result.stage(ConSanPipelineStage::EvidenceRequirements)->status,
              ConSanPipelineStageStatus::Completed);
    EXPECT_EQ(result.stage(ConSanPipelineStage::RuntimeBinding)->status,
              ConSanPipelineStageStatus::Deferred);
  }
}

TEST(ConSanPipeline, MoiEvidenceCapacityComesDirectlyFromTypedRequestPolicyAndCapabilities) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  RuntimeCapabilities capabilities = complete_runtime_capabilities();
  capabilities.max_workgroup_lds_bytes = 96u * 1024u;

  for (const bool expert_limit : {false, true}) {
    TransformPolicy transform_policy;
    transform_policy.max_patches = 7u;
    transform_policy.max_patches_is_expert_limit = expert_limit;
    const std::optional<uint64_t> maximum_access_probe_count =
        expert_limit ? std::optional<uint64_t>{7u} : std::nullopt;

    for (const ConSanMoiEngine engine :
         {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::Sampled, ConSanMoiEngine::InlineShadow}) {
      SCOPED_TRACE(consan_moi_engine_name(engine));
      SCOPED_TRACE(expert_limit);
      ConSanRequest request = moi_request(engine);
      request.moi_auto_report_buffer_size = 4096u;
      const TransformResult result =
          transform_consan(bytes, request, transform_policy, enabled_runtime_policy(),
                           ConSanDebugOverrides{}, capabilities, BoundRuntimeResources{});

      ASSERT_TRUE(result.evidence_requirements) << testing::PrintToString(result.issues);
      switch (engine) {
      case ConSanMoiEngine::RecordReplay:
        EXPECT_EQ(std::get<ConSanRecordReplayEvidenceRequirements>(*result.evidence_requirements),
                  plan_consan_record_replay_evidence(
                      result.observation_plan,
                      {.caller_ceiling_bytes = 4096u,
                       .maximum_access_probe_count = maximum_access_probe_count}));
        break;
      case ConSanMoiEngine::Sampled:
        EXPECT_EQ(std::get<ConSanSampledEvidenceRequirements>(*result.evidence_requirements),
                  plan_consan_sampled_evidence(
                      result.observation_plan,
                      {.caller_ceiling_bytes = 4096u,
                       .maximum_access_probe_count = maximum_access_probe_count}));
        break;
      case ConSanMoiEngine::InlineShadow:
        EXPECT_EQ(std::get<ConSanInlineShadowEvidenceRequirements>(*result.evidence_requirements),
                  plan_consan_inline_shadow_evidence(
                      result.program_inventory, result.observation_plan,
                      {.caller_ceiling_bytes = 4096u,
                       .maximum_access_probe_count = maximum_access_probe_count,
                       .maximum_workgroup_lds_bytes = 96u * 1024u}));
        break;
      }
    }
  }
}

TEST(ConSanPipeline, EmptyPlansNeedNoRuntimeBindingForAnyEngine) {
  constexpr std::array<uint32_t, 1> kTextWords = {0xBFB00000u};
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(kTextWords, "typed_empty_observation_plan");
  const std::array requests = {
      moi_request(ConSanMoiEngine::RecordReplay),
      moi_request(ConSanMoiEngine::Sampled),
      moi_request(ConSanMoiEngine::InlineShadow),
      supercollider_request(),
  };
  for (const ConSanRequest &request : requests) {
    TransformResult result = transform_consan(
        bytes, request, TransformPolicy{}, enabled_runtime_policy(), ConSanDebugOverrides{},
        complete_runtime_capabilities(), BoundRuntimeResources{});
    ASSERT_TRUE(result.well_formed()) << testing::PrintToString(result.issues);
    ASSERT_TRUE(result.evidence_requirements);
    EXPECT_FALSE(
        std::visit([](const auto &requirements) { return requirements.requires_binding(); },
                   *result.evidence_requirements));
    EXPECT_EQ(result.stage(ConSanPipelineStage::RuntimeBinding)->status,
              ConSanPipelineStageStatus::NotApplicable);
  }
}

TEST(ConSanPipeline, ConcreteBindingChecksRuntimeFactsAndLifetimeScope) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  TransformResult inventory =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), {});
  ASSERT_TRUE(inventory.evidence_requirements);
  const auto *requirements =
      std::get_if<ConSanRecordReplayEvidenceRequirements>(&*inventory.evidence_requirements);
  ASSERT_NE(requirements, nullptr);
  ASSERT_TRUE(requirements->complete());

  BoundRuntimeResources bound;
  bound.scope = ConSanRuntimeResourceScope::Executable;
  bound.moi_report_buffer_address = 0x123456780000ull;
  bound.moi_report_buffer_size = requirements->abi_plan.required_bytes;
  TransformResult complete =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), bound);
  ASSERT_TRUE(complete.well_formed()) << testing::PrintToString(complete.issues);
  EXPECT_EQ(complete.stage(ConSanPipelineStage::RuntimeBinding)->status,
            ConSanPipelineStageStatus::Completed);

  RuntimeCapabilities missing_visibility = complete_runtime_capabilities();
  missing_visibility.host_device_visible_memory = false;
  TransformResult rejected =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, missing_visibility, bound);
  ASSERT_TRUE(rejected.well_formed()) << testing::PrintToString(rejected.issues);
  EXPECT_EQ(rejected.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_EQ(rejected.stage(ConSanPipelineStage::RuntimeBinding)->status,
            ConSanPipelineStageStatus::Unsupported);
  EXPECT_EQ(rejected.stage(ConSanPipelineStage::RuntimeBinding)->contract_issue,
            ConSanContractIssue::MissingVisibleMemory);

  BoundRuntimeResources undersized = bound;
  --undersized.moi_report_buffer_size;
  TransformResult rejected_size =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), undersized);
  ASSERT_TRUE(rejected_size.well_formed()) << testing::PrintToString(rejected_size.issues);
  EXPECT_EQ(rejected_size.stage(ConSanPipelineStage::RuntimeBinding)->contract_issue,
            ConSanContractIssue::InvalidResourceSize);

  BoundRuntimeResources wrong_schema;
  wrong_schema.scope = ConSanRuntimeResourceScope::Executable;
  wrong_schema.report_buffer_address = 0x123456780000ull;
  TransformResult rejected_schema =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), wrong_schema);
  ASSERT_TRUE(rejected_schema.well_formed()) << testing::PrintToString(rejected_schema.issues);
  EXPECT_EQ(rejected_schema.stage(ConSanPipelineStage::RuntimeBinding)->contract_issue,
            ConSanContractIssue::InvalidResourceAddress);
}

TEST(ConSanPipeline, SuperColliderBindingRequiresItsStickyMarkerAddress) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = supercollider_request();

  BoundRuntimeResources marker;
  marker.scope = ConSanRuntimeResourceScope::Executable;
  marker.report_buffer_address = 0x123456780000ull;
  TransformResult complete =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), marker);
  ASSERT_TRUE(complete.well_formed()) << testing::PrintToString(complete.issues);
  EXPECT_EQ(complete.stage(ConSanPipelineStage::RuntimeBinding)->status,
            ConSanPipelineStageStatus::Completed);

  BoundRuntimeResources wrong_schema;
  wrong_schema.scope = ConSanRuntimeResourceScope::Executable;
  wrong_schema.moi_report_buffer_address = 0x123456780000ull;
  wrong_schema.moi_report_buffer_size = sizeof(ConSanMoiReportHeader);
  TransformResult rejected =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), wrong_schema);
  ASSERT_TRUE(rejected.well_formed()) << testing::PrintToString(rejected.issues);
  EXPECT_EQ(rejected.stage(ConSanPipelineStage::RuntimeBinding)->contract_issue,
            ConSanContractIssue::InvalidResourceAddress);
}

TEST(ConSanPipeline, ResultValidatorRejectsEveryOwnedCrossTypeInvariant) {
  const TransformResult good = record_replay_inventory_result();
  ASSERT_TRUE(good.well_formed()) << testing::PrintToString(good.issues);

  TransformResult malformed = good;
  malformed.stages.pop_back();
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  std::swap(malformed.stages[0], malformed.stages[1]);
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.stages[1].code_object = make_consan_code_object_id(std::array<uint8_t, 1>{9});
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.configuration_issue = ConSanContractIssue::MissingFlavor;
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.stages.front().status = ConSanPipelineStageStatus::Invalid;
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  ProgramInventoryBuilder foreign(std::array<uint8_t, 2>{7, 8});
  malformed.program_inventory = foreign.view();
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  ASSERT_TRUE(malformed.evidence_requirements);
  std::get<ConSanRecordReplayEvidenceRequirements>(*malformed.evidence_requirements).schema =
      ConSanEvidenceSchema::Sampled;
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.issues.push_back({});
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.dispatch_requirements.kernels.push_back({});
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.outcome = ConSanTransformOutcome::ModifiedValid;
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.replacement_bytes.push_back(1);
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.stages.back().status = ConSanPipelineStageStatus::Invalid;
  EXPECT_FALSE(malformed.well_formed());
}

TEST(ConSanPipeline, InstallActionTruthTableUsesOnlySplitStaticResult) {
  TransformResult result;
  result.outcome = ConSanTransformOutcome::Unchanged;
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::LoadOriginal);

  result.outcome = ConSanTransformOutcome::Unsupported;
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::Reject);
  result.outcome = ConSanTransformOutcome::Invalid;
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::Reject);

  result.outcome = ConSanTransformOutcome::ModifiedValid;
  result.final_validation_passed = false;
  result.replacement_bytes = {1};
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::Reject);
  result.final_validation_passed = true;
  result.replacement_bytes.clear();
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::Reject);
  result.replacement_bytes = {1};
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadReplacement);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::LoadReplacement);
}

TEST(ConSanPipeline, ProductionResultOwnsAllPublishedTransformArtifacts) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  const TransformPolicy transform_policy;
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const ConSanDebugOverrides debug;
  const MutationRequest mutation;
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();
  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = 64u * 1024u * 1024u;

  TransformResult split = transform_consan(bytes, request, transform_policy, runtime_policy, debug,
                                           capabilities, resources);
  ASSERT_TRUE(split.well_formed()) << testing::PrintToString(split.issues);

  EXPECT_TRUE(split.program_inventory.code_object_parsed());
  EXPECT_EQ(split.program_inventory.code_object_id(), split.code_object);
  EXPECT_FALSE(split.observation_plan.probe_intents.empty());
  EXPECT_EQ(split.coverage_ledger.intent_entries().size(),
            split.observation_plan.probe_intents.size());
  EXPECT_FALSE(split.replacement_bytes.empty());
  EXPECT_FALSE(split.patches.empty());
  EXPECT_FALSE(split.resource_plans.empty());
  EXPECT_EQ(split.install_action(false), split.outcome == ConSanTransformOutcome::ModifiedValid
                                             ? ConSanInstallAction::LoadReplacement
                                             : ConSanInstallAction::LoadOriginal);
}

TEST(ConSanPipeline, PristineMoiInventoryPreservesOnlyTheRequestedExtendedBarrierPairs) {
  std::array<uint32_t, 17> text_words{};
  text_words[0] = 0xBE804EC1u; // s_barrier_signal -1
  std::fill(text_words.begin() + 1, text_words.begin() + 15,
            build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[15] = 0xBF94FFFFu; // s_barrier_wait -1
  text_words[16] = 0xBFB00000u;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "typed_pristine_extended_barrier_pair");

  ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  request.moi_track_barriers = true;
  TransformPolicy transform_policy;
  transform_policy.max_patches = 16;

  const TransformResult ordinary = transform_consan_pristine_moi_inventory(
      bytes, request, transform_policy, enabled_runtime_policy(), ConSanDebugOverrides{},
      MutationRequest{}, complete_runtime_capabilities(), BoundRuntimeResources{},
      /*preserve_extended_barrier_pairs=*/false);
  const TransformResult preserved = transform_consan_pristine_moi_inventory(
      bytes, request, transform_policy, enabled_runtime_policy(), ConSanDebugOverrides{},
      MutationRequest{}, complete_runtime_capabilities(), BoundRuntimeResources{},
      /*preserve_extended_barrier_pairs=*/true);
  BoundRuntimeResources supplied_binding;
  supplied_binding.scope = ConSanRuntimeResourceScope::Executable;
  supplied_binding.moi_report_buffer_address = 0x123456780000ull;
  supplied_binding.moi_report_buffer_size = 64u * 1024u * 1024u;
  const TransformResult cleared_binding = transform_consan_pristine_moi_inventory(
      bytes, request, transform_policy, enabled_runtime_policy(), ConSanDebugOverrides{},
      MutationRequest{}, complete_runtime_capabilities(), supplied_binding,
      /*preserve_extended_barrier_pairs=*/true);

  const auto ordinary_sync = ordinary.program_inventory.sync().sync_sequences;
  const auto preserved_sync = preserved.program_inventory.sync().sync_sequences;
  const auto cleared_sync = cleared_binding.program_inventory.sync().sync_sequences;
  EXPECT_EQ(std::ranges::count(ordinary_sync, ConSanSyncOperation::BarrierFull,
                               &ConSanSyncSequence::operation),
            0u);
  EXPECT_EQ(std::ranges::count(preserved_sync, ConSanSyncOperation::BarrierFull,
                               &ConSanSyncSequence::operation),
            1u);
  EXPECT_TRUE(ordinary.replacement_bytes.empty());
  EXPECT_TRUE(preserved.replacement_bytes.empty());
  EXPECT_TRUE(cleared_binding.replacement_bytes.empty());
  ASSERT_EQ(cleared_sync.size(), preserved_sync.size());
  ASSERT_FALSE(cleared_sync.empty());
  EXPECT_EQ(cleared_sync.front().identity, preserved_sync.front().identity);
  EXPECT_EQ(cleared_sync.front().operation, preserved_sync.front().operation);
}

TEST(ConSanPipeline, PristineMoiRetryRemainsInsideTypedPipelineBoundary) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  const TransformPolicy transform_policy;
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const ConSanDebugOverrides debug;
  const MutationRequest mutation;
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();

  TransformResult inventory = transform_consan_pristine_moi_inventory(
      bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities, {},
      /*preserve_extended_barrier_pairs=*/false);
  ASSERT_TRUE(inventory.well_formed()) << testing::PrintToString(inventory.issues);
  ASSERT_TRUE(inventory.evidence_requirements);
  const auto &requirements =
      std::get<ConSanRecordReplayEvidenceRequirements>(*inventory.evidence_requirements);

  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = requirements.abi_plan.required_bytes;
  TransformResult retried = retry_transform_consan_pristine_moi_inventory(
      bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities, resources,
      std::move(inventory));
  const TransformResult direct = transform_consan(bytes, request, transform_policy, runtime_policy,
                                                  debug, capabilities, resources);

  ASSERT_TRUE(retried.well_formed()) << testing::PrintToString(retried.issues);
  ASSERT_EQ(retried.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(retried.install_action(false), ConSanInstallAction::LoadReplacement);
  EXPECT_EQ(retried.code_object, direct.code_object);
  EXPECT_EQ(retried.observation_plan, direct.observation_plan);
  EXPECT_EQ(retried.coverage_ledger, direct.coverage_ledger);
  EXPECT_EQ(retried.replacement_bytes, direct.replacement_bytes);
  EXPECT_EQ(retried.patches.size(), direct.patches.size());
  EXPECT_EQ(retried.resource_plans.size(), direct.resource_plans.size());
  EXPECT_EQ(retried.fault_sites.size(), direct.fault_sites.size());
  EXPECT_EQ(retried.barrier_move_destinations.size(), direct.barrier_move_destinations.size());
  EXPECT_EQ(retried.fault_plans.size(), direct.fault_plans.size());
}

TEST(ConSanPipeline, MoiRetryRejectsAnOrdinaryResultWithoutRetainedInventory) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();
  TransformResult ordinary =
      transform_consan(bytes, request, TransformPolicy{}, runtime_policy, ConSanDebugOverrides{},
                       capabilities, BoundRuntimeResources{});
  ASSERT_TRUE(ordinary.evidence_requirements);
  const auto &requirements =
      std::get<ConSanRecordReplayEvidenceRequirements>(*ordinary.evidence_requirements);

  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = requirements.abi_plan.required_bytes;
  const TransformResult retried = retry_transform_consan_pristine_moi_inventory(
      bytes, request, TransformPolicy{}, runtime_policy, ConSanDebugOverrides{}, MutationRequest{},
      capabilities, resources, std::move(ordinary));

  ASSERT_TRUE(retried.well_formed()) << testing::PrintToString(retried.issues);
  EXPECT_EQ(retried.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_TRUE(std::ranges::any_of(retried.issues, [](const ConSanTransformIssue &issue) {
    return issue.detail.find("no retained inventory") != std::string::npos;
  })) << testing::PrintToString(retried.issues);
}

TEST(ConSanPipeline, OrdinaryAndMutationEntryPointsAreSeparateAndDeterministic) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = supercollider_request();
  const TransformPolicy transform_policy;
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const ConSanDebugOverrides debug;
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();
  const BoundRuntimeResources resources;
  const ConSanRequest request_before = request;
  const RuntimeCapabilities capabilities_before = capabilities;

  const TransformResult first = transform_consan(bytes, request, transform_policy, runtime_policy,
                                                 debug, capabilities, resources);
  const TransformResult second = transform_consan(bytes, request, transform_policy, runtime_policy,
                                                  debug, capabilities, resources);
  ASSERT_TRUE(first.well_formed()) << testing::PrintToString(first.issues);
  ASSERT_TRUE(second.well_formed()) << testing::PrintToString(second.issues);
  EXPECT_EQ(first.code_object, second.code_object);
  EXPECT_EQ(first.stages, second.stages);
  EXPECT_EQ(first.observation_plan, second.observation_plan);
  EXPECT_EQ(first.evidence_requirements, second.evidence_requirements);
  EXPECT_EQ(first.outcome, second.outcome);
  EXPECT_EQ(first.replacement_bytes, second.replacement_bytes);
  EXPECT_EQ(first.issues, second.issues);
  EXPECT_EQ(first.warnings, second.warnings);
  EXPECT_EQ(first.mutation, second.mutation);
  EXPECT_EQ(first.dispatch_requirements, second.dispatch_requirements);
  EXPECT_EQ(request, request_before);
  EXPECT_EQ(capabilities, capabilities_before);

  MutationRequest mutation;
  mutation.fault_lds_wrong_address = true;
  mutation.fault_lds_address_vgpr = 0;
  mutation.fault_dry_run = true;
  TransformResult mutated = transform_consan_with_mutation(
      bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities, resources);
  ASSERT_TRUE(mutated.well_formed()) << testing::PrintToString(mutated.issues);
  EXPECT_GT(mutated.mutation.fault.requested, 0u);
  EXPECT_NE(mutated.mutation, ConSanMutationOutcome{});
  EXPECT_FALSE(mutated.fault_sites.empty());
  EXPECT_FALSE(mutated.fault_plans.empty());
  EXPECT_EQ(first.code_object, mutated.program_inventory.code_object_id());
}

TEST(ConSanPipeline, RuntimeDiscardClearsInstallableTypedArtifacts) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  TransformResult inventory =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), {});
  ASSERT_TRUE(inventory.evidence_requirements);
  const auto &requirements =
      std::get<ConSanRecordReplayEvidenceRequirements>(*inventory.evidence_requirements);

  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = requirements.abi_plan.required_bytes;
  TransformResult result =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), resources);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_FALSE(result.replacement_bytes.empty());
  ASSERT_FALSE(result.patches.empty());
  ASSERT_FALSE(result.dispatch_requirements.kernels.empty());

  result.discard_replacement("runtime report allocation failed");

  ASSERT_TRUE(result.well_formed()) << testing::PrintToString(result.issues);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.final_validation_passed);
  EXPECT_TRUE(result.replacement_bytes.empty());
  EXPECT_TRUE(result.dispatch_requirements.kernels.empty());
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::Reject);
  EXPECT_TRUE(result.patches.empty());
  EXPECT_EQ(result.warnings.back(), "runtime report allocation failed");
  EXPECT_EQ(result.stage(ConSanPipelineStage::LegacyLowering)->status,
            ConSanPipelineStageStatus::Unsupported);
  EXPECT_EQ(result.stage(ConSanPipelineStage::FinalValidation)->status,
            ConSanPipelineStageStatus::NotApplicable);
  EXPECT_EQ(result.stage(ConSanPipelineStage::Complete)->status,
            ConSanPipelineStageStatus::Unsupported);
}

} // namespace
} // namespace rocjitsu

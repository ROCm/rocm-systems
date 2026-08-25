// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

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

  const ConSanResult legacy = std::move(result).take_legacy_result();
  EXPECT_FALSE(legacy.visited_code_object);
  EXPECT_EQ(legacy.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_EQ(legacy.input_size, bytes.size());
  EXPECT_EQ(legacy.errors, (std::vector<std::string>{"invalid-sample-stride"}));
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

TEST(ConSanPipeline, EmptySuperColliderPlanNeedsNoRuntimeBinding) {
  TransformResult result = transform_consan(
      make_rdna4_supported_lds_code_object(), supercollider_request(/*observe_lds=*/false),
      TransformPolicy{}, enabled_runtime_policy(), ConSanDebugOverrides{},
      complete_runtime_capabilities(), BoundRuntimeResources{});
  ASSERT_TRUE(result.well_formed()) << testing::PrintToString(result.issues);
  ASSERT_TRUE(result.evidence_requirements);
  const auto *requirements =
      std::get_if<ConSanSuperColliderEvidenceRequirements>(&*result.evidence_requirements);
  ASSERT_NE(requirements, nullptr);
  EXPECT_EQ(requirements->marker_bytes, 0u);
  EXPECT_FALSE(requirements->requires_binding());
  EXPECT_EQ(result.stage(ConSanPipelineStage::RuntimeBinding)->status,
            ConSanPipelineStageStatus::NotApplicable);
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

TEST(ConSanPipeline, LegacyProjectionRestoresArtifactsWithoutChangingObservableResult) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  const TransformPolicy transform_policy;
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const ConSanDebugOverrides debug;
  const MutationRequest mutation;
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();
  const BoundRuntimeResources resources;

  const ConSanResult direct = LegacyConSanLowering::run(
      bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities, resources);
  TransformResult split = transform_consan(bytes, request, transform_policy, runtime_policy, debug,
                                           capabilities, resources);
  ASSERT_TRUE(split.well_formed()) << testing::PrintToString(split.issues);
  const ConSanResult projected = std::move(split).take_legacy_result();

  EXPECT_EQ(projected.visited_code_object, direct.visited_code_object);
  EXPECT_EQ(projected.parsed_code_object, direct.parsed_code_object);
  EXPECT_EQ(projected.input_fingerprint, direct.input_fingerprint);
  EXPECT_EQ(projected.outcome, direct.outcome);
  EXPECT_EQ(projected.modified, direct.modified);
  EXPECT_EQ(projected.final_validation_passed, direct.final_validation_passed);
  EXPECT_EQ(projected.program_inventory.code_object_id(),
            direct.program_inventory.code_object_id());
  EXPECT_EQ(projected.observation_plan, direct.observation_plan);
  EXPECT_EQ(projected.coverage_ledger, direct.coverage_ledger);
  EXPECT_EQ(projected.elf_bytes, direct.elf_bytes);
  EXPECT_EQ(projected.errors, direct.errors);
  EXPECT_EQ(projected.warnings, direct.warnings);
  EXPECT_EQ(consan_install_action(projected, false),
            projected.outcome == ConSanTransformOutcome::ModifiedValid
                ? ConSanInstallAction::LoadReplacement
                : ConSanInstallAction::LoadOriginal);
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
  EXPECT_EQ(request, request_before);
  EXPECT_EQ(capabilities, capabilities_before);

  MutationRequest mutation;
  mutation.fault_lds_wrong_address = true;
  mutation.fault_lds_address_vgpr = 0;
  mutation.fault_dry_run = true;
  TransformResult mutated = transform_consan_with_mutation(
      bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities, resources);
  ASSERT_TRUE(mutated.well_formed()) << testing::PrintToString(mutated.issues);
  const ConSanResult mutation_projection = std::move(mutated).take_legacy_result();
  EXPECT_GT(mutation_projection.requested_fault_mutations, 0u);
  EXPECT_EQ(first.code_object, mutation_projection.program_inventory.code_object_id());
}

} // namespace
} // namespace rocjitsu

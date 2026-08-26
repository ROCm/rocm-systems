// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_pipeline.h"

#include "rocjitsu/code/patch/consan/consan_legacy_lowering.h"

#include <algorithm>
#include <utility>
#include <variant>

namespace rocjitsu {

namespace {

[[nodiscard]] constexpr bool valid_stage(ConSanPipelineStage stage) {
  return static_cast<uint8_t>(stage) < static_cast<uint8_t>(ConSanPipelineStage::Count);
}

[[nodiscard]] constexpr bool valid_stage_status(ConSanPipelineStageStatus status) {
  return static_cast<uint8_t>(status) < static_cast<uint8_t>(ConSanPipelineStageStatus::Count);
}

[[nodiscard]] constexpr bool valid_issue_kind(ConSanTransformIssueKind kind) {
  return static_cast<uint8_t>(kind) < static_cast<uint8_t>(ConSanTransformIssueKind::Count);
}

[[nodiscard]] constexpr bool valid_contract_issue(ConSanContractIssue issue) {
  return static_cast<uint8_t>(issue) < static_cast<uint8_t>(ConSanContractIssue::Count);
}

[[nodiscard]] ConSanPipelineStageStatus terminal_stage_status(ConSanTransformOutcome outcome) {
  switch (outcome) {
  case ConSanTransformOutcome::Unchanged:
  case ConSanTransformOutcome::ModifiedValid:
    return ConSanPipelineStageStatus::Completed;
  case ConSanTransformOutcome::Unsupported:
    return ConSanPipelineStageStatus::Unsupported;
  case ConSanTransformOutcome::Invalid:
    return ConSanPipelineStageStatus::Invalid;
  }
  return ConSanPipelineStageStatus::Invalid;
}

[[nodiscard]] RuntimeCapabilityRequirements
runtime_requirements(const ConSanEvidenceRequirements &requirements) {
  return std::visit([](const auto &typed) { return typed.runtime_requirements; }, requirements);
}

[[nodiscard]] ConSanRuntimeResourceScope
delivery_scope(const ConSanEvidenceRequirements &requirements) {
  return std::visit([](const auto &typed) { return typed.delivery_scope; }, requirements);
}

[[nodiscard]] bool evidence_requires_binding(const ConSanEvidenceRequirements &requirements) {
  return std::visit(
      [](const auto &typed) {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, ConSanSuperColliderEvidenceRequirements>)
          return typed.requires_binding();
        return typed.complete();
      },
      requirements);
}

[[nodiscard]] bool evidence_is_complete(const ConSanEvidenceRequirements &requirements) {
  return std::visit([](const auto &typed) { return typed.complete(); }, requirements);
}

[[nodiscard]] ConSanContractIssue
validate_evidence_binding(const ConSanEvidenceRequirements &requirements,
                          const BoundRuntimeResources &resources) {
  return std::visit(
      [&](const auto &typed) {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, ConSanSuperColliderEvidenceRequirements>) {
          return typed.requires_binding() && !resources.report_buffer_address
                     ? ConSanContractIssue::InvalidResourceAddress
                     : ConSanContractIssue::None;
        } else {
          if (!resources.moi_report_buffer_address)
            return ConSanContractIssue::InvalidResourceAddress;
          return resources.moi_report_buffer_size < typed.abi_plan.required_bytes
                     ? ConSanContractIssue::InvalidResourceSize
                     : ConSanContractIssue::None;
        }
      },
      requirements);
}

void initialize_stage_records(TransformResult &result) {
  result.stages.reserve(kConSanPipelineStages.size());
  for (ConSanPipelineStage stage : kConSanPipelineStages) {
    result.stages.push_back({
        .stage = stage,
        .status = ConSanPipelineStageStatus::NotApplicable,
        .code_object = result.code_object,
    });
  }
}

ConSanPipelineStageRecord &stage_record(TransformResult &result, ConSanPipelineStage stage) {
  return result.stages[static_cast<size_t>(stage)];
}

void add_contract_issue(TransformResult &result, ConSanPipelineStage stage,
                        ConSanContractIssue issue) {
  result.issues.push_back({
      .kind = ConSanTransformIssueKind::Contract,
      .stage = stage,
      .contract_issue = issue,
      .detail = std::string(consan_contract_issue_name(issue)),
  });
}

[[nodiscard]] ConSanResult make_rejected_legacy_result(const ConSanRequest &request,
                                                       std::span<const uint8_t> code_object_bytes) {
  ConSanResult result;
  result.visited_code_object = false;
  ProgramInventoryBuilder inventory_builder(code_object_bytes);
  result.install_program_inventory(inventory_builder.view());
  result.flavor = request.flavor.value_or(ConSanFlavor::None);
  result.moi_engine = request.moi_engine;
  result.outcome = ConSanTransformOutcome::Invalid;
  return result;
}

} // namespace

bool ConSanPipelineStageRecord::well_formed() const {
  if (!valid_stage(stage) || !valid_stage_status(status) || !code_object.valid() ||
      !valid_contract_issue(contract_issue)) {
    return false;
  }
  if (contract_issue == ConSanContractIssue::None)
    return true;
  if (status != ConSanPipelineStageStatus::Invalid &&
      status != ConSanPipelineStageStatus::Unsupported) {
    return false;
  }
  return stage == ConSanPipelineStage::Configuration ||
         stage == ConSanPipelineStage::TargetAndRuntimeCapabilities ||
         stage == ConSanPipelineStage::RuntimeBinding;
}

bool ConSanTransformIssue::well_formed() const {
  if (!valid_issue_kind(kind) || !valid_stage(stage) || !valid_contract_issue(contract_issue))
    return false;
  if (kind == ConSanTransformIssueKind::Contract)
    return contract_issue != ConSanContractIssue::None && !detail.empty();
  return contract_issue == ConSanContractIssue::None && !detail.empty();
}

const ConSanPipelineStageRecord *TransformResult::stage(ConSanPipelineStage value) const {
  if (!valid_stage(value))
    return nullptr;
  const auto found = std::ranges::find(stages, value, &ConSanPipelineStageRecord::stage);
  return found == stages.end() ? nullptr : &*found;
}

bool TransformResult::well_formed() const {
  if (!code_object.valid() || stages.size() != kConSanPipelineStages.size() ||
      !valid_contract_issue(configuration_issue)) {
    return false;
  }
  for (size_t index = 0; index < stages.size(); ++index) {
    if (stages[index].stage != kConSanPipelineStages[index] || !stages[index].well_formed() ||
        stages[index].code_object != code_object) {
      return false;
    }
  }
  const ConSanPipelineStageRecord *configuration = stage(ConSanPipelineStage::Configuration);
  if (!configuration || configuration->contract_issue != configuration_issue ||
      ((configuration_issue == ConSanContractIssue::None) !=
       (configuration->status == ConSanPipelineStageStatus::Completed))) {
    return false;
  }
  if (!program_inventory.empty() && program_inventory.code_object_id() != code_object)
    return false;
  if (evidence_requirements &&
      (!observation_plan.valid() ||
       !consan_evidence_requirements_well_formed(*evidence_requirements))) {
    return false;
  }
  if (std::ranges::any_of(issues,
                          [](const ConSanTransformIssue &issue) { return !issue.well_formed(); })) {
    return false;
  }
  switch (outcome) {
  case ConSanTransformOutcome::ModifiedValid:
    if (!final_validation_passed || replacement_bytes.empty())
      return false;
    break;
  case ConSanTransformOutcome::Unchanged:
  case ConSanTransformOutcome::Unsupported:
  case ConSanTransformOutcome::Invalid:
    if (final_validation_passed || !replacement_bytes.empty())
      return false;
    break;
  default:
    return false;
  }
  const ConSanPipelineStageRecord *complete = stage(ConSanPipelineStage::Complete);
  const ConSanPipelineStageRecord *final_validation = stage(ConSanPipelineStage::FinalValidation);
  if (!complete || complete->status != terminal_stage_status(outcome) || !final_validation)
    return false;
  if (outcome == ConSanTransformOutcome::ModifiedValid)
    return final_validation->status == ConSanPipelineStageStatus::Completed;
  if (outcome == ConSanTransformOutcome::Invalid)
    return final_validation->status == ConSanPipelineStageStatus::Invalid;
  return final_validation->status == ConSanPipelineStageStatus::NotApplicable;
}

ConSanInstallAction TransformResult::install_action(bool fail_closed) const {
  switch (outcome) {
  case ConSanTransformOutcome::ModifiedValid:
    if (final_validation_passed && !replacement_bytes.empty())
      return ConSanInstallAction::LoadReplacement;
    return ConSanInstallAction::Reject;
  case ConSanTransformOutcome::Unchanged:
    return ConSanInstallAction::LoadOriginal;
  case ConSanTransformOutcome::Unsupported:
  case ConSanTransformOutcome::Invalid:
    return fail_closed ? ConSanInstallAction::Reject : ConSanInstallAction::LoadOriginal;
  }
  return ConSanInstallAction::Reject;
}

void TransformResult::discard_replacement(std::string warning) {
  for (const ConSanProbeIntent &intent : observation_plan.probe_intents) {
    (void)coverage_ledger.set_lowering_outcome(intent.id,
                                               ConSanLoweringOutcomeKind::ResourceRejected,
                                               "runtime-owned report allocation failed");
  }
  for (ConSanSiteDispositionRecord &site : legacy_compatibility_.site_dispositions) {
    if (site.lowering_outcome == ConSanSiteLoweringOutcome::Patched) {
      site.lowering_outcome = ConSanSiteLoweringOutcome::ResourceFailed;
      site.lowering_reason = ConSanSiteLoweringReason::UnsupportedResourcePlan;
      site.resource_reason = ConSanRegisterPlanReason::InvalidRequest;
    }
  }
  outcome = ConSanTransformOutcome::Unsupported;
  final_validation_passed = false;
  replacement_bytes.clear();
  warnings.push_back(std::move(warning));
  legacy_compatibility_.outcome = outcome;
  legacy_compatibility_.modified = false;
  legacy_compatibility_.final_validation_passed = false;
  legacy_compatibility_.elf_bytes.clear();
  legacy_compatibility_.patches.clear();
  legacy_compatibility_.warnings = warnings;
  stage_record(*this, ConSanPipelineStage::LegacyLowering).status =
      ConSanPipelineStageStatus::Unsupported;
  stage_record(*this, ConSanPipelineStage::FinalValidation).status =
      ConSanPipelineStageStatus::NotApplicable;
  stage_record(*this, ConSanPipelineStage::Complete).status =
      ConSanPipelineStageStatus::Unsupported;
}

TransformResult LegacyConSanLowering::run_pristine_moi_inventory(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &disabled_mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &unbound_resources,
    bool preserve_extended_barrier_pairs) {
  ConSanOptions legacy_options =
      LegacyOptionsAdapter::adapt(request, transform_policy, runtime_policy, debug,
                                  disabled_mutation, capabilities, unbound_resources);
  legacy_options.moi_report_buffer_address.reset();
  legacy_options.moi_report_buffer_size = 0;
  legacy_options.moi_report_layout.reset();
  legacy_options.moi_report_generation = 0;
  legacy_options.moi_report_dispatch_id = 0;
  legacy_options.qualify_extended_barrier_pairs = preserve_extended_barrier_pairs;
  return publish(code_object_bytes, request, transform_policy, runtime_policy, debug,
                 disabled_mutation, capabilities, unbound_resources,
                 try_patch_consan(code_object_bytes, legacy_options));
}

TransformResult LegacyConSanLowering::retry_pristine_moi_inventory(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
    TransformResult inventory) {
  const ConSanOptions options = LegacyOptionsAdapter::adapt(
      request, transform_policy, runtime_policy, debug, mutation, capabilities, resources);
  const ConSanMoiInventoryRetryConfig retry{
      .report =
          {
              .buffer_address = options.moi_report_buffer_address,
              .buffer_size = options.moi_report_buffer_size,
              .layout = options.moi_report_layout,
              .generation = options.moi_report_generation,
              .dispatch_id = options.moi_report_dispatch_id,
          },
      .fault = ConSanFaultMutationRetryConfig::from_options(options),
  };
  ConSanResult retried = retry_patch_consan_moi_from_inventory(
      std::move(inventory.legacy_compatibility_), retry, code_object_bytes);
  return publish(code_object_bytes, request, transform_policy, runtime_policy, debug, mutation,
                 capabilities, resources, std::move(retried));
}

TransformResult
transform_consan(std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
                 const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
                 const ConSanDebugOverrides &debug, const RuntimeCapabilities &capabilities,
                 const BoundRuntimeResources &resources) {
  return transform_consan_with_mutation(code_object_bytes, request, transform_policy,
                                        runtime_policy, debug, MutationRequest{}, capabilities,
                                        resources);
}

TransformResult transform_consan_with_mutation(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources) {
  return LegacyConSanLowering::publish_optional(code_object_bytes, request, transform_policy,
                                                runtime_policy, debug, mutation, capabilities,
                                                resources, std::nullopt);
}

TransformResult
LegacyConSanLowering::publish(std::span<const uint8_t> code_object_bytes,
                              const ConSanRequest &request, const TransformPolicy &transform_policy,
                              const RuntimePolicy &runtime_policy,
                              const ConSanDebugOverrides &debug, const MutationRequest &mutation,
                              const RuntimeCapabilities &capabilities,
                              const BoundRuntimeResources &resources, ConSanResult legacy) {
  return publish_optional(code_object_bytes, request, transform_policy, runtime_policy, debug,
                          mutation, capabilities, resources, std::move(legacy));
}

TransformResult LegacyConSanLowering::publish_optional(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
    std::optional<ConSanResult> supplied_legacy) {
  TransformResult result;
  result.code_object = make_consan_code_object_id(code_object_bytes);
  initialize_stage_records(result);

  const ConSanContractIssue configuration_issue = validate_consan_configuration(
      request, transform_policy, runtime_policy, debug, mutation, resources);
  result.configuration_issue = configuration_issue;
  ConSanPipelineStageRecord &configuration =
      stage_record(result, ConSanPipelineStage::Configuration);
  if (configuration_issue != ConSanContractIssue::None) {
    configuration.status = ConSanPipelineStageStatus::Invalid;
    configuration.contract_issue = configuration_issue;
    add_contract_issue(result, ConSanPipelineStage::Configuration, configuration_issue);
    result.outcome = ConSanTransformOutcome::Invalid;
    stage_record(result, ConSanPipelineStage::FinalValidation).status =
        ConSanPipelineStageStatus::Invalid;
    stage_record(result, ConSanPipelineStage::Complete).status = ConSanPipelineStageStatus::Invalid;
    result.legacy_compatibility_ = make_rejected_legacy_result(request, code_object_bytes);
    return result;
  }
  configuration.status = ConSanPipelineStageStatus::Completed;

  ConSanPipelineStageRecord &capability_stage =
      stage_record(result, ConSanPipelineStage::TargetAndRuntimeCapabilities);
  const ConSanContractIssue backend_issue = validate_runtime_capabilities(capabilities);
  if (backend_issue != ConSanContractIssue::None) {
    capability_stage.status = ConSanPipelineStageStatus::Invalid;
    capability_stage.contract_issue = backend_issue;
    add_contract_issue(result, ConSanPipelineStage::TargetAndRuntimeCapabilities, backend_issue);
    result.outcome = ConSanTransformOutcome::Invalid;
    stage_record(result, ConSanPipelineStage::FinalValidation).status =
        ConSanPipelineStageStatus::Invalid;
    stage_record(result, ConSanPipelineStage::Complete).status = ConSanPipelineStageStatus::Invalid;
    result.legacy_compatibility_ = make_rejected_legacy_result(request, code_object_bytes);
    return result;
  }

  ConSanResult legacy;
  if (supplied_legacy) {
    legacy = std::move(*supplied_legacy);
  } else {
    const ConSanOptions legacy_options = LegacyOptionsAdapter::adapt(
        request, transform_policy, runtime_policy, debug, mutation, capabilities, resources);
    legacy = try_patch_consan(code_object_bytes, legacy_options);
  }
  result.program_inventory = legacy.program_inventory;
  result.observation_plan = std::move(legacy.observation_plan);
  result.coverage_ledger = std::move(legacy.coverage_ledger);
  result.replacement_bytes = std::move(legacy.elf_bytes);
  result.outcome = legacy.outcome;
  result.final_validation_passed = legacy.final_validation_passed;
  result.warnings = std::move(legacy.warnings);
  for (std::string &error : legacy.errors) {
    result.issues.push_back({
        .kind = ConSanTransformIssueKind::LegacyLowering,
        .stage = ConSanPipelineStage::LegacyLowering,
        .detail = std::move(error),
    });
  }
  legacy.errors.clear();
  result.legacy_compatibility_ = std::move(legacy);

  const ConSanFlavor flavor = request.flavor.value_or(ConSanFlavor::None);
  if (flavor == ConSanFlavor::None) {
    capability_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (result.legacy_compatibility_.parsed_code_object) {
    capability_stage.status = consan_target_profile(result.program_inventory.target())
                                  ? ConSanPipelineStageStatus::Completed
                                  : ConSanPipelineStageStatus::Unsupported;
  } else {
    capability_stage.status = terminal_stage_status(result.outcome);
  }

  ConSanPipelineStageRecord &inventory_stage =
      stage_record(result, ConSanPipelineStage::ProgramInventory);
  if (flavor == ConSanFlavor::None) {
    inventory_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (!result.program_inventory.empty() &&
             result.program_inventory.code_object_id() == result.code_object) {
    inventory_stage.status = ConSanPipelineStageStatus::Completed;
  } else {
    inventory_stage.status = terminal_stage_status(result.outcome);
    if (inventory_stage.status == ConSanPipelineStageStatus::Completed)
      inventory_stage.status = ConSanPipelineStageStatus::Invalid;
  }

  ConSanPipelineStageRecord &observation_stage =
      stage_record(result, ConSanPipelineStage::ObservationPlan);
  if (flavor == ConSanFlavor::None) {
    observation_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (result.observation_plan.valid()) {
    observation_stage.status = ConSanPipelineStageStatus::Completed;
  } else if (result.outcome == ConSanTransformOutcome::Unsupported) {
    observation_stage.status = ConSanPipelineStageStatus::Unsupported;
  } else if (result.outcome == ConSanTransformOutcome::Invalid) {
    observation_stage.status = ConSanPipelineStageStatus::Invalid;
  } else {
    observation_stage.status = ConSanPipelineStageStatus::NotApplicable;
  }

  const ConSanOptions legacy_options = LegacyOptionsAdapter::adapt(
      request, transform_policy, runtime_policy, debug, mutation, capabilities, resources);
  if (result.observation_plan.valid()) {
    if (flavor == ConSanFlavor::SuperCollider) {
      result.evidence_requirements = plan_consan_supercollider_evidence(result.observation_plan);
    } else if (flavor == ConSanFlavor::Moi) {
      switch (request.moi_engine) {
      case ConSanMoiEngine::RecordReplay:
        result.evidence_requirements = plan_consan_record_replay_evidence(
            result.observation_plan, make_consan_record_replay_capacity_policy(
                                         legacy_options, request.moi_auto_report_buffer_size));
        break;
      case ConSanMoiEngine::Sampled:
        result.evidence_requirements = plan_consan_sampled_evidence(
            result.observation_plan, make_consan_sampled_capacity_policy(
                                         legacy_options, request.moi_auto_report_buffer_size));
        break;
      case ConSanMoiEngine::InlineShadow:
        result.evidence_requirements = plan_consan_inline_shadow_evidence(
            result.program_inventory, result.observation_plan,
            make_consan_inline_shadow_capacity_policy(legacy_options,
                                                      request.moi_auto_report_buffer_size));
        break;
      }
    }
  }

  ConSanPipelineStageRecord &evidence_stage =
      stage_record(result, ConSanPipelineStage::EvidenceRequirements);
  if (result.evidence_requirements &&
      consan_evidence_requirements_well_formed(*result.evidence_requirements)) {
    evidence_stage.status = ConSanPipelineStageStatus::Completed;
  } else if (flavor == ConSanFlavor::None || !result.observation_plan.valid()) {
    evidence_stage.status = observation_stage.status == ConSanPipelineStageStatus::Completed
                                ? ConSanPipelineStageStatus::Invalid
                                : observation_stage.status;
  } else {
    evidence_stage.status = ConSanPipelineStageStatus::Invalid;
  }

  ConSanPipelineStageRecord &binding_stage =
      stage_record(result, ConSanPipelineStage::RuntimeBinding);
  if (!result.evidence_requirements) {
    binding_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (!evidence_is_complete(*result.evidence_requirements)) {
    binding_stage.status = ConSanPipelineStageStatus::Unsupported;
  } else if (!evidence_requires_binding(*result.evidence_requirements)) {
    binding_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (!resources.bound()) {
    binding_stage.status = ConSanPipelineStageStatus::Deferred;
  } else {
    const ConSanContractIssue requirement_issue = validate_runtime_capabilities(
        capabilities, runtime_requirements(*result.evidence_requirements));
    const bool scope_matches = resources.scope == delivery_scope(*result.evidence_requirements);
    const ConSanContractIssue resource_issue =
        validate_evidence_binding(*result.evidence_requirements, resources);
    const ConSanContractIssue binding_issue =
        requirement_issue != ConSanContractIssue::None
            ? requirement_issue
            : (!scope_matches ? ConSanContractIssue::InvalidResourceScope : resource_issue);
    if (binding_issue == ConSanContractIssue::None) {
      binding_stage.status = ConSanPipelineStageStatus::Completed;
    } else {
      binding_stage.status = ConSanPipelineStageStatus::Unsupported;
      binding_stage.contract_issue = binding_issue;
      add_contract_issue(result, ConSanPipelineStage::RuntimeBinding, binding_issue);
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.replacement_bytes.clear();
      result.final_validation_passed = false;
      result.legacy_compatibility_.patches.clear();
    }
  }

  stage_record(result, ConSanPipelineStage::LegacyLowering).status =
      terminal_stage_status(result.outcome);
  stage_record(result, ConSanPipelineStage::FinalValidation).status =
      result.outcome == ConSanTransformOutcome::ModifiedValid
          ? ConSanPipelineStageStatus::Completed
          : (result.outcome == ConSanTransformOutcome::Invalid
                 ? ConSanPipelineStageStatus::Invalid
                 : ConSanPipelineStageStatus::NotApplicable);
  stage_record(result, ConSanPipelineStage::Complete).status =
      terminal_stage_status(result.outcome);
  return result;
}

} // namespace rocjitsu

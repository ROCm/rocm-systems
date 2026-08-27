// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_pipeline.h"

#include "rocjitsu/code/patch/consan/consan_legacy_lowering.h"

#include <algorithm>
#include <map>
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
  return std::visit([](const auto &typed) { return typed.requires_binding(); }, requirements);
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

[[nodiscard]] ConSanDispatchRequirements
build_dispatch_requirements(const ProgramInventory &inventory, const ConSanCoverageLedger &coverage,
                            std::span<const ConSanPatchInfo> patches) {
  std::map<std::string, ConSanKernelDispatchRequirement> requirements_by_name;
  const auto note_kernel = [&](const ConSanKernelInfo &kernel, const auto &apply) {
    if (kernel.name.empty())
      return false;
    ConSanKernelDispatchRequirement &requirement = requirements_by_name[kernel.name];
    requirement.kernel_name = kernel.name;
    apply(requirement);
    return true;
  };
  const auto note_descriptor = [&](uint64_t descriptor_offset, const auto &apply) {
    const auto kernel = std::ranges::find(inventory.kernels(), descriptor_offset,
                                          &ConSanKernelInfo::descriptor_file_offset);
    return kernel != inventory.kernels().end() && note_kernel(*kernel, apply);
  };
  const auto note_physical_site = [&](const PhysicalSiteId &physical, const auto &apply) {
    bool attributed = false;
    for (const ConSanAccessInventorySite &site : inventory.access_sites()) {
      if (site.physical_id != physical)
        continue;
      for (uint64_t owner : site.execution_owner_descriptor_file_offsets)
        attributed |= note_descriptor(owner, apply);
    }
    for (const ConSanSyncEvent &event : inventory.sync().sync_events) {
      if (event.semantic_id.physical != physical)
        continue;
      for (const ConSanExecutionOwner &owner : event.execution_owners)
        attributed |= note_descriptor(owner.descriptor_file_offset, apply);
    }
    if (attributed)
      return;

    // Kernel-local legacy sites may predate explicit execution-owner analysis.
    // Keep this bounded fallback at the publication boundary; runtime code must
    // not repeat symbol-range ownership inference.
    const auto kernel =
        std::ranges::find_if(inventory.kernels(), [&](const ConSanKernelInfo &item) {
          return item.has_text_range && physical.original_text_offset >= item.entry_text_offset &&
                 physical.original_text_offset - item.entry_text_offset < item.code_size;
        });
    if (kernel != inventory.kernels().end())
      (void)note_kernel(*kernel, apply);
  };

  for (const ConSanIntentCoverageEntry &entry : coverage.intent_entries()) {
    if (entry.lowering != ConSanLoweringOutcomeKind::Instrumented)
      continue;
    const auto mark_instrumented = [](ConSanKernelDispatchRequirement &requirement) {
      requirement.has_instrumented_probe = true;
    };
    if (entry.intent.covered_semantic_sites.empty()) {
      note_physical_site(entry.intent.physical_site, mark_instrumented);
      continue;
    }
    for (const SemanticSiteId &semantic : entry.intent.covered_semantic_sites)
      note_physical_site(semantic.physical, mark_instrumented);
  }

  for (const ConSanPatchInfo &patch : patches) {
    const bool has_segment_requirement = patch.required_private_segment_size != 0u ||
                                         patch.dynamic_private_segment_addend != 0u ||
                                         patch.required_group_segment_size != 0u;
    if (!has_segment_requirement)
      continue;
    const auto merge_segments = [&](ConSanKernelDispatchRequirement &requirement) {
      requirement.required_private_bytes =
          std::max(requirement.required_private_bytes, patch.required_private_segment_size);
      requirement.dynamic_private_addend =
          std::max(requirement.dynamic_private_addend, patch.dynamic_private_segment_addend);
      requirement.required_group_bytes =
          std::max(requirement.required_group_bytes, patch.required_group_segment_size);
    };
    if (!patch.owner_descriptor_file_offsets.empty()) {
      for (uint64_t owner : patch.owner_descriptor_file_offsets)
        (void)note_descriptor(owner, merge_segments);
      continue;
    }
    note_physical_site(
        {.code_object = inventory.code_object_id(), .original_text_offset = patch.anchor_offset},
        merge_segments);
  }

  ConSanDispatchRequirements result;
  result.kernels.reserve(requirements_by_name.size());
  for (auto &entry : requirements_by_name)
    result.kernels.push_back(std::move(entry.second));
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
      !valid_contract_issue(configuration_issue) || !dispatch_requirements.well_formed()) {
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
    if (replacement_bytes.empty())
      return false;
    break;
  case ConSanTransformOutcome::Unchanged:
  case ConSanTransformOutcome::Unsupported:
  case ConSanTransformOutcome::Invalid:
    if (!replacement_bytes.empty() || !dispatch_requirements.kernels.empty())
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
    if (!replacement_bytes.empty())
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
  outcome = ConSanTransformOutcome::Unsupported;
  replacement_bytes.clear();
  patches.clear();
  dispatch_requirements = {};
  warnings.push_back(std::move(warning));
  moi_retry_inventory_available_ = false;
  moi_retry_preserves_extended_barrier_pairs_ = false;
  stage_record(*this, ConSanPipelineStage::LegacyLowering).status =
      ConSanPipelineStageStatus::Unsupported;
  stage_record(*this, ConSanPipelineStage::FinalValidation).status =
      ConSanPipelineStageStatus::NotApplicable;
  stage_record(*this, ConSanPipelineStage::Complete).status =
      ConSanPipelineStageStatus::Unsupported;
}

TransformResult transform_consan_pristine_moi_inventory(
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
  TransformResult result = TransformResult::publish_optional(
      code_object_bytes, request, transform_policy, runtime_policy, debug, disabled_mutation,
      capabilities, unbound_resources, try_patch_consan(code_object_bytes, legacy_options));
  result.moi_retry_inventory_available_ = true;
  result.moi_retry_preserves_extended_barrier_pairs_ = preserve_extended_barrier_pairs;
  return result;
}

TransformResult retry_transform_consan_pristine_moi_inventory(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
    TransformResult inventory) {
  const ConSanOptions options = LegacyOptionsAdapter::adapt(
      request, transform_policy, runtime_policy, debug, mutation, capabilities, resources);
  ConSanOptions inventory_options =
      LegacyOptionsAdapter::adapt(request, transform_policy, runtime_policy, debug,
                                  MutationRequest{}, capabilities, BoundRuntimeResources{});
  inventory_options.qualify_extended_barrier_pairs =
      inventory.moi_retry_preserves_extended_barrier_pairs_;
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
  ConSanResult retry_inventory;
  retry_inventory.program_inventory = inventory.program_inventory;
  retry_inventory.observation_plan = std::move(inventory.observation_plan);
  retry_inventory.coverage_ledger = std::move(inventory.coverage_ledger);
  retry_inventory.mutation = std::move(inventory.mutation);
  retry_inventory.fault_sites = std::move(inventory.fault_sites);
  retry_inventory.barrier_move_destinations = std::move(inventory.barrier_move_destinations);
  retry_inventory.fault_plans = std::move(inventory.fault_plans);
  retry_inventory.resource_plans = std::move(inventory.resource_plans);
  retry_inventory.patches = std::move(inventory.patches);
  if (!inventory.moi_retry_inventory_available_)
    retry_inventory.errors.emplace_back("ConSan MOI retry requires a pristine inventory result");
  ConSanResult retried = retry_patch_consan_moi_from_inventory(
      std::move(retry_inventory), std::move(inventory_options), retry, code_object_bytes);
  return TransformResult::publish_optional(code_object_bytes, request, transform_policy,
                                           runtime_policy, debug, mutation, capabilities, resources,
                                           std::move(retried));
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
  return TransformResult::publish_optional(code_object_bytes, request, transform_policy,
                                           runtime_policy, debug, mutation, capabilities, resources,
                                           std::nullopt);
}

TransformResult TransformResult::publish_optional(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
    std::optional<ConSanResult> supplied_mechanism) {
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
    return result;
  }

  ConSanResult legacy;
  if (supplied_mechanism) {
    legacy = std::move(*supplied_mechanism);
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
  result.mutation = std::exchange(legacy.mutation, {});
  result.fault_sites = std::move(legacy.fault_sites);
  result.barrier_move_destinations = std::move(legacy.barrier_move_destinations);
  result.fault_plans = std::move(legacy.fault_plans);
  result.resource_plans = std::move(legacy.resource_plans);
  result.patches = std::move(legacy.patches);
  result.warnings = std::move(legacy.warnings);
  if (result.outcome == ConSanTransformOutcome::ModifiedValid) {
    result.dispatch_requirements = build_dispatch_requirements(
        result.program_inventory, result.coverage_ledger, result.patches);
  }
  for (std::string &error : legacy.errors) {
    result.issues.push_back({
        .kind = ConSanTransformIssueKind::LegacyLowering,
        .stage = ConSanPipelineStage::LegacyLowering,
        .detail = std::move(error),
    });
  }
  legacy.errors.clear();

  const ConSanFlavor flavor = request.flavor.value_or(ConSanFlavor::None);
  if (flavor == ConSanFlavor::None) {
    capability_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (result.program_inventory.code_object_parsed()) {
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

  if (result.observation_plan.valid()) {
    const std::optional<uint64_t> maximum_access_probe_count =
        transform_policy.max_patches_is_expert_limit
            ? std::optional<uint64_t>{transform_policy.max_patches}
            : std::nullopt;
    if (flavor == ConSanFlavor::SuperCollider) {
      result.evidence_requirements = plan_consan_supercollider_evidence(result.observation_plan);
    } else if (flavor == ConSanFlavor::Moi) {
      switch (request.moi_engine) {
      case ConSanMoiEngine::RecordReplay:
        result.evidence_requirements = plan_consan_record_replay_evidence(
            result.observation_plan, {.caller_ceiling_bytes = request.moi_auto_report_buffer_size,
                                      .maximum_access_probe_count = maximum_access_probe_count});
        break;
      case ConSanMoiEngine::Sampled:
        result.evidence_requirements = plan_consan_sampled_evidence(
            result.observation_plan, {.caller_ceiling_bytes = request.moi_auto_report_buffer_size,
                                      .maximum_access_probe_count = maximum_access_probe_count});
        break;
      case ConSanMoiEngine::InlineShadow:
        result.evidence_requirements = plan_consan_inline_shadow_evidence(
            result.program_inventory, result.observation_plan,
            {.caller_ceiling_bytes = request.moi_auto_report_buffer_size,
             .maximum_access_probe_count = maximum_access_probe_count,
             .maximum_workgroup_lds_bytes = capabilities.max_workgroup_lds_bytes});
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
      result.dispatch_requirements = {};
      result.patches.clear();
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

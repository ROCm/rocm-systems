// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_pipeline.h"

#include "rocjitsu/code/patch/consan/consan_lowering.h"

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
          if (resources.scope != ConSanRuntimeResourceScope::CodeObject &&
              resources.scope != ConSanRuntimeResourceScope::Executable)
            return ConSanContractIssue::InvalidResourceScope;
          return typed.requires_binding() && !resources.report_buffer_address
                     ? ConSanContractIssue::InvalidResourceAddress
                     : ConSanContractIssue::None;
        } else {
          if (!resources.moi_report_buffer_address)
            return ConSanContractIssue::InvalidResourceAddress;
          // An automatic allocation carries the exact address-free layout
          // that sized it and remains live for the executable. A caller-bound
          // raw buffer intentionally has no such layout: legacy fixed-record
          // lowering validates its exact engine-specific geometry, while the
          // pipeline only requires the common header and a code-object or
          // executable lifetime. Treating that raw buffer as the automatic
          // multi-bank layout rejects valid small explicit buffers after
          // lowering has already proved and emitted their instrumentation.
          if (resources.moi_report_layout) {
            if (resources.scope != ConSanRuntimeResourceScope::Executable)
              return ConSanContractIssue::InvalidResourceScope;
            return resources.moi_report_buffer_size < typed.abi_plan.required_bytes
                       ? ConSanContractIssue::InvalidResourceSize
                       : ConSanContractIssue::None;
          }
          if (resources.scope != ConSanRuntimeResourceScope::CodeObject &&
              resources.scope != ConSanRuntimeResourceScope::Executable)
            return ConSanContractIssue::InvalidResourceScope;
          return resources.moi_report_buffer_size < sizeof(ConSanMoiReportHeader)
                     ? ConSanContractIssue::InvalidResourceSize
                     : ConSanContractIssue::None;
        }
      },
      requirements);
}

ConSanPipelineStageState &stage_record(TransformResult &result, ConSanPipelineStage stage) {
  return result.stages[static_cast<size_t>(stage)];
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
    const ConSanKernelInfo *kernel = inventory.find_kernel_by_descriptor(descriptor_offset);
    return kernel != nullptr && note_kernel(*kernel, apply);
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

bool ConSanPipelineStageState::well_formed(ConSanPipelineStage stage) const {
  if (!valid_stage(stage) || !valid_stage_status(status) || !valid_contract_issue(contract_issue)) {
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

const ConSanPipelineStageState *TransformResult::stage(ConSanPipelineStage value) const {
  if (!valid_stage(value))
    return nullptr;
  return &stages[static_cast<size_t>(value)];
}

ConSanTransformDebugReport TransformResult::debug_report() const {
  return {
      .fault_sites = private_lowering_.fault_sites,
      .barrier_move_destinations = private_lowering_.barrier_move_destinations,
      .fault_plans = private_lowering_.fault_plans,
      .resource_plans = private_lowering_.resource_plans,
      .committed_lowerings = private_lowering_.committed_lowerings,
      .patches = private_lowering_.patches,
  };
}

void TransformResult::publish_lowering_artifacts(ConSanTransformArtifacts lowering) {
  program_inventory = std::move(lowering.program_inventory);
  observation_plan = std::move(lowering.observation_plan);
  coverage_ledger = std::move(lowering.coverage_ledger);
  runtime_static_mapping = std::move(lowering.runtime_static_mapping);
  mutation = std::move(lowering.mutation);
  replacement = std::move(lowering.replacement);
  outcome = lowering.outcome;
  warnings = std::move(lowering.warnings);
  errors = std::move(lowering.errors);
  private_lowering_.fault_sites = std::move(lowering.fault_sites);
  private_lowering_.barrier_move_destinations = std::move(lowering.barrier_move_destinations);
  private_lowering_.fault_plans = std::move(lowering.fault_plans);
  private_lowering_.resource_plans = std::move(lowering.resource_plans);
  private_lowering_.moi_operating_point = std::move(lowering.moi_operating_point);
  private_lowering_.committed_lowerings = std::move(lowering.committed_lowerings);
  private_lowering_.staged_moi_sync_lowerings = std::move(lowering.staged_moi_sync_lowerings);
  private_lowering_.patches = std::move(lowering.patches);
}

ConSanTransformArtifacts TransformResult::take_lowering_artifacts() {
  ConSanTransformArtifacts lowering;
  lowering.program_inventory = std::move(program_inventory);
  lowering.observation_plan = std::move(observation_plan);
  lowering.coverage_ledger = std::move(coverage_ledger);
  lowering.runtime_static_mapping = std::move(runtime_static_mapping);
  lowering.mutation = std::move(mutation);
  lowering.replacement = std::move(replacement);
  lowering.outcome = outcome;
  lowering.warnings = std::move(warnings);
  lowering.errors = std::move(errors);
  lowering.fault_sites = std::move(private_lowering_.fault_sites);
  lowering.barrier_move_destinations = std::move(private_lowering_.barrier_move_destinations);
  lowering.fault_plans = std::move(private_lowering_.fault_plans);
  lowering.resource_plans = std::move(private_lowering_.resource_plans);
  lowering.moi_operating_point = std::move(private_lowering_.moi_operating_point);
  lowering.committed_lowerings = std::move(private_lowering_.committed_lowerings);
  lowering.staged_moi_sync_lowerings = std::move(private_lowering_.staged_moi_sync_lowerings);
  lowering.patches = std::move(private_lowering_.patches);
  return lowering;
}

bool TransformResult::well_formed() const {
  if (!code_object.valid() || !dispatch_requirements.well_formed()) {
    return false;
  }
  if (!private_lowering_.staged_moi_sync_lowerings.empty())
    return false;
  ConSanRuntimeStaticMapping expected_runtime_mapping;
  for (const ConSanCommittedLowering &commit : private_lowering_.committed_lowerings) {
    if (!consan_runtime_static_mapping_matches_commit(observation_plan, commit))
      return false;
    expected_runtime_mapping.append(commit.runtime_mapping);
  }
  if (expected_runtime_mapping != runtime_static_mapping)
    return false;
  for (size_t index = 0; index < stages.size(); ++index) {
    if (!stages[index].well_formed(kConSanPipelineStages[index])) {
      return false;
    }
  }
  const ConSanPipelineStageState *configuration = stage(ConSanPipelineStage::Configuration);
  if (!configuration || ((configuration->contract_issue == ConSanContractIssue::None) !=
                         (configuration->status == ConSanPipelineStageStatus::Completed))) {
    return false;
  }
  if (!program_inventory.empty() && program_inventory.code_object_id() != code_object)
    return false;
  if (mutation.fault.planned != private_lowering_.fault_plans.size() ||
      mutation.fault.applied > mutation.fault.planned ||
      std::ranges::any_of(private_lowering_.fault_plans, [&](const ConSanFaultMutationPlan &plan) {
        return !plan.well_formed() || plan.source_code_object != code_object;
      })) {
    return false;
  }
  if (evidence_intent_plan.has_value() != evidence_requirements.has_value())
    return false;
  if (evidence_intent_plan &&
      (!observation_plan.valid() || !evidence_intent_plan->well_formed() ||
       *evidence_intent_plan != plan_consan_evidence_intents(observation_plan) ||
       !consan_evidence_requirements_well_formed(*evidence_requirements))) {
    return false;
  }
  if (std::ranges::any_of(errors, &std::string::empty)) {
    return false;
  }
  switch (outcome) {
  case ConSanTransformOutcome::ModifiedValid:
    if (replacement.empty())
      return false;
    break;
  case ConSanTransformOutcome::Unchanged:
  case ConSanTransformOutcome::Unsupported:
  case ConSanTransformOutcome::Invalid:
    if (!replacement.empty() || !dispatch_requirements.kernels.empty())
      return false;
    break;
  default:
    return false;
  }
  return true;
}

ConSanInstallAction TransformResult::install_action(bool fail_closed) const {
  switch (outcome) {
  case ConSanTransformOutcome::ModifiedValid:
    if (!replacement.empty())
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
  outcome = ConSanTransformOutcome::Unsupported;
  replacement.clear();
  private_lowering_.patches.clear();
  runtime_static_mapping = {};
  private_lowering_.staged_moi_sync_lowerings.clear();
  std::erase_if(private_lowering_.committed_lowerings, [](const ConSanCommittedLowering &commit) {
    return commit.outcome == ConSanLoweringOutcomeKind::Instrumented;
  });
  coverage_ledger = ConSanCoverageLedger(observation_plan);
  for (const ConSanCommittedLowering &commit : private_lowering_.committed_lowerings)
    (void)coverage_ledger.publish_lowering_commit(commit);
  std::vector<ConSanCommittedLowering> rejections;
  for (const ConSanProbeIntent &intent : observation_plan.probe_intents) {
    const ConSanIntentCoverageEntry *entry = coverage_ledger.intent_entry(intent.id);
    if (entry == nullptr || entry->lowering != ConSanLoweringOutcomeKind::Pending)
      continue;
    const std::array<ConSanProbeIntentId, 1> ids = {intent.id};
    auto rejection = make_consan_committed_lowering(
        observation_plan, ids, std::span<const ConSanCommittedLoweringLocation>{},
        ConSanLoweringOutcomeKind::ResourceRejected, "runtime-owned report allocation failed");
    if (!rejection) {
      errors.emplace_back("ConSan runtime binding produced an invalid intent rejection");
      break;
    }
    rejections.push_back(std::move(*rejection));
  }
  ConSanCoverageLedger rejected_coverage = coverage_ledger;
  bool rejections_valid = true;
  for (const ConSanCommittedLowering &rejection : rejections) {
    if (!consan_runtime_static_mapping_matches_commit(observation_plan, rejection) ||
        !rejected_coverage.publish_lowering_commit(rejection)) {
      errors.emplace_back("ConSan runtime binding could not publish intent rejections");
      rejections_valid = false;
      break;
    }
  }
  if (rejections_valid) {
    coverage_ledger = std::move(rejected_coverage);
    private_lowering_.committed_lowerings.reserve(private_lowering_.committed_lowerings.size() +
                                                  rejections.size());
    private_lowering_.committed_lowerings.insert(private_lowering_.committed_lowerings.end(),
                                                 std::make_move_iterator(rejections.begin()),
                                                 std::make_move_iterator(rejections.end()));
  }
  dispatch_requirements = {};
  warnings.push_back(std::move(warning));
  moi_retry_inventory_available_ = false;
  stage_record(*this, ConSanPipelineStage::RuntimeBinding).status =
      ConSanPipelineStageStatus::Unsupported;
}

TransformResult transform_consan_pristine_moi_inventory(std::span<const uint8_t> code_object_bytes,
                                                        const ConSanRequest &request,
                                                        const TransformPolicy &transform_policy,
                                                        const RuntimePolicy &runtime_policy,
                                                        const ConSanDebugOverrides &debug,
                                                        const MutationRequest &disabled_mutation,
                                                        const RuntimeCapabilities &capabilities) {
  ConSanOptions lowering_options(request, transform_policy, debug, disabled_mutation, capabilities,
                                 BoundRuntimeResources{});
  TransformResult result = TransformResult::publish_optional(
      code_object_bytes, request, transform_policy, runtime_policy, debug, disabled_mutation,
      capabilities, BoundRuntimeResources{}, lower_consan(code_object_bytes, lowering_options));
  result.moi_retry_inventory_available_ = true;
  return result;
}

TransformResult retry_transform_consan_pristine_moi_inventory(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
    TransformResult inventory) {
  ConSanOptions retry_options(request, transform_policy, debug, mutation, capabilities, resources);
  ConSanTransformArtifacts retry_inventory = inventory.take_lowering_artifacts();
  if (!inventory.moi_retry_inventory_available_)
    retry_inventory.errors.emplace_back("ConSan MOI retry requires a pristine inventory result");
  ConSanTransformArtifacts retried = retry_patch_consan_moi_from_inventory(
      std::move(retry_inventory), std::move(retry_options), code_object_bytes);
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
    std::optional<ConSanTransformArtifacts> supplied_artifacts) {
  TransformResult result;
  result.code_object = make_consan_code_object_id(code_object_bytes);

  const ConSanContractIssue configuration_issue = validate_consan_configuration(
      request, transform_policy, runtime_policy, debug, mutation, resources);
  ConSanPipelineStageState &configuration =
      stage_record(result, ConSanPipelineStage::Configuration);
  if (configuration_issue != ConSanContractIssue::None) {
    configuration.status = ConSanPipelineStageStatus::Invalid;
    configuration.contract_issue = configuration_issue;
    result.outcome = ConSanTransformOutcome::Invalid;
    return result;
  }
  configuration.status = ConSanPipelineStageStatus::Completed;

  ConSanPipelineStageState &capability_stage =
      stage_record(result, ConSanPipelineStage::TargetAndRuntimeCapabilities);
  const ConSanContractIssue backend_issue = validate_runtime_capabilities(capabilities);
  if (backend_issue != ConSanContractIssue::None) {
    capability_stage.status = ConSanPipelineStageStatus::Invalid;
    capability_stage.contract_issue = backend_issue;
    result.outcome = ConSanTransformOutcome::Invalid;
    return result;
  }

  ConSanTransformArtifacts lowering;
  if (supplied_artifacts) {
    lowering = std::move(*supplied_artifacts);
  } else {
    const ConSanOptions lowering_options(request, transform_policy, debug, mutation, capabilities,
                                         resources);
    lowering = lower_consan(code_object_bytes, lowering_options);
  }
  result.publish_lowering_artifacts(std::move(lowering));
  if (result.outcome == ConSanTransformOutcome::ModifiedValid) {
    result.dispatch_requirements = build_dispatch_requirements(
        result.program_inventory, result.coverage_ledger, result.private_lowering_.patches);
  }
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

  ConSanPipelineStageState &inventory_stage =
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

  ConSanPipelineStageState &observation_stage =
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
    result.evidence_intent_plan = plan_consan_evidence_intents(result.observation_plan);
    const std::optional<uint64_t> maximum_access_probe_count =
        transform_policy.max_patches_is_expert_limit
            ? std::optional<uint64_t>{transform_policy.max_patches}
            : std::nullopt;
    if (flavor == ConSanFlavor::SuperCollider) {
      result.evidence_requirements =
          plan_consan_supercollider_evidence(*result.evidence_intent_plan);
    } else if (flavor == ConSanFlavor::Moi) {
      switch (request.moi_engine) {
      case ConSanMoiEngine::RecordReplay:
        result.evidence_requirements = plan_consan_record_replay_evidence(
            *result.evidence_intent_plan,
            {.caller_ceiling_bytes = request.moi_auto_report_buffer_size,
             .maximum_access_probe_count = maximum_access_probe_count});
        break;
      case ConSanMoiEngine::Sampled:
        result.evidence_requirements = plan_consan_sampled_evidence(
            *result.evidence_intent_plan,
            {.caller_ceiling_bytes = request.moi_auto_report_buffer_size,
             .maximum_access_probe_count = maximum_access_probe_count});
        break;
      case ConSanMoiEngine::InlineShadow:
        result.evidence_requirements = plan_consan_inline_shadow_evidence(
            result.program_inventory, *result.evidence_intent_plan,
            {.caller_ceiling_bytes = request.moi_auto_report_buffer_size,
             .maximum_access_probe_count = maximum_access_probe_count,
             .maximum_workgroup_lds_bytes = capabilities.max_workgroup_lds_bytes});
        break;
      }
    }
  }

  ConSanPipelineStageState &evidence_stage =
      stage_record(result, ConSanPipelineStage::EvidenceRequirements);
  if (result.evidence_intent_plan && result.evidence_intent_plan->well_formed() &&
      result.evidence_requirements &&
      consan_evidence_requirements_well_formed(*result.evidence_requirements)) {
    evidence_stage.status = ConSanPipelineStageStatus::Completed;
  } else if (flavor == ConSanFlavor::None || !result.observation_plan.valid()) {
    evidence_stage.status = observation_stage.status == ConSanPipelineStageStatus::Completed
                                ? ConSanPipelineStageStatus::Invalid
                                : observation_stage.status;
  } else {
    evidence_stage.status = ConSanPipelineStageStatus::Invalid;
  }

  ConSanPipelineStageState &binding_stage =
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
    const ConSanContractIssue resource_issue =
        validate_evidence_binding(*result.evidence_requirements, resources);
    const ConSanContractIssue binding_issue =
        requirement_issue != ConSanContractIssue::None ? requirement_issue : resource_issue;
    if (binding_issue == ConSanContractIssue::None) {
      binding_stage.status = ConSanPipelineStageStatus::Completed;
    } else {
      binding_stage.status = ConSanPipelineStageStatus::Unsupported;
      binding_stage.contract_issue = binding_issue;
      const uint64_t binding_required_bytes = std::visit(
          [&](const auto &typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, ConSanSuperColliderEvidenceRequirements>) {
              return typed.runtime_requirements.minimum_report_allocation_bytes.value_or(0u);
            } else {
              return resources.moi_report_layout
                         ? typed.runtime_requirements.minimum_report_allocation_bytes.value_or(0u)
                         : static_cast<uint64_t>(sizeof(ConSanMoiReportHeader));
            }
          },
          *result.evidence_requirements);
      result.warnings.emplace_back(
          "ConSan runtime evidence binding is unsupported: issue=" +
          std::string(consan_contract_issue_name(binding_issue)) +
          ", scope=" + std::string(consan_runtime_resource_scope_name(resources.scope)) +
          ", report-bytes=" + std::to_string(resources.moi_report_buffer_size) +
          ", required-bytes=" + std::to_string(binding_required_bytes));
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.replacement.clear();
      result.private_lowering_.patches.clear();
      result.runtime_static_mapping = {};
      result.private_lowering_.staged_moi_sync_lowerings.clear();
      std::erase_if(result.private_lowering_.committed_lowerings,
                    [](const ConSanCommittedLowering &commit) {
                      return commit.outcome == ConSanLoweringOutcomeKind::Instrumented;
                    });
      result.coverage_ledger = ConSanCoverageLedger(result.observation_plan);
      for (const ConSanCommittedLowering &commit : result.private_lowering_.committed_lowerings)
        (void)result.coverage_ledger.publish_lowering_commit(commit);
      result.dispatch_requirements = {};
    }
  }

  return result;
}

} // namespace rocjitsu

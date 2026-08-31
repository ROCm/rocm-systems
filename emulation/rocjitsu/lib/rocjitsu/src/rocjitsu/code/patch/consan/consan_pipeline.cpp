// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_pipeline.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_transform_diagnostics.h"

#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"

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

[[nodiscard]] constexpr const char *patch_diagnostic_kind_name(ConSanPatchKind kind) {
  switch (kind) {
  case ConSanPatchKind::InlineNopRewrite:
    return "inline-nop-rewrite";
  case ConSanPatchKind::InlineEndpgmRewrite:
    return "inline-endpgm-rewrite";
  case ConSanPatchKind::InlineLdsEndpgmRewrite:
    return "inline-lds-endpgm-rewrite";
  case ConSanPatchKind::InlineLdsLoadCheckTrap:
    return "inline-lds-load-check-trap";
  case ConSanPatchKind::InlineLdsStoreCheckTrap:
    return "inline-lds-store-check-trap";
  case ConSanPatchKind::LocalCaveLdsLoadCheckTrap:
    return "local-cave-lds-load-check-trap";
  case ConSanPatchKind::LocalCaveLdsStoreCheckTrap:
    return "local-cave-lds-store-check-trap";
  case ConSanPatchKind::InlineFlatLoadCheckTrap:
    return "inline-flat-load-check-trap";
  case ConSanPatchKind::InlineFlatStoreCheckTrap:
    return "inline-flat-store-check-trap";
  case ConSanPatchKind::LocalCaveFlatLoadCheckTrap:
    return "local-cave-flat-load-check-trap";
  case ConSanPatchKind::LocalCaveFlatStoreCheckTrap:
    return "local-cave-flat-store-check-trap";
  case ConSanPatchKind::InlineFlatTrapRewrite:
    return "inline-flat-trap-rewrite";
  case ConSanPatchKind::InlineBarrierNopRewrite:
    return "inline-barrier-nop-rewrite";
  case ConSanPatchKind::InlineBarrierIdScopeRewrite:
    return "inline-barrier-id-scope-rewrite";
  case ConSanPatchKind::InlineBarrierParticipantCountRewrite:
    return "inline-barrier-participant-count-rewrite";
  case ConSanPatchKind::InlineBarrierMoveSourceRewrite:
    return "inline-barrier-move-source-rewrite";
  case ConSanPatchKind::InlineBarrierMoveTargetRewrite:
    return "inline-barrier-move-target-rewrite";
  case ConSanPatchKind::InlineAtomicAddressRewrite:
    return "inline-atomic-address-rewrite";
  case ConSanPatchKind::InlineAtomicOrderRewrite:
    return "inline-atomic-order-rewrite";
  case ConSanPatchKind::InlineAtomicScopeRewrite:
    return "inline-atomic-scope-rewrite";
  case ConSanPatchKind::InlineLdsAddressRewrite:
    return "inline-lds-address-rewrite";
  case ConSanPatchKind::InlineOrdinaryOrderRewrite:
    return "inline-ordinary-order-rewrite";
  case ConSanPatchKind::InlineOrdinaryAddressRewrite:
    return "inline-ordinary-address-rewrite";
  case ConSanPatchKind::InlineOrdinaryScopeRewrite:
    return "inline-ordinary-scope-rewrite";
  case ConSanPatchKind::InlineMoiAccessRecordStore:
    return "inline-moi-access-record-store";
  case ConSanPatchKind::TrampolineMoiAccessRecordStore:
    return "trampoline-moi-access-record-store";
  case ConSanPatchKind::InlineMoiExactShadowStore:
    return "inline-moi-exact-shadow-store";
  case ConSanPatchKind::TrampolineMoiExactShadowStore:
    return "trampoline-moi-exact-shadow-store";
  case ConSanPatchKind::InlineMoiSampledWatchpointStore:
    return "inline-moi-sampled-watchpoint-store";
  case ConSanPatchKind::TrampolineMoiSampledWatchpointStore:
    return "trampoline-moi-sampled-watchpoint-store";
  case ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue:
    return "kernel-entry-moi-owner-epoch-prologue";
  case ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue:
    return "kernel-entry-moi-private-epoch-prologue";
  case ConSanPatchKind::TrampolineMoiBarrierRecord:
    return "trampoline-moi-barrier-record";
  case ConSanPatchKind::TrampolineMoiInlineEpochBarrier:
    return "trampoline-moi-inline-epoch-barrier";
  case ConSanPatchKind::TrampolineMoiInlineAtomicOrdering:
    return "trampoline-moi-inline-atomic-ordering";
  case ConSanPatchKind::TrampolineMoiAtomicRecord:
    return "trampoline-moi-atomic-record";
  case ConSanPatchKind::TrampolineMoiSampledSyncMetadata:
    return "trampoline-moi-sampled-sync-metadata";
  case ConSanPatchKind::TrampolineMoiFenceRecord:
    return "trampoline-moi-fence-record";
  case ConSanPatchKind::InlineMalformedBarrierAbort:
    return "inline-malformed-barrier-abort";
  case ConSanPatchKind::TrampolineScPerturbation:
    return "trampoline-sc-perturbation";
  case ConSanPatchKind::TrampolineScIndirectBranchIsland:
    return "trampoline-sc-indirect-branch-island";
  case ConSanPatchKind::TrampolineScDenseCallDispatcher:
    return "trampoline-sc-dense-call-dispatcher";
  case ConSanPatchKind::TrampolineScDenseEntryHost:
    return "trampoline-sc-dense-entry-host";
  case ConSanPatchKind::TrampolineScBranchRelayDonor:
    return "trampoline-sc-branch-relay-donor";
  case ConSanPatchKind::TrampolineBranchRelayReservoir:
    return "trampoline-branch-relay-reservoir";
  case ConSanPatchKind::TrampolineNopBranchRelay:
    return "trampoline-nop-branch-relay";
  case ConSanPatchKind::InlineScalarClauseNopRewrite:
    return "inline-scalar-clause-nop-rewrite";
  case ConSanPatchKind::TrampolineMoiIndirectBranchIsland:
    return "trampoline-moi-indirect-branch-island";
  case ConSanPatchKind::TrampolineNop:
    return "trampoline-nop";
  }
  return "unknown";
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

void block_pipeline_after(TransformResult &result, ConSanPipelineStage completed_or_failed) {
  const size_t first_blocked = static_cast<size_t>(completed_or_failed) + 1u;
  const size_t publication = static_cast<size_t>(ConSanPipelineStage::ResultPublication);
  for (size_t index = first_blocked; index < publication; ++index) {
    result.stages[index].status = ConSanPipelineStageStatus::Blocked;
    result.stages[index].execution_count = 0;
  }
}

template <typename PatchRange>
[[nodiscard]] ConSanDispatchRequirements
build_dispatch_requirements(const ProgramInventory &inventory, const ConSanCoverageLedger &coverage,
                            const PatchRange &patches) {
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

  for (const ConSanPatchLoweringProduct &patch : patches) {
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

/// Authoritative owner of one ConSan transform attempt.
///
/// The transaction validates each public contract, consumes the native
/// lowerer's aggregate exactly once, plans evidence, binds runtime resources,
/// and publishes the reviewed result. Keeping this state in one object makes
/// the execution order explicit and gives later cuts a place to move native
/// inventory and lowering stages without reopening the public result.
class ConSanTransformTransaction {
public:
  ConSanTransformTransaction(std::span<const uint8_t> code_object_bytes,
                             const ConSanRequest &request, const TransformPolicy &transform_policy,
                             const RuntimePolicy &runtime_policy, const ConSanDebugOverrides &debug,
                             const MutationRequest &mutation,
                             const RuntimeCapabilities &capabilities,
                             const BoundRuntimeResources &resources)
      : code_object_bytes_(code_object_bytes), request_(request),
        transform_policy_(transform_policy), runtime_policy_(runtime_policy), debug_(debug),
        mutation_(mutation), capabilities_(capabilities), resources_(resources) {}

  [[nodiscard]] TransformResult
  execute(std::optional<ConSanTransformArtifacts> supplied_lowering = std::nullopt,
          std::optional<ConSanLoweringExecution> supplied_execution = std::nullopt,
          ConSanLoweringExtent extent = ConSanLoweringExtent::Complete,
          std::optional<ConSanLoweringObservation> supplied_observation = std::nullopt);

private:
  std::span<const uint8_t> code_object_bytes_;
  const ConSanRequest &request_;
  const TransformPolicy &transform_policy_;
  const RuntimePolicy &runtime_policy_;
  const ConSanDebugOverrides &debug_;
  const MutationRequest &mutation_;
  const RuntimeCapabilities &capabilities_;
  const BoundRuntimeResources &resources_;
};

[[nodiscard]] TransformResult execute_consan_transaction(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
    std::optional<ConSanTransformArtifacts> supplied_lowering = std::nullopt,
    std::optional<ConSanLoweringExecution> supplied_execution = std::nullopt,
    ConSanLoweringExtent extent = ConSanLoweringExtent::Complete,
    std::optional<ConSanLoweringObservation> supplied_observation = std::nullopt) {
  return ConSanTransformTransaction(code_object_bytes, request, transform_policy, runtime_policy,
                                    debug, mutation, capabilities, resources)
      .execute(std::move(supplied_lowering), std::move(supplied_execution), extent,
               std::move(supplied_observation));
}

bool ConSanPipelineStageState::well_formed(ConSanPipelineStage stage) const {
  if (!valid_stage(stage) || !valid_stage_status(status) || !valid_contract_issue(contract_issue)) {
    return false;
  }
  if (status == ConSanPipelineStageStatus::Blocked && execution_count != 0u)
    return false;
  if (status != ConSanPipelineStageStatus::Blocked &&
      status != ConSanPipelineStageStatus::NotApplicable && execution_count == 0u) {
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

ConSanTransformDiagnosticReport consan_transform_diagnostic_report(const TransformResult &result) {
  ConSanResourcePlanSummary resource_summary =
      summarize_consan_resource_plans(result.private_lowering_.resource_plans);
  for (const ConSanPatchAbiEffects &patch : result.private_lowering_.patches)
    accumulate_consan_emitted_spill(resource_summary, patch);
  ConSanTransformDiagnosticReport report;
  report.resource_summary = resource_summary;

  report.fault_sites.reserve(result.private_lowering_.fault_sites.size());
  for (const ConSanFaultSite &site : result.private_lowering_.fault_sites) {
    report.fault_sites.push_back({
        .kind = site.kind,
        .identity = site.identity,
        .container_name = site.container_name,
        .in_kernel = site.in_kernel,
        .occurrence = site.occurrence,
        .text_offset = site.text_offset,
        .file_offset = site.file_offset,
        .size = site.size,
        .width_bits = site.width_bits,
        .mnemonic = site.mnemonic,
        .semantic_role = site.semantic_role,
        .decoded_operands = site.decoded_operands,
        .ordinary_memory_support_reason = site.ordinary_memory_support_reason,
        .sync_event_identity = site.sync_event_identity,
        .sync_sequence_identity = site.sync_sequence_identity,
        .sync_confidence = site.sync_confidence,
        .sync_memory_role = site.sync_memory_role,
        .execution_owners = site.execution_owners,
    });
  }

  report.barrier_move_destinations.reserve(
      result.private_lowering_.barrier_move_destinations.size());
  for (const ConSanBarrierMoveDestination &destination :
       result.private_lowering_.barrier_move_destinations) {
    report.barrier_move_destinations.push_back({
        .identity = destination.identity,
        .container_name = destination.container_name,
        .in_kernel = destination.in_kernel,
        .basic_block_index = destination.basic_block_index,
        .text_offset = destination.text_offset,
        .file_offset = destination.file_offset,
        .size = destination.size,
        .mnemonic = destination.mnemonic,
        .memory_operation = destination.memory_operation,
        .issue = destination.issue,
        .issue_detail = destination.issue_detail,
        .cfg_contract = destination.cfg_contract,
        .structured_guard_block_index = destination.structured_guard_block_index,
        .structured_source_block_index = destination.structured_source_block_index,
        .structured_guard_offset = destination.structured_guard_offset,
        .structured_source_offset = destination.structured_source_offset,
        .execution_owners = destination.execution_owners,
    });
  }

  report.fault_mutations.reserve(result.private_lowering_.fault_plans.size());
  for (const ConSanFaultMutationPlan &plan : result.private_lowering_.fault_plans) {
    report.fault_mutations.push_back({
        .kind = plan.kind,
        .primary_identity = plan.primary_identity,
        .companion_identity = plan.companion_identity,
        .logical_sequence_identity = plan.logical_sequence_identity,
        .ordered_member_identities = plan.ordered_member_identities,
        .destination_identity = plan.destination_identity,
        .barrier_move_direction = plan.barrier_move_direction,
        .barrier_move_cfg_contract = plan.barrier_move_cfg_contract,
        .original_barrier_id = plan.original_barrier_id,
        .target_barrier_id = plan.target_barrier_id,
        .original_barrier_scope = plan.original_barrier_scope,
        .target_barrier_scope = plan.target_barrier_scope,
        .target_address_vgpr = plan.target_address_vgpr,
    });
  }

  for (const ConSanCandidateResourcePlan &plan : result.private_lowering_.resource_plans) {
    if (plan.source == ConSanRegisterAllocationSource::Unsupported) {
      const auto matching_failure = [&](const ConSanResourceFailureDiagnostic &failure) {
        return failure.site_kind == plan.site_kind && failure.reason == plan.reason;
      };
      auto failure = std::ranges::find_if(report.resource_failures, matching_failure);
      if (failure == report.resource_failures.end()) {
        report.resource_failures.push_back({
            .site_kind = plan.site_kind,
            .reason = plan.reason,
            .count = 1u,
            .min_scratch_vgprs = plan.scratch_vgpr_count,
            .max_scratch_vgprs = plan.scratch_vgpr_count,
            .min_current_vgprs = plan.current_vgpr_count,
            .max_current_vgprs = plan.current_vgpr_count,
            .min_max_referenced_vgprs = plan.max_referenced_vgpr_count,
            .max_max_referenced_vgprs = plan.max_referenced_vgpr_count,
            .min_ordinary_vgpr_limit = plan.ordinary_vgpr_limit,
            .max_ordinary_vgpr_limit = plan.ordinary_vgpr_limit,
            .min_required_vgprs = plan.required_vgpr_count,
            .max_required_vgprs = plan.required_vgpr_count,
            .min_owners = plan.owner_descriptor_file_offsets.size(),
            .max_owners = plan.owner_descriptor_file_offsets.size(),
            .has_indirect_vgpr_access = plan.has_indirect_vgpr_access,
        });
      } else {
        ++failure->count;
        failure->min_scratch_vgprs = std::min(failure->min_scratch_vgprs, plan.scratch_vgpr_count);
        failure->max_scratch_vgprs = std::max(failure->max_scratch_vgprs, plan.scratch_vgpr_count);
        failure->min_current_vgprs = std::min(failure->min_current_vgprs, plan.current_vgpr_count);
        failure->max_current_vgprs = std::max(failure->max_current_vgprs, plan.current_vgpr_count);
        failure->min_max_referenced_vgprs =
            std::min(failure->min_max_referenced_vgprs, plan.max_referenced_vgpr_count);
        failure->max_max_referenced_vgprs =
            std::max(failure->max_max_referenced_vgprs, plan.max_referenced_vgpr_count);
        failure->min_ordinary_vgpr_limit =
            std::min(failure->min_ordinary_vgpr_limit, plan.ordinary_vgpr_limit);
        failure->max_ordinary_vgpr_limit =
            std::max(failure->max_ordinary_vgpr_limit, plan.ordinary_vgpr_limit);
        failure->min_required_vgprs =
            std::min(failure->min_required_vgprs, plan.required_vgpr_count);
        failure->max_required_vgprs =
            std::max(failure->max_required_vgprs, plan.required_vgpr_count);
        failure->min_owners =
            std::min(failure->min_owners, plan.owner_descriptor_file_offsets.size());
        failure->max_owners =
            std::max(failure->max_owners, plan.owner_descriptor_file_offsets.size());
        failure->has_indirect_vgpr_access |= plan.has_indirect_vgpr_access;
      }
    }
    for (size_t alternative_index = 0; alternative_index < plan.alternatives.size();
         ++alternative_index) {
      const ConSanResourcePlanAlternative &alternative = plan.alternatives[alternative_index];
      report.resource_alternatives.push_back({
          .site_kind = plan.site_kind,
          .candidate_index = plan.candidate_index,
          .text_offset = plan.text_offset,
          .attempt_index = alternative_index,
          .kind = alternative.kind,
          .scratch_vgpr_count = alternative.scratch_vgpr_count,
          .source = alternative.source,
          .reason = alternative.reason,
          .outcome = consan_resource_plan_alternative_outcome(plan, alternative),
      });
    }
  }
  std::ranges::sort(report.resource_failures, [](const ConSanResourceFailureDiagnostic &lhs,
                                                 const ConSanResourceFailureDiagnostic &rhs) {
    return std::pair(lhs.site_kind, lhs.reason) < std::pair(rhs.site_kind, rhs.reason);
  });

  report.patches.reserve(result.private_lowering_.patches.size());
  for (const ConSanPatchLoweringProduct &patch : result.private_lowering_.patches) {
    report.patches.push_back({
        .kind = patch_diagnostic_kind_name(patch.kind),
        .anchor_offset = patch.anchor_offset,
        .trampoline_offset = patch.trampoline_offset,
        .original_size = patch.original_size,
        .trampoline_size = patch.trampoline_size,
        .scratch_vgpr = patch.scratch_vgpr,
        .scalar_vcc_spill_sgpr = patch.scalar_vcc_spill_sgpr,
        .scalar_vcc_spill_vgpr = patch.scalar_vcc_spill_vgpr,
        .scalar_vcc_spill_vgpr_count = patch.scalar_vcc_spill_vgpr_count,
        .persistent_epoch_private_offset = patch.persistent_epoch_private_offset,
        .spilled_vgpr_count = patch.spilled_vgpr_count,
        .required_private_segment_size = patch.required_private_segment_size,
        .dynamic_private_segment_addend = patch.dynamic_private_segment_addend,
        .workgroup_shadow_base = patch.workgroup_shadow_base,
        .workgroup_shadow_size = patch.workgroup_shadow_size,
        .required_group_segment_size = patch.required_group_segment_size,
        .sampled_first_slot = patch.sampled_first_slot,
        .sampled_window_bank_count = patch.sampled_window_bank_count,
        .sampled_access_kind = patch.sampled_access_kind,
    });
  }
  return report;
}

void TransformResult::publish_lowering_artifacts(ConSanTransformArtifacts lowering) {
  program_inventory = std::move(lowering.program_inventory);
  observation_plan = std::move(lowering.observation_plan);
  runtime_static_mapping = lowering.coverage_ledger.runtime_static_mapping();
  coverage_ledger = std::move(lowering.coverage_ledger);
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
  private_lowering_.patches = std::move(lowering.patches);
}

ConSanTransformArtifacts TransformResult::take_lowering_artifacts() {
  ConSanTransformArtifacts lowering;
  lowering.program_inventory = std::move(program_inventory);
  lowering.observation_plan = std::move(observation_plan);
  lowering.coverage_ledger = std::move(coverage_ledger);
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
  lowering.patches = std::move(private_lowering_.patches);
  return lowering;
}

bool TransformResult::well_formed() const {
  if (!code_object.valid() || !dispatch_requirements.well_formed()) {
    return false;
  }
  if (!coverage_ledger.matches_plan(observation_plan) ||
      coverage_ledger.runtime_static_mapping() != runtime_static_mapping)
    return false;
  for (size_t index = 0; index < stages.size(); ++index) {
    if (!stages[index].well_formed(kConSanPipelineStages[index])) {
      return false;
    }
  }
  const ConSanPipelineStageState *configuration = stage(ConSanPipelineStage::Configuration);
  const ConSanPipelineStageState *publication = stage(ConSanPipelineStage::ResultPublication);
  if (!configuration ||
      ((configuration->contract_issue == ConSanContractIssue::None) !=
       (configuration->status == ConSanPipelineStageStatus::Completed)) ||
      !publication || publication->status != ConSanPipelineStageStatus::Completed) {
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
  coverage_ledger.discard_instrumented_lowerings();
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
  if (!rejected_coverage.publish_lowering_commits(std::move(rejections))) {
    errors.emplace_back("ConSan runtime binding could not publish intent rejections");
    rejections_valid = false;
  }
  if (rejections_valid) {
    coverage_ledger = std::move(rejected_coverage);
  }
  dispatch_requirements = {};
  warnings.push_back(std::move(warning));
  stage_record(*this, ConSanPipelineStage::RuntimeBinding).status =
      ConSanPipelineStageStatus::Unsupported;
}

bool ConSanDeferredBinding::well_formed() const {
  const bool injected_executor = strategy_ == ResumeStrategy::InvokeExecutor;
  if ((!injected_executor && !inventory_result_.well_formed()) ||
      (injected_executor && !inventory_result_.code_object.valid()) ||
      validate_consan_configuration(request_, transform_policy_, runtime_policy_, debug_,
                                    inventory_mutation_,
                                    BoundRuntimeResources{}) != ConSanContractIssue::None ||
      validate_runtime_capabilities(capabilities_) != ConSanContractIssue::None) {
    return false;
  }
  const MutationRequest expected_inventory_mutation =
      requested_mutation_.has_fault_mutation() && !requested_mutation_.fault_dry_run
          ? without_consan_fault_mutations(requested_mutation_)
          : requested_mutation_;
  if (inventory_mutation_ != expected_inventory_mutation ||
      (!injected_executor &&
       inventory_result_.code_object != inventory_result_.program_inventory.code_object_id()) ||
      !inventory_result_.evidence_intent_plan || !inventory_result_.evidence_requirements ||
      !evidence_is_complete(*inventory_result_.evidence_requirements) ||
      !evidence_requires_binding(*inventory_result_.evidence_requirements)) {
    return false;
  }
  const ConSanPipelineStageState *binding =
      inventory_result_.stage(ConSanPipelineStage::RuntimeBinding);
  if (binding == nullptr || binding->status != ConSanPipelineStageStatus::Deferred ||
      binding->contract_issue != ConSanContractIssue::None) {
    return false;
  }
  const auto engine = request_.flavor
                          ? consan_capability_engine(*request_.flavor, request_.moi_engine)
                          : std::nullopt;
  if (!engine || inventory_result_.observation_plan.engine != *engine)
    return false;
  switch (strategy_) {
  case ResumeStrategy::RelowerFromInput:
    return request_.flavor == ConSanFlavor::SuperCollider && executor_ == nullptr;
  case ResumeStrategy::RetryMoiInventory:
    return request_.flavor == ConSanFlavor::Moi && executor_ == nullptr;
  case ResumeStrategy::InvokeExecutor:
    return (request_.flavor == ConSanFlavor::Moi ||
            request_.flavor == ConSanFlavor::SuperCollider) &&
           executor_ != nullptr;
  }
  return false;
}

ConSanAutomaticTransformPreparation prepare_consan_automatic_transform(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, ConSanTransformExecutor executor) {
  const MutationRequest inventory_mutation =
      mutation.has_fault_mutation() && !mutation.fault_dry_run
          ? without_consan_fault_mutations(mutation)
          : mutation;
  const ConSanFlavor flavor = request.flavor.value_or(ConSanFlavor::None);
  const ConSanDeferredBinding::ResumeStrategy strategy =
      executor != nullptr           ? ConSanDeferredBinding::ResumeStrategy::InvokeExecutor
      : flavor == ConSanFlavor::Moi ? ConSanDeferredBinding::ResumeStrategy::RetryMoiInventory
                                    : ConSanDeferredBinding::ResumeStrategy::RelowerFromInput;
  TransformResult inventory =
      executor != nullptr
          ? executor(code_object_bytes, request, transform_policy, runtime_policy, debug,
                     inventory_mutation, capabilities, BoundRuntimeResources{})
          : execute_consan_transaction(code_object_bytes, request, transform_policy, runtime_policy,
                                       debug, inventory_mutation, capabilities,
                                       BoundRuntimeResources{}, std::nullopt, std::nullopt,
                                       ConSanLoweringExtent::ThroughProgramInventory);
  const ConSanPipelineStageState *binding = inventory.stage(ConSanPipelineStage::RuntimeBinding);
  const ConSanPipelineStageState *lowering =
      inventory.stage(ConSanPipelineStage::ResourceSolvingAndLowering);
  const bool inventory_acceptable =
      executor != nullptr ? inventory.code_object.valid() : inventory.well_formed();
  if (!inventory_acceptable || binding == nullptr ||
      binding->status != ConSanPipelineStageStatus::Deferred || !inventory.evidence_requirements ||
      !evidence_requires_binding(*inventory.evidence_requirements)) {
    const bool continue_without_binding =
        executor == nullptr && binding != nullptr && lowering != nullptr &&
        binding->status == ConSanPipelineStageStatus::NotApplicable &&
        lowering->status == ConSanPipelineStageStatus::Blocked;
    if (inventory_acceptable && (continue_without_binding || inventory_mutation != mutation) &&
        inventory.outcome != ConSanTransformOutcome::Invalid &&
        inventory.outcome != ConSanTransformOutcome::Unsupported) {
      return executor != nullptr
                 ? executor(code_object_bytes, request, transform_policy, runtime_policy, debug,
                            mutation, capabilities, BoundRuntimeResources{})
             : mutation.has_mutation()
                 ? transform_consan_with_mutation(code_object_bytes, request, transform_policy,
                                                  runtime_policy, debug, mutation, capabilities,
                                                  BoundRuntimeResources{})
                 : transform_consan(code_object_bytes, request, transform_policy, runtime_policy,
                                    debug, capabilities, BoundRuntimeResources{});
    }
    return inventory;
  }
  return ConSanDeferredBinding(request, transform_policy, runtime_policy, debug, mutation,
                               inventory_mutation, capabilities, strategy, executor,
                               std::move(inventory));
}

TransformResult resume_consan_automatic_transform(std::span<const uint8_t> code_object_bytes,
                                                  const BoundRuntimeResources &resources,
                                                  ConSanDeferredBinding deferred) {
  const auto invalid_resume = [&](std::string error) {
    ConSanTransformArtifacts invalid;
    invalid.outcome = ConSanTransformOutcome::Invalid;
    invalid.errors.push_back(std::move(error));
    return execute_consan_transaction(code_object_bytes, deferred.request_,
                                      deferred.transform_policy_, deferred.runtime_policy_,
                                      deferred.debug_, deferred.requested_mutation_,
                                      deferred.capabilities_, resources, std::move(invalid));
  };
  if (!deferred.well_formed())
    return invalid_resume("ConSan automatic resume received an invalid deferred-binding value");
  if (deferred.code_object() != make_consan_code_object_id(code_object_bytes))
    return invalid_resume("ConSan automatic resume does not match the prepared input image");

  const ConSanEvidenceRequirements &evidence = *deferred.inventory_result_.evidence_requirements;
  const ConSanContractIssue capability_issue =
      validate_runtime_capabilities(deferred.capabilities_, runtime_requirements(evidence));
  const ConSanContractIssue resource_contract_issue = validate_bound_runtime_resources(resources);
  const ConSanContractIssue evidence_binding_issue = validate_evidence_binding(evidence, resources);
  const ConSanContractIssue binding_issue =
      capability_issue != ConSanContractIssue::None          ? capability_issue
      : resource_contract_issue != ConSanContractIssue::None ? resource_contract_issue
                                                             : evidence_binding_issue;
  if (binding_issue != ConSanContractIssue::None) {
    TransformResult rejected = std::move(deferred.inventory_result_);
    ConSanPipelineStageState &binding = stage_record(rejected, ConSanPipelineStage::RuntimeBinding);
    ++binding.execution_count;
    rejected.discard_replacement(
        "ConSan automatic resume rejected runtime evidence binding: issue=" +
        std::string(consan_contract_issue_name(binding_issue)));
    binding.contract_issue = binding_issue;
    ++stage_record(rejected, ConSanPipelineStage::ResultPublication).execution_count;
    return rejected;
  }

  TransformResult result = std::move(deferred.inventory_result_);
  ConSanPipelineStageState &binding = stage_record(result, ConSanPipelineStage::RuntimeBinding);
  ++binding.execution_count;
  binding.status = ConSanPipelineStageStatus::Completed;
  const ConSanLoweringObservation observation = {
      .plan = result.observation_plan,
      .initial_coverage = result.coverage_ledger,
  };
  ConSanLoweringExecution execution;
  ConSanTransformArtifacts completed;

  switch (deferred.strategy_) {
  case ConSanDeferredBinding::ResumeStrategy::RetryMoiInventory: {
    ConSanOptions retry_options(deferred.request_, deferred.transform_policy_, deferred.debug_,
                                deferred.requested_mutation_, deferred.capabilities_, resources);
    ConSanTransformArtifacts retry_inventory = result.take_lowering_artifacts();
    completed =
        retry_patch_consan_moi_from_inventory(std::move(retry_inventory), std::move(retry_options),
                                              code_object_bytes, &execution, &observation);
    break;
  }
  case ConSanDeferredBinding::ResumeStrategy::RelowerFromInput: {
    const ConSanOptions options(deferred.request_, deferred.transform_policy_, deferred.debug_,
                                deferred.requested_mutation_, deferred.capabilities_, resources);
    completed = lower_consan(code_object_bytes, options, &execution, ConSanLoweringExtent::Complete,
                             &observation);
    break;
  }
  case ConSanDeferredBinding::ResumeStrategy::InvokeExecutor:
    return deferred.executor_(code_object_bytes, deferred.request_, deferred.transform_policy_,
                              deferred.runtime_policy_, deferred.debug_,
                              deferred.requested_mutation_, deferred.capabilities_, resources);
  }

  result.publish_lowering_artifacts(std::move(completed));
  stage_record(result, ConSanPipelineStage::ProgramInventory).execution_count +=
      execution.program_inventory_passes;
  stage_record(result, ConSanPipelineStage::ObservationPlan).execution_count +=
      execution.observation_plan_passes;
  ConSanPipelineStageState &lowering =
      stage_record(result, ConSanPipelineStage::ResourceSolvingAndLowering);
  ConSanPipelineStageState &validation = stage_record(result, ConSanPipelineStage::FinalValidation);
  lowering.execution_count += execution.resource_solving_and_lowering_passes;
  validation.execution_count += execution.final_validation_passes;
  lowering.status = lowering.execution_count == 0u ? ConSanPipelineStageStatus::Blocked
                                                   : terminal_stage_status(result.outcome);
  validation.status = validation.execution_count == 0u ? ConSanPipelineStageStatus::Blocked
                                                       : terminal_stage_status(result.outcome);
  if (result.outcome == ConSanTransformOutcome::ModifiedValid) {
    result.dispatch_requirements = build_dispatch_requirements(
        result.program_inventory, result.coverage_ledger, result.private_lowering_.patches);
  }
  ++stage_record(result, ConSanPipelineStage::ResultPublication).execution_count;
  return result;
}

TransformResult cancel_consan_automatic_transform(ConSanDeferredBinding deferred,
                                                  std::string warning) {
  TransformResult result = std::move(deferred.inventory_result_);
  result.discard_replacement(std::move(warning));
  return result;
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
  return execute_consan_transaction(code_object_bytes, request, transform_policy, runtime_policy,
                                    debug, mutation, capabilities, resources);
}

TransformResult
ConSanTransformTransaction::execute(std::optional<ConSanTransformArtifacts> supplied_artifacts,
                                    std::optional<ConSanLoweringExecution> supplied_execution,
                                    ConSanLoweringExtent extent,
                                    std::optional<ConSanLoweringObservation> supplied_observation) {
  const std::span<const uint8_t> code_object_bytes = code_object_bytes_;
  const ConSanRequest &request = request_;
  const TransformPolicy &transform_policy = transform_policy_;
  const RuntimePolicy &runtime_policy = runtime_policy_;
  const ConSanDebugOverrides &debug = debug_;
  const MutationRequest &mutation = mutation_;
  const RuntimeCapabilities &capabilities = capabilities_;
  const BoundRuntimeResources &resources = resources_;
  TransformResult result;
  result.code_object = make_consan_code_object_id(code_object_bytes);

  const ConSanContractIssue configuration_issue = validate_consan_configuration(
      request, transform_policy, runtime_policy, debug, mutation, resources);
  ConSanPipelineStageState &configuration =
      stage_record(result, ConSanPipelineStage::Configuration);
  configuration.execution_count = 1;
  if (configuration_issue != ConSanContractIssue::None) {
    configuration.status = ConSanPipelineStageStatus::Invalid;
    configuration.contract_issue = configuration_issue;
    result.outcome = ConSanTransformOutcome::Invalid;
    block_pipeline_after(result, ConSanPipelineStage::Configuration);
    ConSanPipelineStageState &publication =
        stage_record(result, ConSanPipelineStage::ResultPublication);
    publication.status = ConSanPipelineStageStatus::Completed;
    publication.execution_count = 1;
    return result;
  }
  configuration.status = ConSanPipelineStageStatus::Completed;

  ConSanPipelineStageState &capability_stage =
      stage_record(result, ConSanPipelineStage::TargetAndRuntimeCapabilities);
  capability_stage.execution_count = 1;
  const ConSanContractIssue backend_issue = validate_runtime_capabilities(capabilities);
  if (backend_issue != ConSanContractIssue::None) {
    capability_stage.status = ConSanPipelineStageStatus::Invalid;
    capability_stage.contract_issue = backend_issue;
    result.outcome = ConSanTransformOutcome::Invalid;
    block_pipeline_after(result, ConSanPipelineStage::TargetAndRuntimeCapabilities);
    ConSanPipelineStageState &publication =
        stage_record(result, ConSanPipelineStage::ResultPublication);
    publication.status = ConSanPipelineStageStatus::Completed;
    publication.execution_count = 1;
    return result;
  }

  ConSanLoweringExecution execution = supplied_execution.value_or(ConSanLoweringExecution{});
  ConSanTransformArtifacts lowering;
  const bool staged_native_lowering = !supplied_artifacts && !supplied_observation &&
                                      extent == ConSanLoweringExtent::Complete &&
                                      !mutation.has_mutation();
  const ConSanLoweringExtent initial_extent =
      staged_native_lowering ? ConSanLoweringExtent::ThroughProgramInventory : extent;
  if (supplied_artifacts) {
    lowering = std::move(*supplied_artifacts);
  } else {
    const ConSanOptions lowering_options(request, transform_policy, debug, mutation, capabilities,
                                         resources);
    ConSanLoweringExecution native_execution;
    lowering = lower_consan(code_object_bytes, lowering_options, &native_execution, initial_extent,
                            supplied_observation ? &*supplied_observation : nullptr);
    execution.append(native_execution);
  }
  const ConSanFlavor flavor = request.flavor.value_or(ConSanFlavor::None);
  if (initial_extent == ConSanLoweringExtent::ThroughProgramInventory &&
      execution.program_inventory_passes != 0u && execution.observation_plan_passes == 0u &&
      flavor != ConSanFlavor::None && lowering.errors.empty() &&
      lowering.program_inventory.code_object_parsed() &&
      consan_target_profile(lowering.program_inventory.target()) != nullptr) {
    ConSanObservationProduct observation =
        assemble_consan_observation_product(lowering.program_inventory, request, debug);
    lowering.observation_plan = std::move(observation.plan);
    lowering.coverage_ledger = std::move(observation.initial_coverage);
    lowering.errors.insert(lowering.errors.end(),
                           std::make_move_iterator(observation.diagnostics.begin()),
                           std::make_move_iterator(observation.diagnostics.end()));
    if (!lowering.errors.empty())
      lowering.outcome = ConSanTransformOutcome::Invalid;
    execution.note_observation_plan();
  }
  result.publish_lowering_artifacts(std::move(lowering));
  ConSanTransformOutcome lowerer_outcome = result.outcome;
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
  inventory_stage.execution_count = execution.program_inventory_passes;
  if (flavor == ConSanFlavor::None) {
    inventory_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (execution.program_inventory_passes == 0u) {
    inventory_stage.status = ConSanPipelineStageStatus::Blocked;
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
  observation_stage.execution_count = execution.observation_plan_passes;
  if (flavor == ConSanFlavor::None) {
    observation_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (execution.observation_plan_passes == 0u) {
    observation_stage.status = ConSanPipelineStageStatus::Blocked;
  } else if (result.observation_plan.valid()) {
    observation_stage.status = ConSanPipelineStageStatus::Completed;
  } else if (result.outcome == ConSanTransformOutcome::Unsupported) {
    observation_stage.status = ConSanPipelineStageStatus::Unsupported;
  } else if (result.outcome == ConSanTransformOutcome::Invalid) {
    observation_stage.status = ConSanPipelineStageStatus::Invalid;
  } else {
    observation_stage.status = ConSanPipelineStageStatus::NotApplicable;
  }

  if (observation_stage.status == ConSanPipelineStageStatus::Completed) {
    result.evidence_intent_plan = plan_consan_evidence_intents(result.observation_plan);
    const std::optional<uint64_t> maximum_access_probe_count =
        transform_policy.max_patches_is_expert_limit
            ? std::optional<uint64_t>{transform_policy.max_patches}
            : std::nullopt;
    if (flavor == ConSanFlavor::SuperCollider) {
      result.evidence_requirements = plan_consan_supercollider_evidence(
          *result.evidence_intent_plan, request.supercollider_evidence_mode);
    } else if (flavor == ConSanFlavor::Moi) {
      result.evidence_requirements = consan_moi_impl::plan_moi_evidence_requirements(
          request.moi_engine, {.program_inventory = result.program_inventory,
                               .evidence_intents = *result.evidence_intent_plan,
                               .requested_report_buffer_size = request.moi_auto_report_buffer_size,
                               .maximum_access_probe_count = maximum_access_probe_count,
                               .maximum_workgroup_lds_bytes = capabilities.max_workgroup_lds_bytes,
                               .dynamic_access_records = request.moi_dynamic_access_records});
    }
  }

  ConSanPipelineStageState &evidence_stage =
      stage_record(result, ConSanPipelineStage::EvidenceRequirements);
  if (flavor == ConSanFlavor::None) {
    evidence_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (observation_stage.status != ConSanPipelineStageStatus::Completed) {
    evidence_stage.status = ConSanPipelineStageStatus::Blocked;
  } else if (result.evidence_intent_plan && result.evidence_intent_plan->well_formed() &&
             result.evidence_requirements &&
             consan_evidence_requirements_well_formed(*result.evidence_requirements)) {
    evidence_stage.execution_count = 1;
    evidence_stage.status = ConSanPipelineStageStatus::Completed;
  } else {
    evidence_stage.execution_count = 1;
    evidence_stage.status = ConSanPipelineStageStatus::Invalid;
  }

  ConSanPipelineStageState &binding_stage =
      stage_record(result, ConSanPipelineStage::RuntimeBinding);
  if (flavor == ConSanFlavor::None) {
    binding_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (evidence_stage.status != ConSanPipelineStageStatus::Completed ||
             !result.evidence_requirements) {
    binding_stage.status = ConSanPipelineStageStatus::Blocked;
  } else if (!evidence_is_complete(*result.evidence_requirements)) {
    binding_stage.execution_count = 1;
    binding_stage.status = ConSanPipelineStageStatus::Unsupported;
  } else if (!evidence_requires_binding(*result.evidence_requirements)) {
    binding_stage.execution_count = 1;
    binding_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else if (!resources.bound()) {
    binding_stage.execution_count = 1;
    binding_stage.status = ConSanPipelineStageStatus::Deferred;
  } else {
    binding_stage.execution_count = 1;
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
      result.coverage_ledger.discard_instrumented_lowerings();
      result.dispatch_requirements = {};
    }
  }

  const bool binding_ready = binding_stage.status == ConSanPipelineStageStatus::Completed ||
                             binding_stage.status == ConSanPipelineStageStatus::NotApplicable;
  if (staged_native_lowering && binding_ready &&
      result.outcome != ConSanTransformOutcome::Invalid &&
      result.outcome != ConSanTransformOutcome::Unsupported) {
    ConSanLoweringObservation observation = {
        .plan = result.observation_plan,
        .initial_coverage = result.coverage_ledger,
    };
    const ConSanOptions lowering_options(request, transform_policy, debug, mutation, capabilities,
                                         resources);
    ConSanLoweringExecution native_execution;
    ConSanTransformArtifacts completed =
        lower_consan(code_object_bytes, lowering_options, &native_execution,
                     ConSanLoweringExtent::Complete, &observation);
    execution.append(native_execution);
    result.publish_lowering_artifacts(std::move(completed));
    lowerer_outcome = result.outcome;
    inventory_stage.execution_count = execution.program_inventory_passes;
  }

  if (result.outcome == ConSanTransformOutcome::ModifiedValid) {
    result.dispatch_requirements = build_dispatch_requirements(
        result.program_inventory, result.coverage_ledger, result.private_lowering_.patches);
  }

  ConSanPipelineStageState &lowering_stage =
      stage_record(result, ConSanPipelineStage::ResourceSolvingAndLowering);
  ConSanPipelineStageState &validation_stage =
      stage_record(result, ConSanPipelineStage::FinalValidation);
  lowering_stage.execution_count = execution.resource_solving_and_lowering_passes;
  validation_stage.execution_count = execution.final_validation_passes;
  if (flavor == ConSanFlavor::None) {
    lowering_stage.status = ConSanPipelineStageStatus::NotApplicable;
    validation_stage.status = ConSanPipelineStageStatus::NotApplicable;
  } else {
    lowering_stage.status = execution.resource_solving_and_lowering_passes == 0u
                                ? ConSanPipelineStageStatus::Blocked
                                : terminal_stage_status(lowerer_outcome);
    validation_stage.status = execution.final_validation_passes == 0u
                                  ? ConSanPipelineStageStatus::Blocked
                                  : terminal_stage_status(lowerer_outcome);
  }
  ConSanPipelineStageState &publication =
      stage_record(result, ConSanPipelineStage::ResultPublication);
  publication.status = ConSanPipelineStageStatus::Completed;
  publication.execution_count = 1;

  return result;
}

TransformResult TransformResult::execute_test_transaction(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
    ConSanTransformArtifacts lowering_artifacts) {
  const ConSanLoweringExecution synthetic_execution = {
      .program_inventory_passes = 1,
      .observation_plan_passes = 1,
      .resource_solving_and_lowering_passes = 1,
      .final_validation_passes = 1,
  };
  return execute_consan_transaction(code_object_bytes, request, transform_policy, runtime_policy,
                                    debug, mutation, capabilities, resources,
                                    std::move(lowering_artifacts), synthetic_execution);
}

} // namespace rocjitsu

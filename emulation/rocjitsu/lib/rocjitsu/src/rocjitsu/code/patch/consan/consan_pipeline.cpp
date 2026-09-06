// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_pipeline.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_transform_diagnostics.h"

#include "rocjitsu/code/patch/consan/consan_inventory_diagnostics.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <utility>
#include <variant>

namespace rocjitsu {

namespace {

[[nodiscard]] constexpr bool valid_contract_issue(ConSanContractIssue issue) {
  return static_cast<uint8_t>(issue) < static_cast<uint8_t>(ConSanContractIssue::Count);
}

[[nodiscard]] std::string fault_semantic_role(const ConSanProgramSite &source,
                                              const ConSanSyncSequence *sequence) {
  if (sequence == nullptr) {
    if (const ConSanBarrierSite *barrier = source.get_if<ConSanBarrierSite>())
      return barrier->mnemonic.find("signal") != std::string::npos ? "barrier-signal"
             : barrier->mnemonic.find("wait") != std::string::npos ? "barrier-wait"
                                                                   : "workgroup-barrier";
    if (const ConSanAtomicSite *atomic = source.get_if<ConSanAtomicSite>())
      return consan_atomic_semantic_role(*atomic);
    if (const ConSanOrdinaryMemorySite *memory = source.get_if<ConSanOrdinaryMemorySite>())
      return memory->operation == ConSanOrdinaryMemoryOperation::Load ? "ordinary-load"
                                                                      : "ordinary-store";
    return source.kind == ConSanLdsAccessKind::Read ? "lds-read" : "lds-write";
  }
  switch (sequence->operation) {
  case ConSanSyncOperation::BarrierSignal:
    return "barrier-signal";
  case ConSanSyncOperation::BarrierWait:
    return "barrier-wait";
  case ConSanSyncOperation::BarrierFull:
    return "workgroup-barrier";
  case ConSanSyncOperation::BarrierInit:
    return "barrier-init-unsupported";
  case ConSanSyncOperation::BarrierJoin:
    return "barrier-join-unsupported";
  case ConSanSyncOperation::BarrierLeave:
    return "barrier-leave-unsupported";
  case ConSanSyncOperation::BarrierWakeup:
    return "barrier-wakeup-unsupported";
  case ConSanSyncOperation::BarrierStateQuery:
    return "barrier-state-query-unsupported";
  case ConSanSyncOperation::AtomicRmw:
  case ConSanSyncOperation::AtomicCompareExchange:
    switch (sequence->memory_role) {
    case ConSanSyncMemoryRole::Acquire:
      return "atomic-acquire";
    case ConSanSyncMemoryRole::Release:
      return "atomic-release";
    case ConSanSyncMemoryRole::AcquireRelease:
      return "atomic-acquire-release";
    case ConSanSyncMemoryRole::SequentiallyConsistent:
      return "atomic-sequentially-consistent";
    case ConSanSyncMemoryRole::Unknown:
    case ConSanSyncMemoryRole::None:
      return "atomic-order-unknown";
    }
  case ConSanSyncOperation::OrdinaryLoad:
    return sequence->memory_role == ConSanSyncMemoryRole::Acquire ? "ordinary-acquire-load"
                                                                  : "ordinary-load";
  case ConSanSyncOperation::OrdinaryStore:
    return sequence->memory_role == ConSanSyncMemoryRole::Release ? "ordinary-release-store"
                                                                  : "ordinary-store";
  case ConSanSyncOperation::Unknown:
  case ConSanSyncOperation::Fence:
    return fault_semantic_role(source, nullptr);
  }
  return {};
}

[[nodiscard]] constexpr const char *patch_diagnostic_kind_name(ConSanPatchKind kind) {
  using E = ConSanPatchKind;
  constexpr auto vocabulary = make_consan_enum_vocabulary(
      "unknown", consan_enum(E::InlineNopRewrite, "inline-nop-rewrite"),
      consan_enum(E::InlineEndpgmRewrite, "inline-endpgm-rewrite"),
      consan_enum(E::InlineLdsEndpgmRewrite, "inline-lds-endpgm-rewrite"),
      consan_enum(E::LdsLoadCheckTrap, "inline-lds-load-check-trap"),
      consan_enum(E::LdsStoreCheckTrap, "inline-lds-store-check-trap"),
      consan_enum(E::FlatLoadCheckTrap, "inline-flat-load-check-trap"),
      consan_enum(E::FlatStoreCheckTrap, "inline-flat-store-check-trap"),
      consan_enum(E::InlineFlatTrapRewrite, "inline-flat-trap-rewrite"),
      consan_enum(E::InlineBarrierNopRewrite, "inline-barrier-nop-rewrite"),
      consan_enum(E::InlineBarrierIdScopeRewrite, "inline-barrier-id-scope-rewrite"),
      consan_enum(E::InlineBarrierParticipantCountRewrite,
                  "inline-barrier-participant-count-rewrite"),
      consan_enum(E::InlineBarrierMoveSourceRewrite, "inline-barrier-move-source-rewrite"),
      consan_enum(E::InlineBarrierMoveTargetRewrite, "inline-barrier-move-target-rewrite"),
      consan_enum(E::InlineAtomicAddressRewrite, "inline-atomic-address-rewrite"),
      consan_enum(E::InlineAtomicOrderRewrite, "inline-atomic-order-rewrite"),
      consan_enum(E::InlineAtomicScopeRewrite, "inline-atomic-scope-rewrite"),
      consan_enum(E::InlineLdsAddressRewrite, "inline-lds-address-rewrite"),
      consan_enum(E::InlineOrdinaryOrderRewrite, "inline-ordinary-order-rewrite"),
      consan_enum(E::InlineOrdinaryAddressRewrite, "inline-ordinary-address-rewrite"),
      consan_enum(E::InlineOrdinaryScopeRewrite, "inline-ordinary-scope-rewrite"),
      consan_enum(E::InlineMoiAccessRecordStore, "inline-moi-access-record-store"),
      consan_enum(E::TrampolineMoiAccessRecordStore, "trampoline-moi-access-record-store"),
      consan_enum(E::InlineMoiExactShadowStore, "inline-moi-exact-shadow-store"),
      consan_enum(E::TrampolineMoiExactShadowStore, "trampoline-moi-exact-shadow-store"),
      consan_enum(E::InlineMoiSampledWatchpointStore, "inline-moi-sampled-watchpoint-store"),
      consan_enum(E::TrampolineMoiSampledWatchpointStore,
                  "trampoline-moi-sampled-watchpoint-store"),
      consan_enum(E::KernelEntryMoiOwnerEpochPrologue, "kernel-entry-moi-owner-epoch-prologue"),
      consan_enum(E::KernelEntryMoiPrivateEpochPrologue, "kernel-entry-moi-private-epoch-prologue"),
      consan_enum(E::TrampolineMoiBarrierRecord, "trampoline-moi-barrier-record"),
      consan_enum(E::TrampolineMoiInlineEpochBarrier, "trampoline-moi-inline-epoch-barrier"),
      consan_enum(E::TrampolineMoiInlineAtomicOrdering, "trampoline-moi-inline-atomic-ordering"),
      consan_enum(E::TrampolineMoiAtomicRecord, "trampoline-moi-atomic-record"),
      consan_enum(E::TrampolineMoiSampledSyncMetadata, "trampoline-moi-sampled-sync-metadata"),
      consan_enum(E::TrampolineMoiFenceRecord, "trampoline-moi-fence-record"),
      consan_enum(E::InlineMalformedBarrierAbort, "inline-malformed-barrier-abort"),
      consan_enum(E::TrampolineScPerturbation, "trampoline-sc-perturbation"),
      consan_enum(E::InlineScalarClauseNopRewrite, "inline-scalar-clause-nop-rewrite"),
      consan_enum(E::TrampolineNop, "trampoline-nop"));
  return vocabulary.name(kind).data();
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

template <typename PatchRange>
[[nodiscard]] ConSanDispatchRequirements
build_dispatch_requirements(const ProgramInventory &inventory, const ConSanCoverageLedger &coverage,
                            const PatchRange &patches) {
  std::map<std::string, ConSanKernelDispatchRequirement> requirements_by_name;
  std::unordered_map<uint64_t, std::vector<const ConSanProgramSite *>> sites_by_text_offset;
  sites_by_text_offset.reserve(inventory.program_sites().size());
  for (const ConSanProgramSite &site : inventory.program_sites())
    sites_by_text_offset[site.physical_id.original_text_offset].push_back(&site);
  const auto note_kernel = [&](const ConSanProgramContainer &kernel, const auto &apply) {
    if (kernel.name.empty())
      return false;
    ConSanKernelDispatchRequirement &requirement = requirements_by_name[kernel.name];
    requirement.kernel_name = kernel.name;
    apply(requirement);
    return true;
  };
  const auto note_descriptor = [&](uint64_t descriptor_offset, const auto &apply) {
    const ConSanProgramContainer *kernel = inventory.find_kernel_by_descriptor(descriptor_offset);
    return kernel != nullptr && note_kernel(*kernel, apply);
  };
  const auto note_physical_site = [&](const PhysicalSiteId &physical, const auto &apply) {
    bool attributed = false;
    const auto candidates = sites_by_text_offset.find(physical.original_text_offset);
    if (candidates != sites_by_text_offset.end()) {
      for (const ConSanProgramSite *site : candidates->second) {
        if (site->physical_id != physical)
          continue;
        for (const ConSanExecutionOwner &owner : site->execution_owners) {
          if (const ConSanProgramContainer *kernel = inventory.kernel(owner))
            attributed |= note_kernel(*kernel, apply);
        }
      }
    }
    if (attributed)
      return;

    // Kernel-local legacy sites may predate explicit execution-owner analysis.
    // Keep this bounded fallback at the publication boundary; runtime code must
    // not repeat symbol-range ownership inference.
    const auto kernel =
        std::ranges::find_if(inventory.kernels(), [&](const ConSanProgramContainer &item) {
          return item.has_text_range && physical.original_text_offset >= item.entry_text_offset &&
                 physical.original_text_offset - item.entry_text_offset < item.code_size;
        });
    if (kernel != inventory.kernels().end())
      (void)note_kernel(*kernel, apply);
  };

  for (const ConSanIntentCoverageEntry &entry : coverage.intent_entries()) {
    if (entry.lowering != ConSanLoweringOutcomeKind::Instrumented)
      continue;
    const ConSanProbeIntent *intent = coverage.intent(entry.intent_id);
    if (intent == nullptr)
      continue;
    const auto mark_instrumented = [](ConSanKernelDispatchRequirement &requirement) {
      requirement.has_instrumented_probe = true;
    };
    if (intent->covered_semantic_sites.empty()) {
      note_physical_site(intent->physical_site, mark_instrumented);
      continue;
    }
    for (const SemanticSiteId &semantic : intent->covered_semantic_sites)
      note_physical_site(semantic.physical, mark_instrumented);
  }

  for (const ConSanPatchLoweringProduct &patch : patches) {
    const bool has_segment_requirement = patch.required_private_segment_size != 0u ||
                                         patch.dynamic_private_segment_addend != 0u ||
                                         patch.required_group_segment_size() != 0u;
    if (!has_segment_requirement)
      continue;
    const auto merge_segments = [&](ConSanKernelDispatchRequirement &requirement) {
      requirement.required_private_bytes =
          std::max(requirement.required_private_bytes, patch.required_private_segment_size);
      requirement.dynamic_private_addend =
          std::max(requirement.dynamic_private_addend, patch.dynamic_private_segment_addend);
      requirement.required_group_bytes =
          std::max(requirement.required_group_bytes, patch.required_group_segment_size());
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

std::optional<ConSanFaultSiteDiagnostic>
consan_fault_site_diagnostic(const ProgramInventory &inventory, const ConSanFaultSite &site) {
  const ConSanProgramSite *source = inventory.program_site(site);
  const ConSanProgramContainer *container =
      source == nullptr ? nullptr : inventory.container(source->container);
  if (source == nullptr || container == nullptr)
    return std::nullopt;
  ConSanFaultSiteDiagnostic diagnostic;
  static_cast<ConSanDecodedSite &>(diagnostic) = source->decoded_site();
  diagnostic.kind = site.kind;
  diagnostic.identity = site.identity;
  diagnostic.container_name = container->name;
  diagnostic.in_kernel = container->is_kernel();
  diagnostic.occurrence = site.occurrence;
  diagnostic.execution_owners = source->execution_owners;
  if (const ConSanBarrierSite *barrier = source->get_if<ConSanBarrierSite>()) {
    diagnostic.decoded_operands = consan_barrier_decoded_operands(*barrier, barrier->raw_encoding);
  } else if (const ConSanAtomicSite *atomic = source->get_if<ConSanAtomicSite>()) {
    diagnostic.width_bits = atomic->width_bits;
    diagnostic.decoded_operands = consan_atomic_decoded_operands(*atomic);
  } else if (const ConSanOrdinaryMemorySite *memory = source->get_if<ConSanOrdinaryMemorySite>()) {
    diagnostic.width_bits = memory->width_bits;
    diagnostic.decoded_operands = consan_ordinary_memory_decoded_operands(*memory);
    diagnostic.ordinary_memory_support_reason = memory->support_reason;
  } else {
    diagnostic.width_bits = source->decoded_width_bits;
    diagnostic.decoded_operands = consan_lds_decoded_operands(source->operands);
  }
  const SynchronizationInventoryView synchronization = inventory.sync();
  if (const ConSanSyncEvent *event = synchronization.find_event(site.source_site))
    diagnostic.sync_event_identity = event->identity;
  const ConSanSyncSequence *sequence =
      synchronization.find_unique_sequence_containing(site.source_site);
  if (sequence != nullptr) {
    diagnostic.sync_sequence_identity = sequence->identity;
    diagnostic.sync_confidence = sequence->confidence;
    diagnostic.sync_memory_role = sequence->memory_role;
  }
  diagnostic.semantic_role = fault_semantic_role(*source, sequence);
  return diagnostic;
}

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
    ConSanLoweringExtent extent = ConSanLoweringExtent::Complete,
    std::optional<ConSanLoweringObservation> supplied_observation = std::nullopt) {
  return ConSanTransformTransaction(code_object_bytes, request, transform_policy, runtime_policy,
                                    debug, mutation, capabilities, resources)
      .execute(std::move(supplied_lowering), extent, std::move(supplied_observation));
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
    if (auto diagnostic = consan_fault_site_diagnostic(result.program_inventory, site))
      report.fault_sites.push_back(std::move(*diagnostic));
  }

  report.barrier_move_destinations.reserve(
      result.private_lowering_.barrier_move_destinations.size());
  for (const ConSanBarrierMoveDestination &destination :
       result.private_lowering_.barrier_move_destinations)
    report.barrier_move_destinations.emplace_back(
        static_cast<const ConSanBarrierMoveDestinationPresentation &>(destination));

  report.fault_mutations.reserve(result.private_lowering_.fault_plans.size());
  for (const ConSanFaultMutationPlan &plan : result.private_lowering_.fault_plans)
    report.fault_mutations.emplace_back(static_cast<const ConSanFaultMutationPresentation &>(plan));

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
            .min_owners = plan.owner_kernel_ids.size(),
            .max_owners = plan.owner_kernel_ids.size(),
            .has_indirect_vgpr_access = plan.has_indirect_vgpr_access,
        });
      } else {
        ++failure->count;
        const auto include = [](auto &minimum, auto &maximum, auto value) {
          minimum = std::min(minimum, value);
          maximum = std::max(maximum, value);
        };
        include(failure->min_scratch_vgprs, failure->max_scratch_vgprs, plan.scratch_vgpr_count);
        include(failure->min_current_vgprs, failure->max_current_vgprs, plan.current_vgpr_count);
        include(failure->min_max_referenced_vgprs, failure->max_max_referenced_vgprs,
                plan.max_referenced_vgpr_count);
        include(failure->min_ordinary_vgpr_limit, failure->max_ordinary_vgpr_limit,
                plan.ordinary_vgpr_limit);
        include(failure->min_required_vgprs, failure->max_required_vgprs, plan.required_vgpr_count);
        include(failure->min_owners, failure->max_owners, plan.owner_kernel_ids.size());
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
        .sc_scalar_vcc_spill = patch.sc_scalar_vcc_spill,
        .persistent_epoch_private_offset =
            patch.private_state_layout
                ? std::optional<uint32_t>(patch.private_state_layout->epoch_offset)
                : std::nullopt,
        .spilled_vgpr_count = patch.spilled_vgpr_count,
        .required_private_segment_size = patch.required_private_segment_size,
        .dynamic_private_segment_addend = patch.dynamic_private_segment_addend,
        .workgroup_shadow_base = patch.workgroup_shadow ? patch.workgroup_shadow->base : 0u,
        .workgroup_shadow_size = patch.workgroup_shadow ? patch.workgroup_shadow->size : 0u,
        .required_group_segment_size = patch.required_group_segment_size(),
    });
  }
  return report;
}

void TransformResult::publish_lowering_artifacts(ConSanTransformArtifacts lowering) {
  program_inventory = std::move(lowering.program_inventory);
  coverage_ledger = std::move(lowering.coverage_ledger);
  mutation = std::move(lowering.mutation);
  replacement = std::move(lowering.replacement);
  outcome = lowering.outcome;
  transform_failure_cause = lowering.transform_failure_cause;
  warnings = std::move(lowering.warnings);
  errors = std::move(lowering.errors);
  private_lowering_.fault_sites = std::move(lowering.fault_sites);
  private_lowering_.barrier_move_destinations = std::move(lowering.barrier_move_destinations);
  private_lowering_.fault_plans = std::move(lowering.fault_plans);
  private_lowering_.resource_plans = std::move(lowering.resource_plans);
  private_lowering_.patches = std::move(lowering.patches);
}

bool TransformResult::well_formed() const {
  if (!code_object.valid() || !dispatch_requirements.well_formed() ||
      !valid_contract_issue(contract_issue)) {
    return false;
  }
  if (contract_issue != ConSanContractIssue::None &&
      outcome != ConSanTransformOutcome::Unsupported &&
      outcome != ConSanTransformOutcome::Invalid) {
    return false;
  }
  if (!program_inventory.empty() && (program_inventory.code_object_id() != code_object ||
                                     !program_inventory.program_site_ids_well_formed() ||
                                     !program_inventory.execution_owners_well_formed()))
    return false;
  if (std::ranges::any_of(private_lowering_.fault_sites, [&](const ConSanFaultSite &site) {
        return program_inventory.program_site(site) == nullptr;
      })) {
    return false;
  }
  if (mutation.fault.planned != private_lowering_.fault_plans.size() ||
      mutation.fault.applied > mutation.fault.planned) {
    return false;
  }
  if (observation_plan().valid() && !evidence_requirements)
    return false;
  if (evidence_requirements &&
      (!observation_plan().valid() ||
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
  coverage_ledger.discard_instrumented_lowerings();
  ConSanCoverageLedger rejected_coverage = coverage_ledger;
  bool rejections_valid = true;
  for (const ConSanProbeIntent &intent : observation_plan().probe_intents) {
    const ConSanIntentCoverageEntry *entry = coverage_ledger.intent_entry(intent.id);
    if (entry == nullptr || entry->lowering != ConSanLoweringOutcomeKind::Pending)
      continue;
    const std::array<ConSanProbeIntentId, 1> ids = {intent.id};
    if (!rejected_coverage.publish_lowering_rejection(ids,
                                                      ConSanLoweringOutcomeKind::ResourceRejected,
                                                      "runtime-owned report allocation failed")) {
      errors.emplace_back("ConSan runtime binding could not publish intent rejections");
      rejections_valid = false;
      break;
    }
  }
  if (rejections_valid)
    coverage_ledger = std::move(rejected_coverage);
  dispatch_requirements = {};
  warnings.push_back(std::move(warning));
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
      !inventory_result_.evidence_requirements ||
      !evidence_is_complete(*inventory_result_.evidence_requirements) ||
      !evidence_requires_binding(*inventory_result_.evidence_requirements) ||
      inventory_result_.contract_issue != ConSanContractIssue::None) {
    return false;
  }
  const auto engine = request_.flavor
                          ? consan_capability_engine(*request_.flavor, request_.moi_engine)
                          : std::nullopt;
  if (!engine || inventory_result_.observation_plan().engine != *engine)
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
                                       BoundRuntimeResources{}, std::nullopt,
                                       ConSanLoweringExtent::ThroughProgramInventory);
  const bool inventory_acceptable =
      executor != nullptr ? inventory.code_object.valid() : inventory.well_formed();
  if (!inventory_acceptable || !inventory.evidence_requirements ||
      !evidence_requires_binding(*inventory.evidence_requirements)) {
    if (inventory_acceptable && (executor == nullptr || inventory_mutation != mutation) &&
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
    rejected.discard_replacement(
        "ConSan automatic resume rejected runtime evidence binding: issue=" +
        std::string(consan_contract_issue_name(binding_issue)));
    rejected.contract_issue = binding_issue;
    return rejected;
  }

  TransformResult result = std::move(deferred.inventory_result_);
  const ConSanLoweringObservation observation = {
      .initial_coverage = result.coverage_ledger,
  };
  ConSanTransformArtifacts completed;

  switch (deferred.strategy_) {
  case ConSanDeferredBinding::ResumeStrategy::RetryMoiInventory: {
    ConSanOptions retry_options(deferred.request_, deferred.transform_policy_, deferred.debug_,
                                deferred.requested_mutation_, deferred.capabilities_, resources);
    completed = retry_patch_consan_moi_from_inventory(
        {.program_inventory = std::move(result.program_inventory),
         .coverage_ledger = std::move(result.coverage_ledger),
         .fault_sites = std::move(result.private_lowering_.fault_sites),
         .barrier_move_destinations =
             std::move(result.private_lowering_.barrier_move_destinations)},
        std::move(retry_options), code_object_bytes, &observation);
    break;
  }
  case ConSanDeferredBinding::ResumeStrategy::RelowerFromInput: {
    const ConSanOptions options(deferred.request_, deferred.transform_policy_, deferred.debug_,
                                deferred.requested_mutation_, deferred.capabilities_, resources);
    completed =
        lower_consan(code_object_bytes, options, ConSanLoweringExtent::Complete, &observation);
    break;
  }
  case ConSanDeferredBinding::ResumeStrategy::InvokeExecutor:
    return deferred.executor_(code_object_bytes, deferred.request_, deferred.transform_policy_,
                              deferred.runtime_policy_, deferred.debug_,
                              deferred.requested_mutation_, deferred.capabilities_, resources);
  }

  result.publish_lowering_artifacts(std::move(completed));
  if (result.outcome == ConSanTransformOutcome::ModifiedValid) {
    result.dispatch_requirements = build_dispatch_requirements(
        result.program_inventory, result.coverage_ledger, result.private_lowering_.patches);
  }
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
  if (configuration_issue != ConSanContractIssue::None) {
    result.contract_issue = configuration_issue;
    return result;
  }

  const ConSanContractIssue backend_issue = validate_runtime_capabilities(capabilities);
  if (backend_issue != ConSanContractIssue::None) {
    result.contract_issue = backend_issue;
    return result;
  }

  const ConSanFlavor flavor = request.flavor.value_or(ConSanFlavor::None);
  const bool staged_native_lowering = !supplied_artifacts && !supplied_observation &&
                                      extent == ConSanLoweringExtent::Complete &&
                                      flavor != ConSanFlavor::None && !mutation.has_mutation();
  const ConSanLoweringExtent initial_extent =
      staged_native_lowering ? ConSanLoweringExtent::ThroughProgramInventory : extent;
  ConSanTransformArtifacts lowering;
  if (supplied_artifacts) {
    lowering = std::move(*supplied_artifacts);
  } else {
    const ConSanOptions lowering_options(request, transform_policy, debug, mutation, capabilities,
                                         resources);
    lowering = lower_consan(code_object_bytes, lowering_options, initial_extent,
                            supplied_observation ? &*supplied_observation : nullptr);
  }

  if (initial_extent == ConSanLoweringExtent::ThroughProgramInventory &&
      flavor != ConSanFlavor::None && lowering.errors.empty() &&
      lowering.program_inventory.code_object_parsed() &&
      consan_target_profile(lowering.program_inventory.target()) != nullptr) {
    ConSanObservationProduct observation =
        assemble_consan_observation_product(lowering.program_inventory, request, debug);
    lowering.coverage_ledger = std::move(observation.initial_coverage);
    lowering.errors.insert(lowering.errors.end(),
                           std::make_move_iterator(observation.diagnostics.begin()),
                           std::make_move_iterator(observation.diagnostics.end()));
    if (!lowering.errors.empty())
      lowering.outcome = ConSanTransformOutcome::Invalid;
  }
  result.publish_lowering_artifacts(std::move(lowering));

  if (flavor != ConSanFlavor::None && result.observation_plan().valid()) {
    const std::optional<uint64_t> maximum_access_probe_count =
        transform_policy.max_patches_is_expert_limit
            ? std::optional<uint64_t>{transform_policy.max_patches}
            : std::nullopt;
    if (flavor == ConSanFlavor::SuperCollider) {
      result.evidence_requirements = plan_consan_supercollider_evidence(
          result.observation_plan(), request.supercollider_evidence_mode);
    } else if (flavor == ConSanFlavor::Moi) {
      result.evidence_requirements = consan_moi_impl::plan_moi_evidence_requirements(
          request.moi_engine, {.program_inventory = result.program_inventory,
                               .observation_plan = result.observation_plan(),
                               .requested_report_buffer_size = request.moi_auto_report_buffer_size,
                               .maximum_access_probe_count = maximum_access_probe_count,
                               .maximum_workgroup_lds_bytes = capabilities.max_workgroup_lds_bytes,
                               .dynamic_access_records = request.moi_dynamic_access_records});
    }
    if (!result.evidence_requirements ||
        !consan_evidence_requirements_well_formed(*result.evidence_requirements)) {
      result.errors.emplace_back("ConSan produced invalid runtime evidence requirements");
      result.outcome = ConSanTransformOutcome::Invalid;
      result.replacement.clear();
      result.private_lowering_.patches.clear();
      result.coverage_ledger.discard_instrumented_lowerings();
    }
  }

  bool binding_ready = flavor == ConSanFlavor::None;
  if (result.evidence_requirements) {
    if (!evidence_is_complete(*result.evidence_requirements)) {
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.replacement.clear();
      result.private_lowering_.patches.clear();
      result.coverage_ledger.discard_instrumented_lowerings();
    } else if (!evidence_requires_binding(*result.evidence_requirements)) {
      binding_ready = true;
    } else if (!resources.bound()) {
      return result;
    } else {
      const ConSanContractIssue requirement_issue = validate_runtime_capabilities(
          capabilities, runtime_requirements(*result.evidence_requirements));
      const ConSanContractIssue resource_issue =
          validate_evidence_binding(*result.evidence_requirements, resources);
      const ConSanContractIssue binding_issue =
          requirement_issue != ConSanContractIssue::None ? requirement_issue : resource_issue;
      if (binding_issue == ConSanContractIssue::None) {
        binding_ready = true;
      } else {
        result.contract_issue = binding_issue;
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
        result.coverage_ledger.discard_instrumented_lowerings();
        result.dispatch_requirements = {};
      }
    }
  }

  if (staged_native_lowering && binding_ready &&
      result.outcome != ConSanTransformOutcome::Invalid &&
      result.outcome != ConSanTransformOutcome::Unsupported) {
    const ConSanLoweringObservation observation = {
        .initial_coverage = result.coverage_ledger,
    };
    const ConSanOptions lowering_options(request, transform_policy, debug, mutation, capabilities,
                                         resources);
    ConSanTransformArtifacts completed = lower_consan(code_object_bytes, lowering_options,
                                                      ConSanLoweringExtent::Complete, &observation);
    result.publish_lowering_artifacts(std::move(completed));
  }

  if (result.outcome == ConSanTransformOutcome::ModifiedValid) {
    result.dispatch_requirements = build_dispatch_requirements(
        result.program_inventory, result.coverage_ledger, result.private_lowering_.patches);
  }
  return result;
}

TransformResult TransformResult::execute_test_transaction(
    std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const ConSanDebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
    ConSanTransformArtifacts lowering_artifacts) {
  return execute_consan_transaction(code_object_bytes, request, transform_policy, runtime_policy,
                                    debug, mutation, capabilities, resources,
                                    std::move(lowering_artifacts));
}

} // namespace rocjitsu

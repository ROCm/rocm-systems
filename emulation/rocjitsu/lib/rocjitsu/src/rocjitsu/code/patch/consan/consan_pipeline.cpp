// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_pipeline.h"
#include "rocjitsu/code/patch/consan/consan_instrumentation.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_transform_diagnostics.h"

#include "rocjitsu/code/patch/consan/consan_inventory_diagnostics.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <utility>
#include <variant>

namespace rocjitsu::consan {

namespace {

[[nodiscard]] constexpr bool valid_contract_issue(ContractIssue issue) {
  return static_cast<uint8_t>(issue) < static_cast<uint8_t>(ContractIssue::Count);
}

[[nodiscard]] std::string fault_semantic_role(const ProgramSite &source,
                                              const SyncSequence *sequence) {
  if (sequence == nullptr) {
    if (const BarrierSite *barrier = source.get_if<BarrierSite>())
      return barrier->mnemonic.find("signal") != std::string::npos ? "barrier-signal"
             : barrier->mnemonic.find("wait") != std::string::npos ? "barrier-wait"
                                                                   : "workgroup-barrier";
    if (const AtomicSite *atomic = source.get_if<AtomicSite>())
      return atomic_semantic_role(*atomic);
    if (const OrdinaryMemorySite *memory = source.get_if<OrdinaryMemorySite>())
      return memory->operation == OrdinaryMemoryOperation::Load ? "ordinary-load"
                                                                : "ordinary-store";
    return source.kind == LdsAccessKind::Read ? "lds-read" : "lds-write";
  }
  switch (sequence->operation) {
  case SyncOperation::BarrierSignal:
    return "barrier-signal";
  case SyncOperation::BarrierWait:
    return "barrier-wait";
  case SyncOperation::BarrierFull:
    return "workgroup-barrier";
  case SyncOperation::BarrierInit:
    return "barrier-init-unsupported";
  case SyncOperation::BarrierJoin:
    return "barrier-join-unsupported";
  case SyncOperation::BarrierLeave:
    return "barrier-leave-unsupported";
  case SyncOperation::BarrierWakeup:
    return "barrier-wakeup-unsupported";
  case SyncOperation::BarrierStateQuery:
    return "barrier-state-query-unsupported";
  case SyncOperation::AtomicRmw:
  case SyncOperation::AtomicCompareExchange:
    switch (sequence->memory_role) {
    case SyncMemoryRole::Acquire:
      return "atomic-acquire";
    case SyncMemoryRole::Release:
      return "atomic-release";
    case SyncMemoryRole::AcquireRelease:
      return "atomic-acquire-release";
    case SyncMemoryRole::SequentiallyConsistent:
      return "atomic-sequentially-consistent";
    case SyncMemoryRole::Unknown:
    case SyncMemoryRole::None:
      return "atomic-order-unknown";
    }
    return "atomic-order-unknown";
  case SyncOperation::OrdinaryLoad:
    return sequence->memory_role == SyncMemoryRole::Acquire ? "ordinary-acquire-load"
                                                            : "ordinary-load";
  case SyncOperation::OrdinaryStore:
    return sequence->memory_role == SyncMemoryRole::Release ? "ordinary-release-store"
                                                            : "ordinary-store";
  case SyncOperation::Unknown:
  case SyncOperation::Fence:
    return fault_semantic_role(source, nullptr);
  }
  return {};
}

[[nodiscard]] constexpr const char *patch_diagnostic_kind_name(PatchKind kind) {
  using E = PatchKind;
  constexpr auto vocabulary = make_enum_vocabulary(
      "unknown", enum_entry(E::LdsLoadCheckTrap, "inline-lds-load-check-trap"),
      enum_entry(E::LdsStoreCheckTrap, "inline-lds-store-check-trap"),
      enum_entry(E::FlatLoadCheckTrap, "inline-flat-load-check-trap"),
      enum_entry(E::FlatStoreCheckTrap, "inline-flat-store-check-trap"),
      enum_entry(E::InlineBarrierNopRewrite, "inline-barrier-nop-rewrite"),
      enum_entry(E::InlineBarrierIdScopeRewrite, "inline-barrier-id-scope-rewrite"),
      enum_entry(E::InlineBarrierParticipantCountRewrite,
                 "inline-barrier-participant-count-rewrite"),
      enum_entry(E::InlineBarrierMoveSourceRewrite, "inline-barrier-move-source-rewrite"),
      enum_entry(E::InlineBarrierMoveTargetRewrite, "inline-barrier-move-target-rewrite"),
      enum_entry(E::InlineAtomicAddressRewrite, "inline-atomic-address-rewrite"),
      enum_entry(E::InlineAtomicOrderRewrite, "inline-atomic-order-rewrite"),
      enum_entry(E::InlineAtomicScopeRewrite, "inline-atomic-scope-rewrite"),
      enum_entry(E::InlineLdsAddressRewrite, "inline-lds-address-rewrite"),
      enum_entry(E::InlineOrdinaryOrderRewrite, "inline-ordinary-order-rewrite"),
      enum_entry(E::InlineOrdinaryAddressRewrite, "inline-ordinary-address-rewrite"),
      enum_entry(E::InlineOrdinaryScopeRewrite, "inline-ordinary-scope-rewrite"),
      enum_entry(E::InlineWatchpointStore, "inline-watchpoint-store"),
      enum_entry(E::TrampolineWatchpointStore, "trampoline-watchpoint-store"),
      enum_entry(E::KernelEntryOwnerEpochPrologue, "kernel-entry-owner-epoch-prologue"),
      enum_entry(E::KernelEntryPrivateEpochPrologue, "kernel-entry-private-epoch-prologue"),
      enum_entry(E::TrampolineSyncMetadata, "trampoline-sync-metadata"),
      enum_entry(E::InlineMalformedBarrierAbort, "inline-malformed-barrier-abort"),
      enum_entry(E::TrampolineSuperColliderPerturbation, "trampoline-sc-perturbation"),
      enum_entry(E::InlineScalarClauseNopRewrite, "inline-scalar-clause-nop-rewrite"));
  return vocabulary.name(kind).data();
}

[[nodiscard]] RuntimeCapabilityRequirements
runtime_requirements(const EvidenceRequirements &requirements) {
  return std::visit([](const auto &typed) { return typed.runtime_requirements; }, requirements);
}

[[nodiscard]] bool evidence_requires_binding(const EvidenceRequirements &requirements) {
  return std::visit([](const auto &typed) { return typed.requires_binding(); }, requirements);
}

[[nodiscard]] bool evidence_is_complete(const EvidenceRequirements &requirements) {
  return std::visit([](const auto &typed) { return typed.complete(); }, requirements);
}

[[nodiscard]] ContractIssue validate_evidence_binding(const EvidenceRequirements &requirements,
                                                      const BoundRuntimeResources &resources) {
  return std::visit(
      [&](const auto &typed) {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, SuperColliderEvidenceRequirements>) {
          if (resources.scope != RuntimeResourceScope::CodeObject &&
              resources.scope != RuntimeResourceScope::Executable)
            return ContractIssue::InvalidResourceScope;
          return typed.requires_binding() && !resources.supercollider_report_buffer_address
                     ? ContractIssue::InvalidResourceAddress
                     : ContractIssue::None;
        } else {
          if (!resources.report_buffer_address)
            return ContractIssue::InvalidResourceAddress;
          // An automatic allocation carries the exact address-free layout
          // that sized it and remains live for the executable. A caller-bound
          // raw buffer intentionally has no such layout: direct-buffer
          // lowering validates its exact geometry, while the
          // pipeline only requires the common header and a code-object or
          // executable lifetime. Treating that raw buffer as the automatic
          // multi-bank layout rejects valid small explicit buffers after
          // lowering has already proved and emitted their instrumentation.
          if (resources.report_layout) {
            if (resources.scope != RuntimeResourceScope::Executable)
              return ContractIssue::InvalidResourceScope;
            return resources.report_buffer_size < typed.abi_plan.required_bytes
                       ? ContractIssue::InvalidResourceSize
                       : ContractIssue::None;
          }
          if (resources.scope != RuntimeResourceScope::CodeObject &&
              resources.scope != RuntimeResourceScope::Executable)
            return ContractIssue::InvalidResourceScope;
          return resources.report_buffer_size < sizeof(ReportHeader)
                     ? ContractIssue::InvalidResourceSize
                     : ContractIssue::None;
        }
      },
      requirements);
}

template <typename PatchRange>
[[nodiscard]] DispatchRequirements build_dispatch_requirements(const ProgramInventory &inventory,
                                                               const CoverageLedger &coverage,
                                                               const PatchRange &patches) {
  std::map<std::string, KernelDispatchRequirement> requirements_by_name;
  std::unordered_map<uint64_t, std::vector<const ProgramSite *>> sites_by_text_offset;
  sites_by_text_offset.reserve(inventory.program_sites().size());
  for (const ProgramSite &site : inventory.program_sites())
    sites_by_text_offset[site.physical_id.original_text_offset].push_back(&site);
  const auto note_kernel = [&](const ProgramContainer &kernel, const auto &apply) {
    if (kernel.name.empty())
      return false;
    KernelDispatchRequirement &requirement = requirements_by_name[kernel.name];
    requirement.kernel_name = kernel.name;
    apply(requirement);
    return true;
  };
  const auto note_descriptor = [&](uint64_t descriptor_offset, const auto &apply) {
    const ProgramContainer *kernel = inventory.find_kernel_by_descriptor(descriptor_offset);
    return kernel != nullptr && note_kernel(*kernel, apply);
  };
  const auto note_physical_site = [&](const PhysicalSiteId &physical, const auto &apply) {
    bool attributed = false;
    const auto candidates = sites_by_text_offset.find(physical.original_text_offset);
    if (candidates != sites_by_text_offset.end()) {
      for (const ProgramSite *site : candidates->second) {
        if (site->physical_id != physical)
          continue;
        for (const ExecutionOwner &owner : site->execution_owners) {
          if (const ProgramContainer *kernel = inventory.kernel(owner))
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
        std::ranges::find_if(inventory.kernels(), [&](const ProgramContainer &item) {
          return item.has_text_range && physical.original_text_offset >= item.entry_text_offset &&
                 physical.original_text_offset - item.entry_text_offset < item.code_size;
        });
    if (kernel != inventory.kernels().end())
      (void)note_kernel(*kernel, apply);
  };

  for (const IntentCoverageEntry &entry : coverage.intent_entries()) {
    if (entry.lowering != LoweringOutcomeKind::Instrumented)
      continue;
    const ProbeIntent *intent = coverage.intent(entry.intent_id);
    if (intent == nullptr)
      continue;
    const auto mark_instrumented = [](KernelDispatchRequirement &requirement) {
      requirement.has_instrumented_probe = true;
    };
    if (intent->covered_semantic_sites.empty()) {
      note_physical_site(intent->physical_site, mark_instrumented);
      continue;
    }
    for (const SemanticSiteId &semantic : intent->covered_semantic_sites)
      note_physical_site(semantic.physical, mark_instrumented);
  }

  for (const PatchLoweringProduct &patch : patches) {
    const bool has_segment_requirement =
        patch.required_private_segment_size != 0u || patch.dynamic_private_segment_addend != 0u;
    if (!has_segment_requirement)
      continue;
    const auto merge_segments = [&](KernelDispatchRequirement &requirement) {
      requirement.required_private_bytes =
          std::max(requirement.required_private_bytes, patch.required_private_segment_size);
      requirement.dynamic_private_addend =
          std::max(requirement.dynamic_private_addend, patch.dynamic_private_segment_addend);
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

  DispatchRequirements result;
  result.kernels.reserve(requirements_by_name.size());
  for (auto &entry : requirements_by_name)
    result.kernels.push_back(std::move(entry.second));
  return result;
}

} // namespace

std::optional<FaultSiteDiagnostic> fault_site_diagnostic(const ProgramInventory &inventory,
                                                         const FaultSite &site) {
  const ProgramSite *source = inventory.program_site(site);
  const ProgramContainer *container =
      source == nullptr ? nullptr : inventory.container(source->container);
  if (source == nullptr || container == nullptr)
    return std::nullopt;
  FaultSiteDiagnostic diagnostic;
  static_cast<DecodedSite &>(diagnostic) = source->decoded_site();
  diagnostic.kind = site.kind;
  diagnostic.identity = site.identity;
  diagnostic.container_name = container->name;
  diagnostic.in_kernel = container->is_kernel();
  diagnostic.occurrence = site.occurrence;
  diagnostic.execution_owners = source->execution_owners;
  if (const BarrierSite *barrier = source->get_if<BarrierSite>()) {
    diagnostic.decoded_operands = barrier_decoded_operands(*barrier, barrier->raw_encoding);
  } else if (const AtomicSite *atomic = source->get_if<AtomicSite>()) {
    diagnostic.width_bits = atomic->width_bits;
    diagnostic.decoded_operands = atomic_decoded_operands(*atomic);
  } else if (const OrdinaryMemorySite *memory = source->get_if<OrdinaryMemorySite>()) {
    diagnostic.width_bits = memory->width_bits;
    diagnostic.decoded_operands = ordinary_memory_decoded_operands(*memory);
    diagnostic.ordinary_memory_support_reason = memory->support_reason;
  } else {
    diagnostic.width_bits = source->decoded_width_bits;
    diagnostic.decoded_operands = lds_decoded_operands(source->operands);
  }
  const SynchronizationInventoryView synchronization = inventory.sync();
  if (const SyncEvent *event = synchronization.find_event(site.source_site))
    diagnostic.sync_event_identity = event->identity;
  const SyncSequence *sequence = synchronization.find_unique_sequence_containing(site.source_site);
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
class TransformTransaction {
public:
  TransformTransaction(std::span<const uint8_t> code_object_bytes, const Request &request,
                       const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
                       const DebugOverrides &debug, const MutationRequest &mutation,
                       const RuntimeCapabilities &capabilities,
                       const BoundRuntimeResources &resources)
      : code_object_bytes_(code_object_bytes), request_(request),
        transform_policy_(transform_policy), runtime_policy_(runtime_policy), debug_(debug),
        mutation_(mutation), capabilities_(capabilities), resources_(resources) {}

  [[nodiscard]] TransformResult
  execute(std::optional<TransformArtifacts> supplied_lowering = std::nullopt,
          LoweringExtent extent = LoweringExtent::Complete,
          std::optional<LoweringObservation> supplied_observation = std::nullopt);

private:
  std::span<const uint8_t> code_object_bytes_;
  const Request &request_;
  const TransformPolicy &transform_policy_;
  const RuntimePolicy &runtime_policy_;
  const DebugOverrides &debug_;
  const MutationRequest &mutation_;
  const RuntimeCapabilities &capabilities_;
  const BoundRuntimeResources &resources_;
};

[[nodiscard]] TransformResult
execute_transaction(std::span<const uint8_t> code_object_bytes, const Request &request,
                    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
                    const DebugOverrides &debug, const MutationRequest &mutation,
                    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
                    std::optional<TransformArtifacts> supplied_lowering = std::nullopt,
                    LoweringExtent extent = LoweringExtent::Complete,
                    std::optional<LoweringObservation> supplied_observation = std::nullopt) {
  return TransformTransaction(code_object_bytes, request, transform_policy, runtime_policy, debug,
                              mutation, capabilities, resources)
      .execute(std::move(supplied_lowering), extent, std::move(supplied_observation));
}

TransformDiagnosticReport transform_diagnostic_report(const TransformResult &result) {
  ResourcePlanSummary resource_summary =
      summarize_resource_plans(result.private_lowering_.resource_plans);
  for (const PatchAbiEffects &patch : result.private_lowering_.patches)
    accumulate_emitted_spill(resource_summary, patch);
  TransformDiagnosticReport report;
  report.resource_summary = resource_summary;

  report.fault_sites.reserve(result.private_lowering_.fault_sites.size());
  for (const FaultSite &site : result.private_lowering_.fault_sites) {
    if (auto diagnostic = fault_site_diagnostic(result.program_inventory, site))
      report.fault_sites.push_back(std::move(*diagnostic));
  }

  report.barrier_move_destinations.reserve(
      result.private_lowering_.barrier_move_destinations.size());
  for (const BarrierMoveDestination &destination :
       result.private_lowering_.barrier_move_destinations)
    report.barrier_move_destinations.emplace_back(
        static_cast<const BarrierMoveDestinationPresentation &>(destination));

  report.fault_mutations.reserve(result.private_lowering_.fault_plans.size());
  for (const FaultMutationPlan &plan : result.private_lowering_.fault_plans)
    report.fault_mutations.emplace_back(static_cast<const FaultMutationPresentation &>(plan));

  for (const CandidateResourcePlan &plan : result.private_lowering_.resource_plans) {
    if (plan.source == RegisterAllocationSource::Unsupported) {
      const auto matching_failure = [&](const ResourceFailureDiagnostic &failure) {
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
      const ResourcePlanAlternative &alternative = plan.alternatives[alternative_index];
      report.resource_alternatives.push_back({
          .site_kind = plan.site_kind,
          .candidate_index = plan.candidate_index,
          .text_offset = plan.text_offset,
          .attempt_index = alternative_index,
          .kind = alternative.kind,
          .scratch_vgpr_count = alternative.scratch_vgpr_count,
          .source = alternative.source,
          .reason = alternative.reason,
          .outcome = resource_plan_alternative_outcome(plan, alternative),
      });
    }
  }
  std::ranges::sort(report.resource_failures, [](const ResourceFailureDiagnostic &lhs,
                                                 const ResourceFailureDiagnostic &rhs) {
    return std::pair(lhs.site_kind, lhs.reason) < std::pair(rhs.site_kind, rhs.reason);
  });

  report.patches.reserve(result.private_lowering_.patches.size());
  for (const PatchLoweringProduct &patch : result.private_lowering_.patches) {
    report.patches.push_back({
        .kind = patch_diagnostic_kind_name(patch.kind),
        .anchor_offset = patch.anchor_offset,
        .trampoline_offset = patch.trampoline_offset,
        .original_size = patch.original_size,
        .trampoline_size = patch.trampoline_size,
        .scratch_vgpr = patch.scratch_vgpr,
        .scalar_vcc_spill = patch.scalar_vcc_spill,
        .persistent_epoch_private_offset =
            patch.private_state_layout
                ? std::optional<uint32_t>(patch.private_state_layout->epoch_offset)
                : std::nullopt,
        .spilled_vgpr_count = patch.spilled_vgpr_count,
        .required_private_segment_size = patch.required_private_segment_size,
        .dynamic_private_segment_addend = patch.dynamic_private_segment_addend,
    });
  }
  return report;
}

void TransformResult::publish_lowering_artifacts(TransformArtifacts lowering) {
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
  if (contract_issue != ContractIssue::None && outcome != TransformOutcome::Unsupported &&
      outcome != TransformOutcome::Invalid) {
    return false;
  }
  if (!program_inventory.empty() && (program_inventory.code_object_id() != code_object ||
                                     !program_inventory.program_site_ids_well_formed() ||
                                     !program_inventory.execution_owners_well_formed()))
    return false;
  if (std::ranges::any_of(private_lowering_.fault_sites, [&](const FaultSite &site) {
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
      (!observation_plan().valid() || !evidence_requirements_well_formed(*evidence_requirements))) {
    return false;
  }
  if (std::ranges::any_of(errors, &std::string::empty)) {
    return false;
  }
  switch (outcome) {
  case TransformOutcome::ModifiedValid:
    if (replacement.empty())
      return false;
    break;
  case TransformOutcome::Unchanged:
  case TransformOutcome::Unsupported:
  case TransformOutcome::Invalid:
    if (!replacement.empty() || !dispatch_requirements.kernels.empty())
      return false;
    break;
  default:
    return false;
  }
  return true;
}

InstallAction TransformResult::install_action(bool fail_closed) const {
  switch (outcome) {
  case TransformOutcome::ModifiedValid:
    if (!replacement.empty())
      return InstallAction::LoadReplacement;
    return InstallAction::Reject;
  case TransformOutcome::Unchanged:
    return InstallAction::LoadOriginal;
  case TransformOutcome::Unsupported:
  case TransformOutcome::Invalid:
    return fail_closed ? InstallAction::Reject : InstallAction::LoadOriginal;
  }
  return InstallAction::Reject;
}

void TransformResult::discard_replacement(std::string warning) {
  outcome = TransformOutcome::Unsupported;
  replacement.clear();
  private_lowering_.patches.clear();
  coverage_ledger.discard_instrumented_lowerings();
  CoverageLedger rejected_coverage = coverage_ledger;
  bool rejections_valid = true;
  for (const ProbeIntent &intent : observation_plan().probe_intents) {
    const IntentCoverageEntry *entry = coverage_ledger.intent_entry(intent.id);
    if (entry == nullptr || entry->lowering != LoweringOutcomeKind::Pending)
      continue;
    const std::array<ProbeIntentId, 1> ids = {intent.id};
    if (!rejected_coverage.publish_lowering_rejection(ids, LoweringOutcomeKind::ResourceRejected,
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

bool DeferredBinding::well_formed() const {
  const bool injected_executor = strategy_ == ResumeStrategy::InvokeExecutor;
  if ((!injected_executor && !inventory_result_.well_formed()) ||
      (injected_executor && !inventory_result_.code_object.valid()) ||
      validate_configuration(request_, transform_policy_, runtime_policy_, debug_,
                             inventory_mutation_, BoundRuntimeResources{}) != ContractIssue::None ||
      validate_runtime_capabilities(capabilities_) != ContractIssue::None) {
    return false;
  }
  const MutationRequest expected_inventory_mutation =
      requested_mutation_.has_fault_mutation() && !requested_mutation_.fault_dry_run
          ? without_fault_mutations(requested_mutation_)
          : requested_mutation_;
  if (inventory_mutation_ != expected_inventory_mutation ||
      (!injected_executor &&
       inventory_result_.code_object != inventory_result_.program_inventory.code_object_id()) ||
      !inventory_result_.evidence_requirements ||
      !evidence_is_complete(*inventory_result_.evidence_requirements) ||
      !evidence_requires_binding(*inventory_result_.evidence_requirements) ||
      inventory_result_.contract_issue != ContractIssue::None) {
    return false;
  }
  const auto mode = request_.mode ? enabled_mode(*request_.mode) : std::nullopt;
  if (!mode || inventory_result_.observation_plan().mode != *mode)
    return false;
  switch (strategy_) {
  case ResumeStrategy::RelowerFromInput:
    return request_.mode == Mode::SuperCollider && executor_ == nullptr;
  case ResumeStrategy::RetryInventory:
    return request_.mode == Mode::Default && executor_ == nullptr;
  case ResumeStrategy::InvokeExecutor:
    return (request_.mode == Mode::Default || request_.mode == Mode::SuperCollider) &&
           executor_ != nullptr;
  }
  return false;
}

AutomaticTransformPreparation
prepare_automatic_transform(std::span<const uint8_t> code_object_bytes, const Request &request,
                            const TransformPolicy &transform_policy,
                            const RuntimePolicy &runtime_policy, const DebugOverrides &debug,
                            const MutationRequest &mutation,
                            const RuntimeCapabilities &capabilities, TransformExecutor executor) {
  const MutationRequest inventory_mutation =
      mutation.has_fault_mutation() && !mutation.fault_dry_run ? without_fault_mutations(mutation)
                                                               : mutation;
  const Mode mode = request.mode.value_or(Mode::None);
  const DeferredBinding::ResumeStrategy strategy =
      executor != nullptr     ? DeferredBinding::ResumeStrategy::InvokeExecutor
      : mode == Mode::Default ? DeferredBinding::ResumeStrategy::RetryInventory
                              : DeferredBinding::ResumeStrategy::RelowerFromInput;
  TransformResult inventory =
      executor != nullptr
          ? executor(code_object_bytes, request, transform_policy, runtime_policy, debug,
                     inventory_mutation, capabilities, BoundRuntimeResources{})
          : execute_transaction(code_object_bytes, request, transform_policy, runtime_policy, debug,
                                inventory_mutation, capabilities, BoundRuntimeResources{},
                                std::nullopt, LoweringExtent::ThroughProgramInventory);
  const bool inventory_acceptable =
      executor != nullptr ? inventory.code_object.valid() : inventory.well_formed();
  if (!inventory_acceptable || !inventory.evidence_requirements ||
      !evidence_requires_binding(*inventory.evidence_requirements)) {
    if (inventory_acceptable && (executor == nullptr || inventory_mutation != mutation) &&
        inventory.outcome != TransformOutcome::Invalid &&
        inventory.outcome != TransformOutcome::Unsupported) {
      return executor != nullptr
                 ? executor(code_object_bytes, request, transform_policy, runtime_policy, debug,
                            mutation, capabilities, BoundRuntimeResources{})
             : mutation.has_mutation()
                 ? transform_with_mutation(code_object_bytes, request, transform_policy,
                                           runtime_policy, debug, mutation, capabilities,
                                           BoundRuntimeResources{})
                 : transform(code_object_bytes, request, transform_policy, runtime_policy, debug,
                             capabilities, BoundRuntimeResources{});
    }
    return inventory;
  }
  return DeferredBinding(request, transform_policy, runtime_policy, debug, mutation,
                         inventory_mutation, capabilities, strategy, executor,
                         std::move(inventory));
}

TransformResult resume_automatic_transform(std::span<const uint8_t> code_object_bytes,
                                           const BoundRuntimeResources &resources,
                                           DeferredBinding deferred) {
  const auto invalid_resume = [&](std::string error) {
    TransformArtifacts invalid;
    invalid.outcome = TransformOutcome::Invalid;
    invalid.errors.push_back(std::move(error));
    return execute_transaction(code_object_bytes, deferred.request_, deferred.transform_policy_,
                               deferred.runtime_policy_, deferred.debug_,
                               deferred.requested_mutation_, deferred.capabilities_, resources,
                               std::move(invalid));
  };
  if (!deferred.well_formed())
    return invalid_resume("ConSan automatic resume received an invalid deferred-binding value");
  if (deferred.code_object() != make_code_object_id(code_object_bytes))
    return invalid_resume("ConSan automatic resume does not match the prepared input image");

  const EvidenceRequirements &evidence = *deferred.inventory_result_.evidence_requirements;
  const ContractIssue capability_issue =
      validate_runtime_capabilities(deferred.capabilities_, runtime_requirements(evidence));
  const ContractIssue resource_contract_issue = validate_bound_runtime_resources(resources);
  const ContractIssue evidence_binding_issue = validate_evidence_binding(evidence, resources);
  const ContractIssue binding_issue = capability_issue != ContractIssue::None ? capability_issue
                                      : resource_contract_issue != ContractIssue::None
                                          ? resource_contract_issue
                                          : evidence_binding_issue;
  if (binding_issue != ContractIssue::None) {
    TransformResult rejected = std::move(deferred.inventory_result_);
    rejected.discard_replacement(
        "ConSan automatic resume rejected runtime evidence binding: issue=" +
        std::string(contract_issue_name(binding_issue)));
    rejected.contract_issue = binding_issue;
    return rejected;
  }

  TransformResult result = std::move(deferred.inventory_result_);
  const LoweringObservation observation = {
      .initial_coverage = result.coverage_ledger,
  };
  TransformArtifacts completed;

  switch (deferred.strategy_) {
  case DeferredBinding::ResumeStrategy::RetryInventory: {
    Options retry_options(deferred.request_, deferred.transform_policy_, deferred.debug_,
                          deferred.requested_mutation_, deferred.capabilities_, resources);
    completed =
        retry_patch_from_inventory({.program_inventory = std::move(result.program_inventory),
                                    .coverage_ledger = std::move(result.coverage_ledger),
                                    .fault_sites = std::move(result.private_lowering_.fault_sites),
                                    .barrier_move_destinations = std::move(
                                        result.private_lowering_.barrier_move_destinations)},
                                   std::move(retry_options), code_object_bytes, &observation);
    break;
  }
  case DeferredBinding::ResumeStrategy::RelowerFromInput: {
    const Options options(deferred.request_, deferred.transform_policy_, deferred.debug_,
                          deferred.requested_mutation_, deferred.capabilities_, resources);
    completed = lower(code_object_bytes, options, LoweringExtent::Complete, &observation);
    break;
  }
  case DeferredBinding::ResumeStrategy::InvokeExecutor:
    return deferred.executor_(code_object_bytes, deferred.request_, deferred.transform_policy_,
                              deferred.runtime_policy_, deferred.debug_,
                              deferred.requested_mutation_, deferred.capabilities_, resources);
  }

  result.publish_lowering_artifacts(std::move(completed));
  if (result.outcome == TransformOutcome::ModifiedValid) {
    result.dispatch_requirements = build_dispatch_requirements(
        result.program_inventory, result.coverage_ledger, result.private_lowering_.patches);
  }
  return result;
}

TransformResult cancel_automatic_transform(DeferredBinding deferred, std::string warning) {
  TransformResult result = std::move(deferred.inventory_result_);
  result.discard_replacement(std::move(warning));
  return result;
}

TransformResult transform(std::span<const uint8_t> code_object_bytes, const Request &request,
                          const TransformPolicy &transform_policy,
                          const RuntimePolicy &runtime_policy, const DebugOverrides &debug,
                          const RuntimeCapabilities &capabilities,
                          const BoundRuntimeResources &resources) {
  return transform_with_mutation(code_object_bytes, request, transform_policy, runtime_policy,
                                 debug, MutationRequest{}, capabilities, resources);
}

TransformResult
transform_with_mutation(std::span<const uint8_t> code_object_bytes, const Request &request,
                        const TransformPolicy &transform_policy,
                        const RuntimePolicy &runtime_policy, const DebugOverrides &debug,
                        const MutationRequest &mutation, const RuntimeCapabilities &capabilities,
                        const BoundRuntimeResources &resources) {
  return execute_transaction(code_object_bytes, request, transform_policy, runtime_policy, debug,
                             mutation, capabilities, resources);
}

TransformResult
TransformTransaction::execute(std::optional<TransformArtifacts> supplied_artifacts,
                              LoweringExtent extent,
                              std::optional<LoweringObservation> supplied_observation) {
  const std::span<const uint8_t> code_object_bytes = code_object_bytes_;
  const Request &request = request_;
  const TransformPolicy &transform_policy = transform_policy_;
  const RuntimePolicy &runtime_policy = runtime_policy_;
  const DebugOverrides &debug = debug_;
  const MutationRequest &mutation = mutation_;
  const RuntimeCapabilities &capabilities = capabilities_;
  const BoundRuntimeResources &resources = resources_;
  TransformResult result;
  result.code_object = make_code_object_id(code_object_bytes);

  const ContractIssue configuration_issue =
      validate_configuration(request, transform_policy, runtime_policy, debug, mutation, resources);
  if (configuration_issue != ContractIssue::None) {
    result.contract_issue = configuration_issue;
    return result;
  }

  const ContractIssue backend_issue = validate_runtime_capabilities(capabilities);
  if (backend_issue != ContractIssue::None) {
    result.contract_issue = backend_issue;
    return result;
  }

  const Mode mode = request.mode.value_or(Mode::None);
  const bool staged_native_lowering = !supplied_artifacts && !supplied_observation &&
                                      extent == LoweringExtent::Complete && mode != Mode::None &&
                                      !mutation.has_mutation();
  const LoweringExtent initial_extent =
      staged_native_lowering ? LoweringExtent::ThroughProgramInventory : extent;
  TransformArtifacts lowering;
  if (supplied_artifacts) {
    lowering = std::move(*supplied_artifacts);
  } else {
    const Options lowering_options(request, transform_policy, debug, mutation, capabilities,
                                   resources);
    lowering = lower(code_object_bytes, lowering_options, initial_extent,
                     supplied_observation ? &*supplied_observation : nullptr);
  }

  if (initial_extent == LoweringExtent::ThroughProgramInventory && mode != Mode::None &&
      lowering.errors.empty() && lowering.program_inventory.code_object_parsed() &&
      target_profile(lowering.program_inventory.target()) != nullptr) {
    ObservationProduct observation =
        assemble_observation_product(lowering.program_inventory, request, debug);
    lowering.coverage_ledger = std::move(observation.initial_coverage);
    lowering.errors.insert(lowering.errors.end(),
                           std::make_move_iterator(observation.diagnostics.begin()),
                           std::make_move_iterator(observation.diagnostics.end()));
    if (!lowering.errors.empty())
      lowering.outcome = TransformOutcome::Invalid;
  }
  result.publish_lowering_artifacts(std::move(lowering));

  if (mode != Mode::None && result.observation_plan().valid()) {
    const std::optional<uint64_t> maximum_access_probe_count =
        transform_policy.max_patches_is_expert_limit
            ? std::optional<uint64_t>{transform_policy.max_patches}
            : std::nullopt;
    if (mode == Mode::SuperCollider) {
      result.evidence_requirements = plan_supercollider_evidence(
          result.observation_plan(), request.supercollider_evidence_mode);
    } else if (mode == Mode::Default) {
      result.evidence_requirements = detail::plan_evidence_requirements(
          {.program_inventory = result.program_inventory,
           .observation_plan = result.observation_plan(),
           .requested_report_buffer_size = request.auto_report_buffer_size,
           .maximum_access_probe_count = maximum_access_probe_count,
           .maximum_workgroup_lds_bytes = capabilities.max_workgroup_lds_bytes,
           .watchpoint_banks = request.watchpoint_banks});
    }
    if (!result.evidence_requirements ||
        !evidence_requirements_well_formed(*result.evidence_requirements)) {
      result.errors.emplace_back("ConSan produced invalid runtime evidence requirements");
      result.outcome = TransformOutcome::Invalid;
      result.replacement.clear();
      result.private_lowering_.patches.clear();
      result.coverage_ledger.discard_instrumented_lowerings();
    }
  }

  bool binding_ready = mode == Mode::None;
  if (result.evidence_requirements) {
    if (!evidence_is_complete(*result.evidence_requirements)) {
      result.outcome = TransformOutcome::Unsupported;
      result.replacement.clear();
      result.private_lowering_.patches.clear();
      result.coverage_ledger.discard_instrumented_lowerings();
    } else if (!evidence_requires_binding(*result.evidence_requirements)) {
      binding_ready = true;
    } else if (!resources.bound()) {
      return result;
    } else {
      const ContractIssue requirement_issue = validate_runtime_capabilities(
          capabilities, runtime_requirements(*result.evidence_requirements));
      const ContractIssue resource_issue =
          validate_evidence_binding(*result.evidence_requirements, resources);
      const ContractIssue binding_issue =
          requirement_issue != ContractIssue::None ? requirement_issue : resource_issue;
      if (binding_issue == ContractIssue::None) {
        binding_ready = true;
      } else {
        result.contract_issue = binding_issue;
        const uint64_t binding_required_bytes = std::visit(
            [&](const auto &typed) {
              using T = std::decay_t<decltype(typed)>;
              if constexpr (std::is_same_v<T, SuperColliderEvidenceRequirements>) {
                return typed.runtime_requirements.minimum_report_allocation_bytes.value_or(0u);
              } else {
                return resources.report_layout
                           ? typed.runtime_requirements.minimum_report_allocation_bytes.value_or(0u)
                           : static_cast<uint64_t>(sizeof(ReportHeader));
              }
            },
            *result.evidence_requirements);
        result.warnings.emplace_back(
            "ConSan runtime evidence binding is unsupported: issue=" +
            std::string(contract_issue_name(binding_issue)) +
            ", scope=" + std::string(runtime_resource_scope_name(resources.scope)) +
            ", report-bytes=" + std::to_string(resources.report_buffer_size) +
            ", required-bytes=" + std::to_string(binding_required_bytes));
        result.outcome = TransformOutcome::Unsupported;
        result.replacement.clear();
        result.private_lowering_.patches.clear();
        result.coverage_ledger.discard_instrumented_lowerings();
        result.dispatch_requirements = {};
      }
    }
  }

  if (staged_native_lowering && binding_ready && result.outcome != TransformOutcome::Invalid &&
      result.outcome != TransformOutcome::Unsupported) {
    const LoweringObservation observation = {
        .initial_coverage = result.coverage_ledger,
    };
    const Options lowering_options(request, transform_policy, debug, mutation, capabilities,
                                   resources);
    TransformArtifacts completed =
        lower(code_object_bytes, lowering_options, LoweringExtent::Complete, &observation);
    result.publish_lowering_artifacts(std::move(completed));
  }

  if (result.outcome == TransformOutcome::ModifiedValid) {
    result.dispatch_requirements = build_dispatch_requirements(
        result.program_inventory, result.coverage_ledger, result.private_lowering_.patches);
  }
  return result;
}

TransformResult TransformResult::execute_test_transaction(
    std::span<const uint8_t> code_object_bytes, const Request &request,
    const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
    const DebugOverrides &debug, const MutationRequest &mutation,
    const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
    TransformArtifacts lowering_artifacts) {
  return execute_transaction(code_object_bytes, request, transform_policy, runtime_policy, debug,
                             mutation, capabilities, resources, std::move(lowering_artifacts));
}

} // namespace rocjitsu::consan

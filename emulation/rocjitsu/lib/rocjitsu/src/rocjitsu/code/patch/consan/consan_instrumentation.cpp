// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_instrumentation.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/consan/consan_candidate_projection.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_prologue.h"
#include "rocjitsu/code/patch/consan/consan_resource_planning.h"
#include "rocjitsu/code/patch/consan/consan_text_relocation.h"
#include "rocjitsu/code/patch/consan/targets/consan_vgpr_bank_state.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::consan {

TransformArtifacts try_patch(TransformArtifacts result, const Options &options,
                             const OperatingPoint &initial_operating_point,
                             std::span<const uint8_t> code_object_bytes, rj_code_arch_t arch) {

  const major_image_ownership::ScopedOwner result_owner(
      major_image_ownership::OwnerKind::ResultImage, result.replacement);
  Options effective_options = options;
  OperatingPoint effective_point = initial_operating_point;
  if (!result.operating_point.owner_transient_sgprs.empty()) {
    effective_point.owner_transient_sgprs = result.operating_point.owner_transient_sgprs;
  }
  if (!result.operating_point.owner_persistent_vgprs.empty()) {
    effective_point.owner_persistent_vgprs = result.operating_point.owner_persistent_vgprs;
  }
  result.outcome = result.errors.empty() ? TransformOutcome::Unchanged : TransformOutcome::Invalid;
  result.replacement.clear();
  result.resource_plans.clear();
  result.patches.clear();
  if (!result.errors.empty())
    return result;

  /// Close every ConSan intent that did not participate in a committed byte
  /// mutation. Unsupported resource plans reject the exact intent IDs they
  /// governed; all other still-pending intents receive an explicit placement
  /// rejection. This terminal transaction never searches patches, anchors,
  /// or graph geometry to rediscover what lowering attempted.
  const auto publish_pending_lowering_rejections =
      [&](std::optional<RegisterPlanReason> whole_transform_resource_failure = std::nullopt) {
        for (const CandidateResourcePlan &plan : result.resource_plans) {
          if (plan.source != RegisterAllocationSource::Unsupported || plan.intent_ids.empty())
            continue;
          std::vector<ProbeIntentId> pending;
          for (ProbeIntentId id : plan.intent_ids) {
            const IntentCoverageEntry *entry = result.coverage_ledger.intent_entry(id);
            if (entry != nullptr && entry->lowering == LoweringOutcomeKind::Pending &&
                std::ranges::find(pending, id) == pending.end()) {
              pending.push_back(id);
            }
          }
          if (pending.empty())
            continue;
          if (!result.coverage_ledger.publish_lowering_rejection(
                  pending, LoweringOutcomeKind::ResourceRejected,
                  "ConSan resource plan rejected intents: " +
                      std::string(register_plan_reason_name(plan.reason)),
                  plan.reason)) {
            result.errors.emplace_back("ConSan could not publish a resource rejection");
            return;
          }
        }

        for (const ProbeIntent &intent : result.observation_plan().probe_intents) {
          const IntentCoverageEntry *entry = result.coverage_ledger.intent_entry(intent.id);
          if (entry == nullptr || entry->lowering != LoweringOutcomeKind::Pending)
            continue;
          const std::array<ProbeIntentId, 1> ids = {intent.id};
          const LoweringOutcomeKind outcome = whole_transform_resource_failure
                                                  ? LoweringOutcomeKind::ResourceRejected
                                                  : LoweringOutcomeKind::PlacementRejected;
          const std::string_view detail =
              whole_transform_resource_failure
                  ? "ConSan whole-transform resource validation rejected the admitted intent"
                  : "ConSan lowerer selected no placement for the admitted intent";
          if (!result.coverage_ledger.publish_lowering_rejection(
                  ids, outcome, std::string(detail), whole_transform_resource_failure)) {
            result.errors.emplace_back("ConSan could not publish a placement rejection");
            return;
          }
        }
      };
  std::vector<Candidate> candidates =
      detail::build_candidates(result.program_inventory, result.observation_plan(), result.errors);
  if (!result.errors.empty())
    return result;
  if (arch_has_selectable_vgpr_bank(arch)) {
    for (Candidate &candidate : candidates) {
      const ProgramContainer *container =
          result.program_inventory.container(candidate.site().container);
      if (container == nullptr || candidate.anchor() < container->entry_text_offset ||
          candidate.site().decoded_file_offset() < candidate.anchor())
        continue;
      const uint64_t text_file_offset = candidate.site().decoded_file_offset() - candidate.anchor();
      candidate.incoming_vgpr_bank_mode = selectable_vgpr_bank_mode_at(
          arch, code_object_bytes, text_file_offset, container->entry_text_offset,
          candidate.site().decoded_file_offset());
    }
  }
  detail::ObjectFacts object_facts;
  object_facts.has_access_candidate = !candidates.empty();
  object_facts.admitted_atomic_count = std::ranges::count_if(
      result.observation_plan().atomic_site_decisions, [](const AtomicSiteDecision &decision) {
        return decision.kind == SiteDecisionKind::Admitted;
      });
  detail::ObjectModePlan mode_plan =
      plan_object_mode(effective_options, effective_options, effective_options, object_facts);
  effective_options.owner_source = mode_plan.owner_source;
  effective_options.track_atomics = mode_plan.track_atomics;
  if (!mode_plan.warning.empty())
    result.warnings.emplace_back(mode_plan.warning);
  if (!result.errors.empty())
    return result;
  // Register selection iterates as automatic persistent and transient state is
  // chosen. The code bytes, decoded CFG, ownership scopes, and liveness facts
  // do not change during those iterations; retain one analysis state instead
  // of rebuilding the full instruction graph for every option refinement.
  const ResourceProblem resource_problem(
      code_object_bytes, arch, effective_options, effective_options, result.program_inventory,
      result.observation_plan(), candidates, mode_plan.semantics);
  detail::ResourcePlanningStatePtr resource_planning_state_owner =
      detail::make_resource_planning_state(resource_problem, effective_point,
                                           result.resource_plans);
  detail::ResourcePlanningState &resource_planning_state = *resource_planning_state_owner;
  const auto rebuild_resource_plans = [&] {
    ResourcePlanningResult planning = plan_resources(resource_planning_state, effective_options,
                                                     effective_point, resource_problem);
    result.resource_plans = std::move(planning.plans);
    result.errors.insert(result.errors.end(), std::make_move_iterator(planning.errors.begin()),
                         std::make_move_iterator(planning.errors.end()));
  };
  rebuild_resource_plans();
  effective_point.dynamic_stack_spill =
      is_capability_arch(arch) &&
      std::ranges::any_of(result.resource_plans, [&](const CandidateResourcePlan &plan) {
        if (plan.source != RegisterAllocationSource::SpillRequired)
          return false;
        return std::ranges::any_of(plan.owner_kernel_ids, [&](ProgramContainerId owner) {
          const ProgramContainer *kernel = result.program_inventory.container(owner);
          return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
        });
      });
  if (configure_automatic_owner_sgpr(effective_point, resource_problem, result.resource_plans,
                                     result.warnings, resource_planning_state))
    rebuild_resource_plans();
  // Dispatch identity is persistent across every instrumented site, whereas
  // the larger EXEC/VCC/SCC save window is needed only while a probe runs and
  // can use CFG-proven dead registers. Reserve the persistent pair first so a
  // high referenced SGPR does not let the transient window consume the last
  // fresh registers and make dispatch identity spuriously impossible.
  OperatingPointUpdate dispatch_placement = configure_automatic_dispatch_id_sgprs(
      effective_point, resource_problem, result.resource_plans, resource_planning_state);
  result.warnings.insert(result.warnings.end(),
                         std::make_move_iterator(dispatch_placement.diagnostics.begin()),
                         std::make_move_iterator(dispatch_placement.diagnostics.end()));
  if (!dispatch_placement.accepted()) {
    result.outcome = TransformOutcome::Unsupported;
  } else {
    const bool dispatch_placement_changed =
        dispatch_placement.attempted_operating_point != effective_point;
    effective_point = std::move(dispatch_placement.attempted_operating_point);
    if (dispatch_placement_changed)
      rebuild_resource_plans();
  }
  ResourcePlanningResult exec_planning = solve_automatic_exec_save_resources(
      resource_planning_state, effective_options, effective_point, resource_problem);
  if (!exec_planning.success()) {
    result.errors.insert(result.errors.end(), std::make_move_iterator(exec_planning.errors.begin()),
                         std::make_move_iterator(exec_planning.errors.end()));
  } else if (auto accepted = std::move(exec_planning).accept()) {
    effective_point = std::move(accepted->operating_point);
    result.resource_plans = std::move(accepted->site_plans);
    result.warnings.insert(result.warnings.end(),
                           std::make_move_iterator(accepted->diagnostics.begin()),
                           std::make_move_iterator(accepted->diagnostics.end()));
  }
  // Preserve the last complete scalar-placement proof even if a subsequent
  // dispatch override rejects the transform. Unsupported results use this
  // typed partial plan to explain which safe registers had already been
  // established without exposing mutable search options.
  result.operating_point = effective_point;
  OperatingPointAttempt dispatch_fallback =
      detail::plan_dispatch_id_fallback(effective_options, effective_options, effective_point,
                                        resource_problem, result.resource_plans);
  if (dispatch_fallback.accepted()) {
    effective_point = std::move(dispatch_fallback.attempted_operating_point);
    result.warnings.insert(result.warnings.end(),
                           std::make_move_iterator(dispatch_fallback.diagnostics.begin()),
                           std::make_move_iterator(dispatch_fallback.diagnostics.end()));
    rebuild_resource_plans();
  }
  if (result.outcome != TransformOutcome::Unsupported)
    result.operating_point = effective_point;
  std::optional<std::string> scalar_validation_failure;
  if (result.outcome != TransformOutcome::Unsupported) {
    scalar_validation_failure =
        detail::validate_scalar_state(effective_options, effective_options, effective_point, arch);
  }
  if (scalar_validation_failure) {
    result.outcome = TransformOutcome::Unsupported;
    result.warnings.push_back(std::move(*scalar_validation_failure));
  }
  if (result.outcome == TransformOutcome::Unsupported) {
    publish_pending_lowering_rejections(RegisterPlanReason::NoLegalWindow);
    return result;
  }
  // ConSan persistent-state demand depends on the immutable semantic sync
  // admission plan, not merely on the user's request to inventory barriers or
  // atomics. A code object containing only rejected sync sites remains
  // access-only instrumentation.
  PersistentPlacementUpdate persistent_placement =
      configure_automatic_persistent_vgprs(effective_point, resource_problem, effective_options,
                                           result.resource_plans, resource_planning_state);
  result.warnings.insert(result.warnings.end(),
                         std::make_move_iterator(persistent_placement.diagnostics.begin()),
                         std::make_move_iterator(persistent_placement.diagnostics.end()));
  std::vector<PrologueScratchVgprAssignment> prologue_scratch_assignments;
  if (!persistent_placement.accepted()) {
    result.outcome = TransformOutcome::Unsupported;
  } else {
    const bool persistent_placement_changed =
        persistent_placement.attempted_operating_point != effective_point;
    effective_point = std::move(persistent_placement.attempted_operating_point);
    prologue_scratch_assignments = std::move(persistent_placement.prologue_scratch_assignments);
    if (persistent_placement_changed)
      rebuild_resource_plans();
  }
  result.operating_point = effective_point;
  if (result.outcome == TransformOutcome::Unsupported) {
    publish_pending_lowering_rejections(RegisterPlanReason::NoLegalWindow);
    return result;
  }
  if (effective_point.initialize_owner_epoch) {
    for (const ProgramContainer &kernel : result.program_inventory.kernels()) {
      if (!kernel.has_text_range || !kernel.uses_dynamic_stack.value_or(false))
        continue;
      // Dynamic-stack owner/epoch setup must branch in place from the
      // original entry. Keep every ConSan cave/relay allocator off that
      // anchor until the prologue is emitted in the final growth pass.
      (void)resource_reserve_range_if_uncovered(
          resource_planning_state,
          {.text_offset = kernel.entry_text_offset, .size = sizeof(uint32_t)});
    }
  }
  if (std::ranges::any_of(result.resource_plans, [](const CandidateResourcePlan &plan) {
        return plan.reason == RegisterPlanReason::DynamicStack;
      })) {
    result.warnings.emplace_back("ConSan spill does not support a dynamic-stack owning kernel");
  }
  if (effective_options.report_buffer_address &&
      effective_options.report_buffer_size < sizeof(ReportHeader)) {
    result.warnings.emplace_back("ConSan report buffer is smaller than the report ABI header");
  }
  if (effective_options.report_buffer_address &&
      effective_options.report_buffer_size > std::numeric_limits<uint32_t>::max()) {
    result.errors.emplace_back(
        "ConSan report buffer exceeds the 32-bit dynamic record-offset window");
  }
  if (result.errors.empty())
    apply_mode_patches(code_object_bytes, effective_options, effective_point, arch,
                       resource_planning_state, candidates, mode_plan.semantics, result);
  if (result.errors.empty() && result.modified())
    detail::try_apply_owner_epoch_prologue_patch(code_object_bytes, effective_options,
                                                 effective_options, effective_point,
                                                 prologue_scratch_assignments, arch, result);
  if (result.errors.empty())
    result.operating_point = effective_point;
  if (result.outcome == TransformOutcome::Unsupported || !result.errors.empty()) {
    publish_pending_lowering_rejections();
    return result;
  }
  if (result.errors.empty())
    (void)detail::enable_full_workgroup_id_payload(arch, result);
  if (result.errors.empty() && !result.staged_text_fragments.empty() &&
      !finalize_text_rewrites(result.replacement, arch,
                              effective_options.patched_image_growth_limit, "ConSan text programs",
                              result)) {
    result.discard_candidate_modification();
  }
  publish_pending_lowering_rejections();
  std::vector<PatchKind> patch_kinds;
  patch_kinds.reserve(result.patches.size());
  std::ranges::transform(result.patches, std::back_inserter(patch_kinds),
                         [](const auto &patch) { return patch.kind; });
  std::vector<std::string> lowering_summary =
      detail::summarize_lowering(result.modified(), patch_kinds);
  result.warnings.insert(result.warnings.end(), std::make_move_iterator(lowering_summary.begin()),
                         std::make_move_iterator(lowering_summary.end()));
  return result;
}

} // namespace rocjitsu::consan

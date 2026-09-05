// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_candidate_projection.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_pipeline.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
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

namespace rocjitsu {

ConSanTransformArtifacts
try_patch_consan_moi(ConSanTransformArtifacts result, const ConSanOptions &options,
                     const ConSanMoiOperatingPoint &initial_operating_point,
                     std::span<const uint8_t> code_object_bytes, rj_code_arch_t arch) {
  using namespace consan_moi_impl;

  const major_image_ownership::ScopedOwner result_owner(
      major_image_ownership::OwnerKind::ResultImage, result.replacement);
  ConSanOptions effective_options = options;
  ConSanMoiOperatingPoint effective_point = initial_operating_point;
  if (!result.moi_operating_point.owner_transient_sgprs.empty()) {
    effective_point.owner_transient_sgprs = result.moi_operating_point.owner_transient_sgprs;
  }
  if (!result.moi_operating_point.owner_persistent_vgprs.empty()) {
    effective_point.owner_persistent_vgprs = result.moi_operating_point.owner_persistent_vgprs;
  }
  result.outcome =
      result.errors.empty() ? ConSanTransformOutcome::Unchanged : ConSanTransformOutcome::Invalid;
  result.replacement.clear();
  result.resource_plans.clear();
  result.patches.clear();
  if (!result.errors.empty())
    return result;

  /// Close every MOI intent that did not participate in a committed byte
  /// mutation. Unsupported resource plans reject the exact intent IDs they
  /// governed; all other still-pending intents receive an explicit placement
  /// rejection. This terminal transaction never searches patches, anchors,
  /// or graph geometry to rediscover what lowering attempted.
  const auto publish_pending_moi_lowering_rejections =
      [&](std::optional<ConSanRegisterPlanReason> whole_transform_resource_failure = std::nullopt) {
        for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
          if (plan.source != ConSanRegisterAllocationSource::Unsupported || plan.intent_ids.empty())
            continue;
          std::vector<ConSanProbeIntentId> pending;
          for (ConSanProbeIntentId id : plan.intent_ids) {
            const ConSanIntentCoverageEntry *entry = result.coverage_ledger.intent_entry(id);
            if (entry != nullptr && entry->lowering == ConSanLoweringOutcomeKind::Pending &&
                std::ranges::find(pending, id) == pending.end()) {
              pending.push_back(id);
            }
          }
          if (pending.empty())
            continue;
          if (!result.coverage_ledger.publish_lowering_rejection(
                  pending, ConSanLoweringOutcomeKind::ResourceRejected,
                  "MOI resource plan rejected intents: " +
                      std::string(consan_register_plan_reason_name(plan.reason)),
                  plan.reason)) {
            result.errors.emplace_back("ConSan MOI could not publish a resource rejection");
            return;
          }
        }

        for (const ConSanProbeIntent &intent : result.observation_plan().probe_intents) {
          const ConSanIntentCoverageEntry *entry = result.coverage_ledger.intent_entry(intent.id);
          if (entry == nullptr || entry->lowering != ConSanLoweringOutcomeKind::Pending)
            continue;
          const std::array<ConSanProbeIntentId, 1> ids = {intent.id};
          const ConSanLoweringOutcomeKind outcome =
              whole_transform_resource_failure ? ConSanLoweringOutcomeKind::ResourceRejected
                                               : ConSanLoweringOutcomeKind::PlacementRejected;
          const std::string_view detail =
              whole_transform_resource_failure
                  ? "MOI whole-transform resource validation rejected the admitted intent"
                  : "MOI lowerer selected no placement for the admitted intent";
          if (!result.coverage_ledger.publish_lowering_rejection(
                  ids, outcome, std::string(detail), whole_transform_resource_failure)) {
            result.errors.emplace_back("ConSan MOI could not publish a placement rejection");
            return;
          }
        }
      };
  std::vector<ConSanMoiCandidate> moi_candidates = consan_detail::build_moi_candidates(
      result.program_inventory, result.observation_plan(), result.errors);
  if (!result.errors.empty())
    return result;
  if (consan_arch_has_selectable_vgpr_bank(arch)) {
    for (ConSanMoiCandidate &candidate : moi_candidates) {
      const ConSanProgramContainer *container =
          result.program_inventory.container(candidate.site().container.id);
      if (container == nullptr || candidate.anchor() < container->entry_text_offset ||
          candidate.site().decoded_file_offset() < candidate.anchor())
        continue;
      const uint64_t text_file_offset = candidate.site().decoded_file_offset() - candidate.anchor();
      candidate.incoming_vgpr_bank_mode = consan_selectable_vgpr_bank_mode_at(
          arch, code_object_bytes, text_file_offset, container->entry_text_offset,
          candidate.site().decoded_file_offset());
    }
  }
  MoiObjectFacts object_facts;
  object_facts.has_access_candidate = !moi_candidates.empty();
  object_facts.admitted_atomic_count =
      std::ranges::count_if(result.observation_plan().atomic_site_decisions,
                            [](const ConSanAtomicSiteDecision &decision) {
                              return decision.kind == ConSanSiteDecisionKind::Admitted;
                            });
  object_facts.has_admitted_atomic = object_facts.admitted_atomic_count != 0u;
  object_facts.has_admitted_fence = std::ranges::any_of(
      result.observation_plan().fence_site_decisions, [](const ConSanFenceSiteDecision &decision) {
        return decision.kind == ConSanSiteDecisionKind::Admitted;
      });
  object_facts.has_admitted_barrier =
      std::ranges::any_of(result.observation_plan().barrier_site_decisions,
                          [](const ConSanBarrierSiteDecision &decision) {
                            return decision.kind == ConSanSiteDecisionKind::Admitted;
                          });
  object_facts.has_explicit_persistent_state = effective_point.moi_owner_epoch_vgprs.complete();
  object_facts.has_report_buffer = effective_options.moi_report_buffer_address.has_value();
  MoiObjectModePlan mode_plan =
      plan_moi_object_mode(effective_options, effective_options, effective_options, effective_point,
                           object_facts, result.observation_plan(), result.program_inventory);
  effective_options.moi_owner_source = mode_plan.owner_source;
  effective_options.moi_track_atomics = mode_plan.track_atomics;
  effective_options.moi_track_barriers = mode_plan.track_barriers;
  effective_point.moi_initialize_owner_epoch = mode_plan.initialize_owner_epoch;
  result.warnings.insert(result.warnings.end(), std::make_move_iterator(mode_plan.warnings.begin()),
                         std::make_move_iterator(mode_plan.warnings.end()));
  result.errors.insert(result.errors.end(), std::make_move_iterator(mode_plan.errors.begin()),
                       std::make_move_iterator(mode_plan.errors.end()));
  if (!result.errors.empty())
    return result;
  // Register selection iterates as automatic persistent and transient state is
  // chosen. The code bytes, decoded CFG, ownership scopes, and liveness facts
  // do not change during those iterations; retain one analysis state instead
  // of rebuilding the full instruction graph for every option refinement.
  const MoiResourceProblem resource_problem(
      code_object_bytes, arch, effective_options, effective_options, result.program_inventory,
      result.observation_plan(), moi_candidates, mode_plan.semantics);
  MoiResourcePlanningStatePtr resource_planning_state_owner =
      make_moi_resource_planning_state(resource_problem, effective_point, result.resource_plans);
  MoiResourcePlanningState &resource_planning_state = *resource_planning_state_owner;
  const auto rebuild_resource_plans = [&] {
    ConSanMoiResourcePlanningResult planning = plan_moi_resources(
        resource_planning_state, effective_options, effective_point, resource_problem);
    result.resource_plans = std::move(planning.plans);
    result.errors.insert(result.errors.end(), std::make_move_iterator(planning.errors.begin()),
                         std::make_move_iterator(planning.errors.end()));
  };
  rebuild_resource_plans();
  const MoiDynamicStackSpillPolicy dynamic_stack_spill =
      plan_moi_dynamic_stack_spill(effective_options.moi_engine, arch);
  effective_point.moi_dynamic_stack_spill =
      dynamic_stack_spill.backend_supported &&
      std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        if (plan.source != ConSanRegisterAllocationSource::SpillRequired)
          return false;
        return std::ranges::any_of(plan.owner_kernel_ids, [&](ConSanProgramContainerId owner) {
          const ConSanProgramContainer *kernel = result.program_inventory.container(owner);
          return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
        });
      });
  if (configure_automatic_moi_owner_sgpr(effective_point, resource_problem, result.resource_plans,
                                         result.warnings, resource_planning_state))
    rebuild_resource_plans();
  // Dispatch identity is persistent across every instrumented site, whereas
  // the larger EXEC/VCC/SCC save window is needed only while a probe runs and
  // can use CFG-proven dead registers. Reserve the persistent pair first so a
  // high referenced SGPR does not let the transient window consume the last
  // fresh registers and make dispatch identity spuriously impossible.
  ConSanMoiOperatingPointUpdate dispatch_placement = configure_automatic_moi_dispatch_id_sgprs(
      effective_point, resource_problem, result.resource_plans, resource_planning_state);
  result.warnings.insert(result.warnings.end(),
                         std::make_move_iterator(dispatch_placement.diagnostics.begin()),
                         std::make_move_iterator(dispatch_placement.diagnostics.end()));
  if (!dispatch_placement.accepted()) {
    result.outcome = ConSanTransformOutcome::Unsupported;
  } else {
    const bool dispatch_placement_changed =
        dispatch_placement.attempted_operating_point != effective_point;
    effective_point = std::move(dispatch_placement.attempted_operating_point);
    if (dispatch_placement_changed)
      rebuild_resource_plans();
  }
  ConSanMoiResourcePlanningResult exec_planning = solve_automatic_moi_exec_save_resources(
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
  if (configure_inline_moi_owner_sgpr(effective_options, effective_point, result.warnings))
    rebuild_resource_plans();
  // Preserve the last complete scalar-placement proof even if a subsequent
  // dispatch override rejects the transform. Unsupported results use this
  // typed partial plan to explain which safe registers had already been
  // established without exposing mutable search options.
  result.moi_operating_point = effective_point;
  ConSanMoiOperatingPointAttempt dispatch_fallback =
      plan_moi_dispatch_id_fallback(effective_options, effective_options, effective_point,
                                    resource_problem, result.resource_plans);
  if (dispatch_fallback.accepted()) {
    effective_point = std::move(dispatch_fallback.attempted_operating_point);
    result.warnings.insert(result.warnings.end(),
                           std::make_move_iterator(dispatch_fallback.diagnostics.begin()),
                           std::make_move_iterator(dispatch_fallback.diagnostics.end()));
    rebuild_resource_plans();
  }
  if (result.outcome != ConSanTransformOutcome::Unsupported)
    result.moi_operating_point = effective_point;
  std::optional<std::string> scalar_validation_failure;
  if (result.outcome != ConSanTransformOutcome::Unsupported) {
    scalar_validation_failure = validate_moi_scalar_state(
        effective_options, effective_options, effective_point, mode_plan.semantics, arch);
  }
  if (scalar_validation_failure) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.push_back(std::move(*scalar_validation_failure));
  }
  if (result.outcome == ConSanTransformOutcome::Unsupported) {
    publish_pending_moi_lowering_rejections(ConSanRegisterPlanReason::NoLegalWindow);
    return result;
  }
  // Sampled persistent-state demand depends on the immutable semantic sync
  // admission plan, not merely on the user's request to inventory barriers or
  // atomics. A code object containing only rejected sync sites remains
  // access-only instrumentation.
  ConSanMoiPersistentPlacementUpdate persistent_placement =
      configure_automatic_moi_persistent_vgprs(effective_point, resource_problem, effective_options,
                                               result.resource_plans, resource_planning_state);
  result.warnings.insert(result.warnings.end(),
                         std::make_move_iterator(persistent_placement.diagnostics.begin()),
                         std::make_move_iterator(persistent_placement.diagnostics.end()));
  std::vector<ConSanMoiPrologueScratchVgprAssignment> prologue_scratch_assignments;
  if (!persistent_placement.accepted()) {
    result.outcome = ConSanTransformOutcome::Unsupported;
  } else {
    const bool persistent_placement_changed =
        persistent_placement.attempted_operating_point != effective_point;
    effective_point = std::move(persistent_placement.attempted_operating_point);
    prologue_scratch_assignments = std::move(persistent_placement.prologue_scratch_assignments);
    if (persistent_placement_changed)
      rebuild_resource_plans();
  }
  result.moi_operating_point = effective_point;
  if (result.outcome != ConSanTransformOutcome::Unsupported)
    scalar_validation_failure = validate_moi_dispatch_id_vgprs(effective_point);
  if (scalar_validation_failure) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.push_back(std::move(*scalar_validation_failure));
  }
  if (result.outcome == ConSanTransformOutcome::Unsupported) {
    publish_pending_moi_lowering_rejections(ConSanRegisterPlanReason::NoLegalWindow);
    return result;
  }
  if (mode_plan.reserve_dynamic_stack_prologue_entry &&
      effective_point.moi_initialize_owner_epoch) {
    for (const ConSanProgramContainer &kernel : result.program_inventory.kernels()) {
      if (!kernel.has_text_range || !kernel.uses_dynamic_stack.value_or(false))
        continue;
      // Dynamic-stack owner/epoch setup must branch in place from the
      // original entry. Keep every Sampled cave/relay allocator off that
      // anchor until the prologue is emitted in the final growth pass.
      (void)moi_resource_reserve_range_if_uncovered(
          resource_planning_state,
          {.text_offset = kernel.entry_text_offset, .size = sizeof(uint32_t)});
    }
  }
  if (std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return plan.reason == ConSanRegisterPlanReason::DynamicStack;
      })) {
    result.warnings.emplace_back("ConSan MOI spill does not support a dynamic-stack owning kernel");
  }
  if (effective_options.moi_report_buffer_address &&
      effective_options.moi_report_buffer_size < sizeof(ConSanMoiReportHeader)) {
    result.warnings.emplace_back("ConSan MOI report buffer is smaller than the report ABI header");
  }
  if (effective_options.moi_report_buffer_address &&
      effective_options.moi_report_buffer_size > std::numeric_limits<uint32_t>::max()) {
    result.errors.emplace_back(
        "ConSan MOI report buffer exceeds the 32-bit dynamic record-offset window");
  }
  if (result.errors.empty())
    apply_moi_mode_patches(code_object_bytes, effective_options, effective_point, arch,
                           resource_planning_state, moi_candidates, object_facts,
                           mode_plan.semantics, result);
  if (result.errors.empty() && (!mode_plan.prologue_requires_consumer || result.modified()))
    try_apply_owner_epoch_prologue_patch(code_object_bytes, effective_options, effective_options,
                                         effective_point, prologue_scratch_assignments,
                                         mode_plan.semantics, mode_plan.prologue, arch, result);
  if (result.errors.empty())
    result.moi_operating_point = effective_point;
  if (result.outcome == ConSanTransformOutcome::Unsupported || !result.errors.empty()) {
    publish_pending_moi_lowering_rejections();
    return result;
  }
  if (result.errors.empty())
    (void)enable_moi_full_workgroup_id_payload(arch, result);
  if (result.errors.empty() && !result.staged_text_fragments.empty() &&
      !finalize_consan_text_rewrites(result.replacement, arch,
                                     effective_options.patched_image_growth_limit,
                                     "MOI text programs", result)) {
    result.discard_candidate_modification();
  }
  publish_pending_moi_lowering_rejections();
  std::vector<ConSanPatchKind> patch_kinds;
  patch_kinds.reserve(result.patches.size());
  std::ranges::transform(result.patches, std::back_inserter(patch_kinds),
                         [](const auto &patch) { return patch.kind; });
  std::vector<std::string> lowering_summary =
      summarize_moi_lowering(effective_options.moi_engine, result.modified(), patch_kinds);
  result.warnings.insert(result.warnings.end(), std::make_move_iterator(lowering_summary.begin()),
                         std::make_move_iterator(lowering_summary.end()));
  return result;
}

} // namespace rocjitsu

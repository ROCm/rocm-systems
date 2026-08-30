// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"
#include "rocjitsu/code/patch/consan/consan_moi_candidate_projection.h"
#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_pipeline.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_replay.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled.h"
#include "rocjitsu/code/patch/consan/consan_vgpr_bank_state.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {

ConSanTransformArtifacts try_patch_consan_moi(ConSanTransformArtifacts result,
                                              const MoiOptions &options,
                                              std::span<const uint8_t> code_object_bytes,
                                              rj_code_arch_t arch,
                                              ConSanLoweringExecution *execution) {
  using namespace consan_moi_impl;

  const major_image_ownership::ScopedOwner result_owner(
      major_image_ownership::OwnerKind::ResultImage, result.replacement);
  MoiOptions effective_options = options;
  if (!result.moi_operating_point.owner_transient_sgprs.empty()) {
    effective_options.owner_transient_sgprs = result.moi_operating_point.owner_transient_sgprs;
  }
  if (!result.moi_operating_point.owner_persistent_vgprs.empty()) {
    effective_options.owner_persistent_vgprs = result.moi_operating_point.owner_persistent_vgprs;
  }
  result.outcome =
      result.errors.empty() ? ConSanTransformOutcome::Unchanged : ConSanTransformOutcome::Invalid;
  result.replacement.clear();
  result.resource_plans.clear();
  result.patches.clear();
  if (!result.errors.empty())
    return result;
  if (execution != nullptr)
    execution->note_resource_solving_and_lowering();
  std::vector<ConSanMoiCandidate> moi_candidates = consan_detail::build_moi_candidates(
      result.program_inventory, result.observation_plan, result.errors);
  if (!result.errors.empty())
    return result;
  if (consan_arch_has_selectable_vgpr_bank(arch)) {
    for (ConSanMoiCandidate &candidate : moi_candidates) {
      if (candidate.anchor() < candidate.container.entry_text_offset ||
          candidate.file_offset < candidate.anchor())
        continue;
      const uint64_t text_file_offset = candidate.file_offset - candidate.anchor();
      candidate.incoming_vgpr_bank_mode = consan_selectable_vgpr_bank_mode_at(
          arch, code_object_bytes, text_file_offset, candidate.container.entry_text_offset,
          candidate.file_offset);
    }
  }
  MoiObjectFacts object_facts;
  object_facts.has_access_candidate = !moi_candidates.empty();
  object_facts.has_admitted_atomic = std::ranges::any_of(
      result.observation_plan.atomic_site_decisions, [](const ConSanAtomicSiteDecision &decision) {
        return decision.kind == ConSanSiteDecisionKind::Admitted;
      });
  object_facts.has_admitted_fence = std::ranges::any_of(
      result.observation_plan.fence_site_decisions, [](const ConSanFenceSiteDecision &decision) {
        return decision.kind == ConSanSiteDecisionKind::Admitted;
      });
  object_facts.admitted_barrier_count =
      std::ranges::count_if(result.observation_plan.barrier_site_decisions,
                            [](const ConSanBarrierSiteDecision &decision) {
                              return decision.kind == ConSanSiteDecisionKind::Admitted;
                            });
  object_facts.has_admitted_barrier = object_facts.admitted_barrier_count != 0u;
  object_facts.target_supports_dense_barrier_router = consan_is_capability_arch(arch);
  object_facts.has_explicit_persistent_state =
      effective_options.moi_owner_vgpr || effective_options.moi_epoch_vgpr;
  AmdGpuCodeObject original_code_object(code_object_bytes.data(), code_object_bytes.size());
  const uint64_t original_text_size = original_code_object.text_sections().size() == 1
                                          ? original_code_object.text_sections().front()->size()
                                          : 0u;
  object_facts.has_stranded_admitted_barrier =
      original_text_size != 0u &&
      std::ranges::any_of(result.observation_plan.barrier_site_decisions,
                          [&](const ConSanBarrierSiteDecision &decision) {
                            return decision.kind == ConSanSiteDecisionKind::Admitted &&
                                   !compute_sopp_branch_simm16(
                                       decision.semantic_site.physical.original_text_offset,
                                       original_text_size);
                          });
  MoiObjectModePlan mode_plan = plan_moi_object_mode(effective_options, effective_options,
                                                     object_facts, result.observation_plan);
  effective_options.moi_owner_source = mode_plan.owner_source;
  effective_options.moi_track_atomics = mode_plan.track_atomics;
  effective_options.moi_track_barriers = mode_plan.track_barriers;
  effective_options.moi_initialize_owner_epoch = mode_plan.initialize_owner_epoch;
  effective_options.moi_record_replay_dense_barrier_router = mode_plan.dense_barrier_router;
  effective_options.moi_inline_access_present = mode_plan.inline_access_present;
  result.warnings.insert(result.warnings.end(), std::make_move_iterator(mode_plan.warnings.begin()),
                         std::make_move_iterator(mode_plan.warnings.end()));
  result.errors.insert(result.errors.end(), std::make_move_iterator(mode_plan.errors.begin()),
                       std::make_move_iterator(mode_plan.errors.end()));
  if (!result.errors.empty())
    return result;
  const bool has_supported_barrier = object_facts.has_admitted_barrier;
  const bool atomic_or_fence_relevant = mode_plan.atomic_or_fence_relevant;
  const bool explicit_persistent_state = object_facts.has_explicit_persistent_state;
  const bool inline_atomic_without_access = mode_plan.inline_atomic_without_access;
  // Register selection iterates as automatic persistent and transient state is
  // chosen. The code bytes, decoded CFG, ownership scopes, and liveness facts
  // do not change during those iterations; retain one analysis state instead
  // of rebuilding the full instruction graph for every option refinement.
  const MoiResourceProblem resource_problem(code_object_bytes, arch, effective_options,
                                            effective_options, result.program_inventory,
                                            result.observation_plan, moi_candidates);
  MoiResourcePlanningStatePtr resource_planning_state_owner =
      make_moi_resource_planning_state(resource_problem, effective_options, result.resource_plans);
  MoiResourcePlanningState &resource_planning_state = *resource_planning_state_owner;
  const auto rebuild_resource_plans = [&] {
    rebuild_moi_resource_plans(resource_planning_state, effective_options, effective_options,
                               effective_options, effective_options, moi_candidates, result);
  };
  rebuild_resource_plans();
  effective_options.moi_dynamic_stack_spill =
      moi_supports_dynamic_stack_spill(arch, effective_options.moi_engine) &&
      std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        if (plan.source != ConSanRegisterAllocationSource::SpillRequired)
          return false;
        return std::ranges::any_of(plan.owner_descriptor_file_offsets, [&](uint64_t offset) {
          const ConSanKernelInfo *kernel =
              result.program_inventory.find_kernel_by_descriptor(offset);
          return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
        });
      });
  if (configure_automatic_moi_owner_sgpr(effective_options, resource_problem, result.resource_plans,
                                         result.warnings, resource_planning_state))
    rebuild_resource_plans();
  // Dispatch identity is persistent across every instrumented site, whereas
  // the larger EXEC/VCC/SCC save window is needed only while a probe runs and
  // can use CFG-proven dead registers. Reserve the persistent pair first so a
  // high referenced SGPR does not let the transient window consume the last
  // fresh registers and make dispatch identity spuriously impossible.
  ConSanMoiOperatingPointUpdate dispatch_placement = configure_automatic_moi_dispatch_id_sgprs(
      effective_options, resource_problem, result.resource_plans, resource_planning_state);
  result.warnings.insert(result.warnings.end(),
                         std::make_move_iterator(dispatch_placement.diagnostics.begin()),
                         std::make_move_iterator(dispatch_placement.diagnostics.end()));
  if (!dispatch_placement.accepted()) {
    result.outcome = ConSanTransformOutcome::Unsupported;
  } else {
    const bool dispatch_placement_changed = dispatch_placement.changed;
    static_cast<ConSanMoiOperatingPoint &>(effective_options) =
        std::move(dispatch_placement.attempted_operating_point);
    if (dispatch_placement_changed)
      rebuild_resource_plans();
  }
  ConSanMoiResourcePlanningResult exec_planning = solve_automatic_moi_exec_save_resources(
      resource_planning_state, effective_options, effective_options, resource_problem,
      moi_candidates, result);
  if (!exec_planning.success()) {
    result.errors.insert(result.errors.end(), std::make_move_iterator(exec_planning.errors.begin()),
                         std::make_move_iterator(exec_planning.errors.end()));
  } else if (auto accepted = std::move(exec_planning).accept()) {
    static_cast<ConSanMoiOperatingPoint &>(effective_options) =
        std::move(accepted->operating_point);
    result.resource_plans = std::move(accepted->site_plans);
    result.warnings.insert(result.warnings.end(),
                           std::make_move_iterator(accepted->diagnostics.begin()),
                           std::make_move_iterator(accepted->diagnostics.end()));
  }
  if (configure_inline_moi_owner_sgpr(effective_options, effective_options, result.warnings))
    rebuild_resource_plans();
  // Preserve the last complete scalar-placement proof even if a subsequent
  // dispatch override rejects the transform. Unsupported results use this
  // typed partial plan to explain which safe registers had already been
  // established without exposing mutable search options.
  result.moi_operating_point = effective_options;
  ConSanMoiOperatingPointAttempt dispatch_fallback = plan_moi_dispatch_id_fallback(
      effective_options, effective_options, resource_problem, result.resource_plans);
  if (dispatch_fallback.accepted()) {
    static_cast<ConSanMoiOperatingPoint &>(effective_options) =
        std::move(dispatch_fallback.attempted_operating_point);
    result.warnings.insert(result.warnings.end(),
                           std::make_move_iterator(dispatch_fallback.diagnostics.begin()),
                           std::make_move_iterator(dispatch_fallback.diagnostics.end()));
    rebuild_resource_plans();
  }
  if (result.outcome != ConSanTransformOutcome::Unsupported)
    result.moi_operating_point = effective_options;
  std::optional<ConSanMoiScalarValidationFailure> scalar_validation_failure;
  if (result.outcome != ConSanTransformOutcome::Unsupported) {
    scalar_validation_failure = validate_moi_dispatch_id_sgprs(effective_options, effective_options,
                                                               effective_options, arch);
    if (!scalar_validation_failure) {
      scalar_validation_failure = validate_moi_ordinary_scalar_state(
          effective_options, effective_options, effective_options, arch);
    }
  }
  if (scalar_validation_failure) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.push_back(std::move(scalar_validation_failure->diagnostic));
  }
  if (result.outcome == ConSanTransformOutcome::Unsupported) {
    publish_pending_moi_lowering_rejections(result);
    return result;
  }
  // Sampled persistent-state demand depends on the immutable semantic sync
  // admission plan, not merely on the user's request to inventory barriers or
  // atomics. A code object containing only rejected sync sites remains
  // access-only instrumentation.
  ConSanMoiPersistentPlacementUpdate persistent_placement =
      configure_automatic_moi_persistent_vgprs(effective_options, resource_problem,
                                               effective_options, result.resource_plans,
                                               resource_planning_state);
  result.warnings.insert(result.warnings.end(),
                         std::make_move_iterator(persistent_placement.diagnostics.begin()),
                         std::make_move_iterator(persistent_placement.diagnostics.end()));
  std::vector<ConSanMoiPrologueScratchVgprAssignment> prologue_scratch_assignments;
  if (!persistent_placement.accepted()) {
    result.outcome = ConSanTransformOutcome::Unsupported;
  } else {
    const bool persistent_placement_changed = persistent_placement.changed;
    static_cast<ConSanMoiOperatingPoint &>(effective_options) =
        std::move(persistent_placement.attempted_operating_point);
    prologue_scratch_assignments = std::move(persistent_placement.prologue_scratch_assignments);
    if (persistent_placement_changed)
      rebuild_resource_plans();
  }
  result.moi_operating_point = effective_options;
  if (result.outcome != ConSanTransformOutcome::Unsupported)
    scalar_validation_failure = validate_moi_dispatch_id_vgprs(effective_options);
  if (scalar_validation_failure) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.push_back(std::move(scalar_validation_failure->diagnostic));
  }
  if (result.outcome == ConSanTransformOutcome::Unsupported) {
    publish_pending_moi_lowering_rejections(result);
    return result;
  }
  if (effective_options.moi_engine == ConSanMoiEngine::Sampled &&
      moi_initializes_owner_epoch(effective_options, effective_options)) {
    for (const ConSanKernelInfo &kernel : result.program_inventory.kernels()) {
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
  bool owner_epoch_prologue_applied_early = false;
  const bool has_usable_atomic_plan =
      std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return plan.site_kind == ConSanResourceSiteKind::Atomic &&
               plan.source != ConSanRegisterAllocationSource::Unsupported;
      });
  if (result.errors.empty() && inline_atomic_without_access && has_usable_atomic_plan &&
      moi_initializes_owner_epoch(effective_options, effective_options)) {
    // Atomic-only objects do not need access-layout information in their
    // owner/epoch prologue. Emit it before the large atomic helpers so the
    // original kernel entry can reach it without consuming a scarce local
    // branch island.
    try_apply_owner_epoch_prologue_patch(code_object_bytes, effective_options,
                                         prologue_scratch_assignments, arch, result);
    owner_epoch_prologue_applied_early =
        std::ranges::any_of(result.patches, [](const ConSanPatchLoweringProduct &patch) {
          return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
        });
  }
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::Sampled)
    try_apply_direct_sampled_watchpoint_patch(code_object_bytes, effective_options, arch,
                                              resource_planning_state, moi_candidates, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::Sampled)
    try_apply_sampled_atomic_sync_patch(code_object_bytes, effective_options, arch,
                                        resource_planning_state, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::Sampled)
    try_apply_sampled_barrier_sync_patch(code_object_bytes, effective_options, arch,
                                         resource_planning_state, moi_candidates, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::InlineShadow)
    try_apply_inline_shadow_patch(code_object_bytes, effective_options, arch,
                                  resource_planning_state, moi_candidates, result);
  MoiRecordReplayAccessOutput record_replay_access_output;
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::RecordReplay) {
    if (const ConSanTargetProfile *target = consan_target_profile(arch)) {
      try_apply_first_light_access_record_patch(
          code_object_bytes, effective_options, *target, resource_planning_state,
          record_replay_access_output, moi_candidates, result);
    } else if (effective_options.moi_report_buffer_address) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe does not support this architecture");
    }
  }
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::RecordReplay &&
      !explicit_persistent_state &&
      std::ranges::none_of(result.patches,
                           [](const ConSanPatchLoweringProduct &patch) {
                             return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
                                    patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
                           }) &&
      !has_supported_barrier && !atomic_or_fence_relevant) {
    // Planning may admit access sites whose bodies all fail placement. Drop
    // automatic state only when no standalone record can consume it.
    effective_options.moi_initialize_owner_epoch = false;
    effective_options.moi_owner_vgpr.reset();
    effective_options.moi_epoch_vgpr.reset();
    effective_options.moi_record_replay_workgroup_vgprs = {};
    effective_options.moi_persistent_sgprs.record_replay_workgroup = {};
    effective_options.moi_dispatch_id_vgpr.reset();
    effective_options.owner_persistent_vgprs.clear();
    result.moi_operating_point = effective_options;
    result.warnings.emplace_back(
        "ConSan MOI record/replay dropped unconsumed automatic state after all access probes "
        "failed placement");
  }
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::RecordReplay)
    try_apply_atomic_record_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty())
    try_apply_barrier_epoch_patch(code_object_bytes, effective_options, arch,
                                  resource_planning_state, record_replay_access_output, result);
  if (result.errors.empty())
    try_apply_inline_atomic_ordering_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty())
    try_apply_fence_record_patch(code_object_bytes, effective_options, arch, result);
  // Sampled and inline prologues only initialize state consumed by an emitted
  // access or sync probe. The automatic Record/Replay report-sizing pass also
  // has no buffer and therefore cannot emit a consumer yet; keep its semantic
  // inventory pristine so the allocated-buffer retry can select and emit the
  // exact entry tuple. Explicit Record/Replay register controls retain their
  // standalone prologue behavior for host-level lowering tests.
  const bool automatic_record_replay_inventory =
      effective_options.moi_engine == ConSanMoiEngine::RecordReplay &&
      !effective_options.moi_report_buffer_address && !explicit_persistent_state;
  const bool prologue_needs_consumer =
      effective_options.moi_engine == ConSanMoiEngine::Sampled ||
      effective_options.moi_engine == ConSanMoiEngine::InlineShadow ||
      automatic_record_replay_inventory;
  if (result.errors.empty() && !owner_epoch_prologue_applied_early &&
      (!prologue_needs_consumer || result.modified()))
    try_apply_owner_epoch_prologue_patch(code_object_bytes, effective_options,
                                         prologue_scratch_assignments, arch, result);
  if (result.errors.empty())
    result.moi_operating_point = effective_options;
  if (result.outcome == ConSanTransformOutcome::Unsupported || !result.errors.empty()) {
    publish_pending_moi_lowering_rejections(result);
    return result;
  }
  if (result.errors.empty())
    (void)enable_moi_full_workgroup_id_payload(arch, result);
  publish_pending_moi_lowering_rejections(result);
  if (result.modified()) {
    const auto patch_count = [&result](ConSanPatchKind kind) {
      return static_cast<uint32_t>(std::count_if(
          result.patches.begin(), result.patches.end(),
          [kind](const ConSanPatchLoweringProduct &patch) { return patch.kind == kind; }));
    };
    if (patch_count(ConSanPatchKind::InlineMoiAccessRecordStore) != 0) {
      result.warnings.emplace_back(std::string("ConSan MOI ") +
                                   consan_moi_engine_name(effective_options.moi_engine) +
                                   " engine emitted a first-light access record probe");
    }
    if (patch_count(ConSanPatchKind::TrampolineMoiAccessRecordStore) != 0) {
      result.warnings.emplace_back(std::string("ConSan MOI ") +
                                   consan_moi_engine_name(effective_options.moi_engine) +
                                   " engine emitted an appended-cave first-light access record "
                                   "probe");
    }
    if (patch_count(ConSanPatchKind::InlineMoiExactShadowStore) != 0) {
      result.warnings.emplace_back("ConSan MOI inline-shadow engine emitted an exact-shadow "
                                   "publish probe");
    }
    if (patch_count(ConSanPatchKind::TrampolineMoiExactShadowStore) != 0) {
      result.warnings.emplace_back("ConSan MOI inline-shadow engine emitted an appended-cave "
                                   "exact-shadow publish probe");
    }
    if (patch_count(ConSanPatchKind::InlineMoiSampledWatchpointStore) != 0) {
      result.warnings.emplace_back("ConSan MOI sampled engine emitted a direct sampled "
                                   "watchpoint probe");
    }
    if (patch_count(ConSanPatchKind::TrampolineMoiSampledWatchpointStore) != 0) {
      result.warnings.emplace_back("ConSan MOI sampled engine emitted an appended-cave direct "
                                   "sampled watchpoint probe");
    }
    if (patch_count(ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue) != 0) {
      result.warnings.emplace_back(
          "ConSan MOI initialized owner/epoch VGPRs with a kernel-entry prologue");
    }
    if (patch_count(ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue) != 0) {
      result.warnings.emplace_back(
          "ConSan MOI initialized private epoch state with a kernel-entry prologue");
    }
    if (const uint32_t barrier_patches = patch_count(ConSanPatchKind::TrampolineMoiBarrierRecord);
        barrier_patches != 0) {
      result.warnings.emplace_back("ConSan MOI emitted " + std::to_string(barrier_patches) +
                                   " barrier record probe(s)");
    }
    if (const uint32_t inline_epoch_barriers =
            patch_count(ConSanPatchKind::TrampolineMoiInlineEpochBarrier);
        inline_epoch_barriers != 0) {
      result.warnings.emplace_back(
          std::string("ConSan MOI ") + consan_moi_engine_name(effective_options.moi_engine) +
          " engine emitted " + std::to_string(inline_epoch_barriers) + " barrier epoch probe(s)");
    }
    if (const uint32_t inline_atomic_patches =
            patch_count(ConSanPatchKind::TrampolineMoiInlineAtomicOrdering);
        inline_atomic_patches != 0) {
      result.warnings.emplace_back("ConSan MOI inline-shadow engine emitted " +
                                   std::to_string(inline_atomic_patches) +
                                   " inline atomic ordering probe(s)");
    }
    if (const uint32_t atomic_patches = patch_count(ConSanPatchKind::TrampolineMoiAtomicRecord);
        atomic_patches != 0) {
      result.warnings.emplace_back("ConSan MOI emitted " + std::to_string(atomic_patches) +
                                   " atomic record probe(s)");
    }
    if (const uint32_t sampled_sync_patches =
            patch_count(ConSanPatchKind::TrampolineMoiSampledSyncMetadata);
        sampled_sync_patches != 0) {
      result.warnings.emplace_back("ConSan MOI sampled engine emitted " +
                                   std::to_string(sampled_sync_patches) +
                                   " typed synchronization probe(s)");
    }
    if (const uint32_t fence_patches = patch_count(ConSanPatchKind::TrampolineMoiFenceRecord);
        fence_patches != 0) {
      result.warnings.emplace_back("ConSan MOI emitted " + std::to_string(fence_patches) +
                                   " fence record probe(s)");
    }
  } else {
    result.warnings.emplace_back(std::string("ConSan MOI ") +
                                 consan_moi_engine_name(effective_options.moi_engine) +
                                 " engine is an inventory-only stub");
  }
  return result;
}

} // namespace rocjitsu

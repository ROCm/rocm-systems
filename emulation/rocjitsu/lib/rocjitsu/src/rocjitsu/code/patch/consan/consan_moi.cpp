// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_candidate_projection.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

using consan_detail::append_moi_workitem_owner_derivation;
using consan_detail::build_moi_relocated_guest_access_words;
using consan_detail::ConSanMoiDispatchIdCapture;
using consan_detail::has_recent_saveexec;
using consan_detail::moi_guest_access_relocation_requires_adjusted_address;
using consan_detail::moi_workgroup_shadow_initialization_lanes;
using consan_detail::moi_workgroup_shadow_preferred_zero_vgpr_count;
using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_detail::MoiBarrierEvidenceSitePlan;
using consan_detail::MoiEntryScalarBackup;
using consan_detail::MoiFenceEvidenceSitePlan;
using consan_detail::MoiOwnerEpochPrologueEmissionPlan;
using consan_detail::MoiPrivateEpochPrologueEmissionPlan;
using consan_detail::MoiSpecialStateSgprs;
using consan_detail::MoiWorkgroupKeyRegisterPlan;
using consan_detail::MoiWorkgroupShadowClearStoreForm;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_detail::plan_moi_workgroup_shadow_clear;
using consan_detail::range_overlaps;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_atomic_load_u32;
using consan_moi_detail::append_atomic_or_u32_literal;
using consan_moi_detail::append_compare_moi_report_dispatch_id_word;
using consan_moi_detail::append_dynamic_diagnostic_record_address;
using consan_moi_detail::append_dynamic_record_address;
using consan_moi_detail::append_dynamic_record_event_index_store;
using consan_moi_detail::append_dynamic_record_store_moi_report_dispatch_id_pair;
using consan_moi_detail::append_dynamic_record_store_u32_literal;
using consan_moi_detail::append_dynamic_record_store_u32_scalar_src;
using consan_moi_detail::append_dynamic_record_store_u32_vgpr;
using consan_moi_detail::append_dynamic_record_store_workgroup_source;
using consan_moi_detail::append_load_u32_vgpr_at_offset;
using consan_moi_detail::append_moi_report_dispatch_id_pair;
using consan_moi_detail::append_moi_report_dispatch_id_word;
using consan_moi_detail::append_publish_visible_evidence_if_zero;
using consan_moi_detail::append_store_moi_report_dispatch_id_pair;
using consan_moi_detail::append_store_u32_literal;
using consan_moi_detail::append_store_u32_sgpr;
using consan_moi_detail::append_store_u32_vgpr;
using consan_moi_detail::append_store_u32_vgpr_at_offset;
using consan_moi_detail::append_word_bytes;
using consan_moi_detail::append_words_bytes;
using consan_moi_detail::ConSanMoiLiteralDispatchIdPolicy;
using consan_moi_detail::ConSanMoiRecordEmitter;
using consan_moi_detail::ConSanMoiReportDispatchIdWordSource;
using consan_moi_detail::count_nop_padding;
using consan_moi_detail::decode_relocatable_entry_instruction;
using consan_moi_detail::DynamicRecordLayout;
using consan_moi_detail::kAccessRecordLayout;
using consan_moi_detail::kAtomicRecordLayout;
using consan_moi_detail::kBarrierRecordLayout;
using consan_moi_detail::kDiagnosticRecordLayout;
using consan_moi_detail::kFenceRecordLayout;
using consan_moi_detail::moi_has_runtime_hardware_dispatch_id;
using consan_moi_detail::moi_permits_literal_dispatch_identity;
using consan_moi_detail::moi_report_dispatch_id_source_permitted;
using consan_moi_detail::moi_report_dispatch_id_word_source;
using consan_moi_detail::note_moi_persistent_vgpr_state;
using consan_moi_detail::plan_prebuilt_appended_cave;
using consan_moi_detail::record_replay_entry_workgroup_capture_is_unambiguous;
using consan_moi_detail::record_replay_has_entry_workgroup_capture;
using consan_moi_detail::record_replay_requires_entry_workgroup_capture;
using consan_moi_detail::record_replay_uses_automatic_banked_capture;
using consan_moi_detail::resolve_moi_report_layout;

namespace consan_moi_impl {

#include "rocjitsu/code/patch/consan/consan_moi_placement.inc"

#include "rocjitsu/code/patch/consan/consan_moi_emission.inc"

#include "rocjitsu/code/patch/consan/consan_moi_access_apply.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.inc"

#include "rocjitsu/code/patch/consan/consan_moi_probe_planning.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_planning.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_access_emission.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_access.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_replay.inc"

#include "rocjitsu/code/patch/consan/consan_moi_prologue.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sync_common.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.inc"

#include "rocjitsu/code/patch/consan/consan_moi_barrier.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_atomic.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_atomic_emission.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_atomic.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_sync.inc"

#include "rocjitsu/code/patch/consan/consan_moi_pipeline.inc"

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
  if (effective_options.moi_owner_source == ConSanMoiOwnerSource::Automatic) {
    effective_options.moi_owner_source =
        effective_options.moi_engine == ConSanMoiEngine::InlineShadow
            ? ConSanMoiOwnerSource::HwId
            : ConSanMoiOwnerSource::WorkitemId;
  }
  result.outcome =
      result.errors.empty() ? ConSanTransformOutcome::Unchanged : ConSanTransformOutcome::Invalid;
  result.replacement.clear();
  result.resource_plans.clear();
  result.patches.clear();
  if (effective_options.moi_engine == ConSanMoiEngine::InlineShadow &&
      effective_options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
    result.errors.emplace_back(
        "ConSan MOI Inline Shadow requires resident-wave ownership; workitem_id_x is not exact "
        "for multidimensional workgroups");
  }
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
      candidate.incoming_vgpr_bank_mode =
          gfx1250_vgpr_msb_mode_at(code_object_bytes, text_file_offset,
                                   candidate.container.entry_text_offset, candidate.file_offset);
    }
  }
  if (effective_options.moi_engine == ConSanMoiEngine::Sampled &&
      effective_options.moi_track_atomics && moi_candidates.empty()) {
    // Sampled atomics publish ordering only into a selected LDS watchpoint's
    // causal window. In an access-free code object there is no consumer for
    // that metadata, so treating standalone runtime atomics as required
    // instrumentation would reject the workload without improving coverage.
    // Keep them in decoded/fault inventory, but make them operationally not
    // applicable for this object's sampled report and coverage ledger.
    effective_options.moi_track_atomics = false;
    result.warnings.emplace_back(
        "ConSan MOI sampled engine skipped atomic ordering in a code object with no selected "
        "LDS access candidates");
  }
  const bool has_supported_atomic_or_fence =
      std::ranges::any_of(result.observation_plan.atomic_site_decisions,
                          [](const ConSanAtomicSiteDecision &decision) {
                            return decision.kind == ConSanSiteDecisionKind::Admitted;
                          }) ||
      (effective_options.moi_engine == ConSanMoiEngine::RecordReplay &&
       std::ranges::any_of(result.observation_plan.fence_site_decisions,
                           [](const ConSanFenceSiteDecision &decision) {
                             return decision.kind == ConSanSiteDecisionKind::Admitted;
                           }));
  const bool has_supported_barrier =
      std::ranges::any_of(result.observation_plan.barrier_site_decisions,
                          [](const ConSanBarrierSiteDecision &decision) {
                            return decision.kind == ConSanSiteDecisionKind::Admitted;
                          });
  if (effective_options.moi_engine == ConSanMoiEngine::InlineShadow &&
      effective_options.moi_track_barriers && !has_supported_barrier) {
    // The standard profile requests barrier tracking, but an access-only
    // object has no synchronization probe that can consume it. Keep the
    // effective option aligned with admitted instrumentation so downstream
    // layout selection can retain generation-qualified local LDS.
    effective_options.moi_track_barriers = false;
    result.warnings.emplace_back(
        "ConSan MOI skipped barrier tracking for a code object with no admitted barrier sites");
  }
  const size_t supported_barrier_members =
      std::ranges::count_if(result.observation_plan.barrier_site_decisions,
                            [](const ConSanBarrierSiteDecision &decision) {
                              return decision.kind == ConSanSiteDecisionKind::Admitted;
                            });
  AmdGpuCodeObject original_code_object(code_object_bytes.data(), code_object_bytes.size());
  const uint64_t original_text_size = original_code_object.text_sections().size() == 1
                                          ? original_code_object.text_sections().front()->size()
                                          : 0u;
  const bool has_stranded_record_replay_barrier =
      original_text_size != 0u &&
      std::ranges::any_of(result.observation_plan.barrier_site_decisions,
                          [&](const ConSanBarrierSiteDecision &decision) {
                            return decision.kind == ConSanSiteDecisionKind::Admitted &&
                                   !compute_sopp_branch_simm16(
                                       decision.semantic_site.physical.original_text_offset,
                                       original_text_size);
                          });
  // Record/Replay's compact persistent-epoch operating point handles bounded
  // barrier inventories whose sites can reach the appended reservation without
  // the relocated dense router. A large generated object can strand even a
  // small inventory, so reserve the router's key and call-return state based on
  // architectural branch reach as well as the conservative member-count bound.
  // Do not impose that wider liveness window on ordinary kernels: on
  // full-pressure compiler output, otherwise-unused adjacent SGPRs may still
  // carry entry ABI state.
  constexpr size_t kCompactRecordReplayBarrierMemberLimit = 32u;
  effective_options.moi_record_replay_dense_barrier_router =
      effective_options.moi_record_replay_dense_barrier_router ||
      (consan_is_capability_arch(arch) &&
       effective_options.moi_engine == ConSanMoiEngine::RecordReplay &&
       (supported_barrier_members > kCompactRecordReplayBarrierMemberLimit ||
        has_stranded_record_replay_barrier));
  const bool atomic_or_fence_relevant =
      effective_options.moi_track_atomics && has_supported_atomic_or_fence;
  if (effective_options.moi_engine == ConSanMoiEngine::InlineShadow &&
      effective_options.moi_track_atomics && !atomic_or_fence_relevant) {
    // Atomic-token tracking enlarges every Inline access probe even though its
    // release/acquire tables can never be populated in an object with no
    // admitted atomic event. Keep every typed unsupported decision, but do
    // not impose the unusable transaction path or its extra scratch demand.
    for (const ConSanAtomicSiteDecision &decision : result.observation_plan.atomic_site_decisions) {
      if (decision.kind != ConSanSiteDecisionKind::Unsupported)
        continue;
      for (const std::string &container_name : decision.source_containers) {
        result.warnings.emplace_back(
            "ConSan MOI inline atomic ordering skipped " +
            std::string(consan_atomic_policy_reason_name(decision.reason)) + " in " +
            container_name);
      }
    }
    effective_options.moi_track_atomics = false;
    result.warnings.emplace_back(
        "ConSan MOI skipped atomic ordering instrumentation for a code object with no relevant "
        "atomic sites");
  }
  const bool explicit_persistent_state =
      effective_options.moi_owner_vgpr || effective_options.moi_epoch_vgpr;
  if (effective_options.moi_engine == ConSanMoiEngine::RecordReplay && moi_candidates.empty() &&
      !explicit_persistent_state && !has_supported_barrier && !atomic_or_fence_relevant) {
    // Do not synthesize persistent state for an inventory-only object.
    // Standalone barrier, atomic, and fence records still require an exact
    // entry-captured workgroup tuple, so only the true no-consumer case can
    // disable the prologue.
    effective_options.moi_initialize_owner_epoch = false;
    effective_options.moi_track_barriers = false;
    result.warnings.emplace_back(
        "ConSan MOI record/replay skipped persistent state for a code object "
        "with no admitted access, barrier, atomic, or fence sites");
  }
  bool inline_atomic_without_access = false;
  if (effective_options.moi_engine == ConSanMoiEngine::InlineShadow) {
    effective_options.moi_inline_access_present = !moi_candidates.empty();
    inline_atomic_without_access =
        moi_candidates.empty() && effective_options.moi_track_atomics && atomic_or_fence_relevant;
  }
  // Register selection iterates as automatic persistent and transient state is
  // chosen. The code bytes, decoded CFG, ownership scopes, and liveness facts
  // do not change during those iterations; retain one analysis state instead
  // of rebuilding the full instruction graph for every option refinement.
  const MoiResourceProblem resource_problem(code_object_bytes, arch, effective_options,
                                            effective_options, result.program_inventory,
                                            result.observation_plan, moi_candidates);
  MoiResourcePlanningState resource_planning_state(resource_problem, effective_options,
                                                   result.resource_plans);
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
      const bool entry_already_reserved = std::ranges::any_of(
          resource_planning_state.reserved_ranges,
          [&](const ConSanPreappliedReservedRange &reserved) {
            return reserved.size != 0u && kernel.entry_text_offset >= reserved.text_offset &&
                   kernel.entry_text_offset - reserved.text_offset < reserved.size;
          });
      if (!entry_already_reserved) {
        // Dynamic-stack owner/epoch setup must branch in place from the
        // original entry. Keep every Sampled cave/relay allocator off that
        // anchor until the prologue is emitted in the final growth pass.
        resource_planning_state.reserved_ranges.push_back(
            {.text_offset = kernel.entry_text_offset, .size = sizeof(uint32_t)});
      }
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
        std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
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
                           [](const ConSanPatchInfo &patch) {
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
      return static_cast<uint32_t>(
          std::count_if(result.patches.begin(), result.patches.end(),
                        [kind](const ConSanPatchInfo &patch) { return patch.kind == kind; }));
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

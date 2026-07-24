// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
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
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

#include "rocjitsu/code/patch/consan/consan_moi_candidates.inc"

#include "rocjitsu/code/patch/consan/consan_moi_placement.inc"

#include "rocjitsu/code/patch/consan/consan_moi_emission.inc"

#include "rocjitsu/code/patch/consan/consan_moi_access_apply.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_access.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_replay.inc"

#include "rocjitsu/code/patch/consan/consan_moi_prologue.inc"

#include "rocjitsu/code/patch/consan/consan_moi_barrier.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sync_common.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_atomic.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_atomic.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_sync.inc"

#include "rocjitsu/code/patch/consan/consan_moi_pipeline.inc"

bool consan_moi_supports_native_lds_record_replay_mnemonic(std::string_view mnemonic) {
  return is_single_range_native_lds_mnemonic(mnemonic) ||
         two_address_native_lds_offset_scale(mnemonic).has_value();
}

ConSanResult try_patch_consan_moi(ConSanResult result, const ConSanOptions &options,
                                  std::span<const uint8_t> code_object_bytes, rj_code_arch_t arch) {
  ConSanOptions effective_options = options;
  result.flavor = ConSanFlavor::Moi;
  result.moi_engine = effective_options.moi_engine;
  result.moi_report_dispatch_id = effective_options.moi_report_dispatch_id;
  result.modified = false;
  result.elf_bytes.clear();
  result.moi_candidates.clear();
  result.site_dispositions.clear();
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_moi_candidates(kernel, effective_options.flat_provenance_mode, arch, result);
  for (const ConSanFunctionInfo &function : result.functions)
    append_moi_candidates(function, effective_options.flat_provenance_mode, arch, result);
  if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
    for (ConSanMoiCandidate &candidate : result.moi_candidates) {
      if (candidate.text_offset < candidate.container_entry_text_offset ||
          candidate.file_offset < candidate.text_offset)
        continue;
      const uint64_t text_file_offset = candidate.file_offset - candidate.text_offset;
      candidate.gfx1250_vgpr_msb_mode =
          gfx1250_vgpr_msb_mode_at(code_object_bytes, text_file_offset,
                                   candidate.container_entry_text_offset, candidate.file_offset);
    }
  }
  if (!effective_options.test_kernel_name_filter.empty()) {
    std::erase_if(result.moi_candidates, [&](const ConSanMoiCandidate &candidate) {
      return candidate.container_name.find(effective_options.test_kernel_name_filter) ==
             std::string::npos;
    });
  }
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_moi_access_site_dispositions(code_object_bytes, kernel,
                                        effective_options.flat_provenance_mode, effective_options,
                                        arch, result);
  for (const ConSanFunctionInfo &function : result.functions)
    append_moi_access_site_dispositions(code_object_bytes, function,
                                        effective_options.flat_provenance_mode, effective_options,
                                        arch, result);
  if (effective_options.moi_engine == ConSanMoiEngine::Sampled &&
      effective_options.moi_track_atomics && result.moi_candidates.empty()) {
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
  ConSanResult sync_admission = result;
  append_moi_sync_site_dispositions(effective_options, sync_admission);
  const bool has_supported_atomic_or_fence = std::ranges::any_of(
      sync_admission.site_dispositions, [&](const ConSanSiteDispositionRecord &site) {
        return site.disposition == ConSanSiteDisposition::Supported &&
               (site.site_kind == ConSanResourceSiteKind::Atomic ||
                (effective_options.moi_engine == ConSanMoiEngine::RecordReplay &&
                 site.site_kind == ConSanResourceSiteKind::Fence));
      });
  const bool has_supported_barrier = std::ranges::any_of(
      sync_admission.site_dispositions, [](const ConSanSiteDispositionRecord &site) {
        return site.disposition == ConSanSiteDisposition::Supported &&
               site.site_kind == ConSanResourceSiteKind::Barrier;
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
  const size_t supported_barrier_members = std::ranges::count_if(
      sync_admission.site_dispositions, [](const ConSanSiteDispositionRecord &site) {
        return site.disposition == ConSanSiteDisposition::Supported &&
               site.site_kind == ConSanResourceSiteKind::Barrier;
      });
  // Record/Replay's compact persistent-epoch operating point handles bounded
  // barrier inventories without the relocated dense router. Larger RDNA
  // inventories can strand appended barrier bodies beyond SOPP branch reach,
  // so reserve the router's key and call-return state up front. Do not impose
  // that wider liveness window on ordinary kernels: on full-pressure compiler
  // output, otherwise-unused adjacent SGPRs may still carry entry ABI state.
  constexpr size_t kCompactRecordReplayBarrierMemberLimit = 32u;
  effective_options.moi_record_replay_dense_barrier_router =
      effective_options.moi_record_replay_dense_barrier_router ||
      ((is_rdna4_family_arch(arch) || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
        arch == ROCJITSU_CODE_ARCH_CDNA4) &&
       effective_options.moi_engine == ConSanMoiEngine::RecordReplay &&
       supported_barrier_members > kCompactRecordReplayBarrierMemberLimit);
  const auto has_operational_atomic = [&](const auto &container, bool in_kernel) {
    return std::ranges::any_of(container.atomic_sites, [&](const ConSanAtomicSite &site) {
      return atomic_event_kind_for_site(result, container.name, in_kernel, site).has_value();
    });
  };
  const bool has_operational_atomic_or_fence =
      std::ranges::any_of(
          result.kernels,
          [&](const ConSanKernelInfo &kernel) { return has_operational_atomic(kernel, true); }) ||
      std::ranges::any_of(result.functions,
                          [&](const ConSanFunctionInfo &function) {
                            return has_operational_atomic(function, false);
                          }) ||
      std::ranges::any_of(result.moi_fence_candidates, &ConSanMoiFenceCandidate::eligible);
  const bool atomic_or_fence_relevant =
      effective_options.moi_track_atomics &&
      (effective_options.moi_engine == ConSanMoiEngine::RecordReplay
           ? has_operational_atomic_or_fence
           : has_supported_atomic_or_fence);
  if (effective_options.moi_engine == ConSanMoiEngine::InlineShadow &&
      effective_options.moi_track_atomics && !atomic_or_fence_relevant) {
    // Atomic-token tracking enlarges every Inline access probe even though its
    // release/acquire tables can never be populated in an object with no
    // admitted atomic event. Keep every typed unsupported disposition, but do
    // not impose the unusable transaction path or its extra scratch demand.
    for (const ConSanSiteDispositionRecord &site : sync_admission.site_dispositions) {
      if (site.site_kind != ConSanResourceSiteKind::Atomic ||
          site.disposition != ConSanSiteDisposition::Unsupported)
        continue;
      std::string reason = consan_site_disposition_reason_name(site.reason);
      std::ranges::replace(reason, '_', '-');
      result.warnings.emplace_back("ConSan MOI inline atomic ordering skipped " + reason + " in " +
                                   site.container_name);
    }
    effective_options.moi_track_atomics = false;
    result.warnings.emplace_back(
        "ConSan MOI skipped atomic ordering instrumentation for a code object with no relevant "
        "atomic sites");
  }
  const bool explicit_persistent_state =
      effective_options.moi_owner_vgpr || effective_options.moi_epoch_vgpr;
  if (effective_options.moi_engine == ConSanMoiEngine::RecordReplay &&
      result.moi_candidates.empty() && !explicit_persistent_state) {
    // Without access records there is no epoch-bearing consumer for a
    // persistent owner/epoch prologue. Barrier, atomic, and fence records can
    // derive their wave owner at the probe from the common owning descriptor;
    // avoiding unused persistent VGPRs also keeps shared-helper transforms
    // independent of entry-prologue reachability.
    effective_options.moi_init_owner_epoch = false;
    if (!has_supported_barrier && !atomic_or_fence_relevant) {
      effective_options.moi_track_barriers = false;
      result.warnings.emplace_back(
          "ConSan MOI record/replay skipped persistent state for a code object "
          "with no admitted access, barrier, atomic, or fence sites");
    } else {
      result.warnings.emplace_back(
          "ConSan MOI record/replay uses probe-local owner derivation without access records");
    }
  }
  bool inline_atomic_without_access = false;
  if (effective_options.moi_engine == ConSanMoiEngine::InlineShadow) {
    std::vector<const ConSanMoiCandidate *> inline_access_candidates =
        find_inline_shadow_access_candidates(result, code_object_bytes, arch);
    apply_test_kernel_filter(inline_access_candidates, effective_options);
    effective_options.moi_inline_access_present = !inline_access_candidates.empty();
    inline_atomic_without_access = inline_access_candidates.empty() &&
                                   effective_options.moi_track_atomics && atomic_or_fence_relevant;
    if (effective_options.moi_inline_workgroup_shadow && inline_access_candidates.empty()) {
      // The final EXEC-save pair belongs exclusively to the workgroup-filtered
      // exact-shadow access path. Atomic publication uses masks through +20:+21
      // and must not reserve +22:+23 when no access probe can consume them.
      // Besides avoiding dead state, this leaves the complete s0:s105 RDNA4
      // scalar file usable by high-pressure atomic-only helpers.
      effective_options.moi_inline_workgroup_shadow = false;
      result.warnings.emplace_back(
          "ConSan MOI inline shadow omitted unused access-only workgroup-filter state");
    }
  }
  // Register selection iterates as automatic persistent and transient state is
  // chosen. The code bytes, decoded CFG, ownership scopes, and liveness facts
  // do not change during those iterations; retain one analysis state instead
  // of rebuilding the full instruction graph for every option refinement.
  MoiResourcePlanningState resource_planning_state(code_object_bytes, arch, effective_options,
                                                   result);
  rebuild_moi_resource_plans(resource_planning_state, effective_options, result);
  const bool supports_dynamic_stack_spill =
      effective_options.moi_engine == ConSanMoiEngine::InlineShadow ||
      ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
       (effective_options.moi_engine == ConSanMoiEngine::RecordReplay ||
        effective_options.moi_engine == ConSanMoiEngine::Sampled));
  effective_options.moi_dynamic_stack_spill =
      supports_dynamic_stack_spill &&
      std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        if (plan.source != ConSanRegisterAllocationSource::SpillRequired)
          return false;
        return std::ranges::any_of(plan.owner_descriptor_file_offsets, [&](uint64_t offset) {
          const ConSanKernelInfo *kernel = kernel_for_descriptor(result, offset);
          return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
        });
      });
  if (configure_automatic_moi_owner_sgpr(effective_options, result, arch))
    rebuild_moi_resource_plans(resource_planning_state, effective_options, result);
  // Dispatch identity is persistent across every instrumented site, whereas
  // the larger EXEC/VCC/SCC save window is needed only while a probe runs and
  // can use CFG-proven dead registers. Reserve the persistent pair first so a
  // high referenced SGPR does not let the transient window consume the last
  // fresh registers and make dispatch identity spuriously impossible.
  if (configure_automatic_moi_dispatch_id_sgprs(effective_options, result, arch))
    rebuild_moi_resource_plans(resource_planning_state, effective_options, result);
  if (configure_automatic_moi_exec_save_sgprs(effective_options, result, code_object_bytes, arch))
    rebuild_moi_resource_plans(resource_planning_state, effective_options, result);
  if (configure_gfx1250_record_replay_dispatch_id_overrides(effective_options, result,
                                                            code_object_bytes, arch))
    rebuild_moi_resource_plans(resource_planning_state, effective_options, result);
  if (result.outcome == ConSanTransformOutcome::Unsupported ||
      !validate_moi_dispatch_id_sgprs(effective_options, result, arch) ||
      !validate_moi_ordinary_scalar_state(effective_options, result, arch)) {
    finalize_moi_site_lowering_outcomes(result);
    summarize_moi_resource_plans(result);
    return result;
  }
  // Sampled persistent-state demand depends on semantic sync admission, not
  // merely on the user's request to inventory barriers or atomics. Append its
  // dispositions before choosing automatic owner/epoch VGPRs so a code object
  // containing only rejected sync sites remains access-only instrumentation.
  if (effective_options.moi_engine == ConSanMoiEngine::Sampled)
    append_moi_sync_site_dispositions(effective_options, result);
  if (configure_automatic_moi_persistent_vgprs(effective_options, result, code_object_bytes, arch))
    rebuild_moi_resource_plans(resource_planning_state, effective_options, result);
  if (result.outcome == ConSanTransformOutcome::Unsupported) {
    finalize_moi_site_lowering_outcomes(result);
    summarize_moi_resource_plans(result);
    return result;
  }
  if (std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return plan.reason == ConSanRegisterPlanReason::DynamicStack;
      })) {
    result.warnings.emplace_back("ConSan MOI spill does not support a dynamic-stack owning kernel");
  }
  if (!result.resolved_moi_owner_vgpr)
    result.resolved_moi_owner_vgpr = effective_options.moi_owner_vgpr;
  if (!result.resolved_moi_epoch_vgpr)
    result.resolved_moi_epoch_vgpr = effective_options.moi_epoch_vgpr;
  if (!result.resolved_moi_workgroup_key_vgpr)
    result.resolved_moi_workgroup_key_vgpr = effective_options.moi_workgroup_key_vgpr;
  if (!result.resolved_moi_sample_sequence_vgpr)
    result.resolved_moi_sample_sequence_vgpr = effective_options.moi_sample_sequence_vgpr;
  if (!result.resolved_moi_exec_save_sgpr)
    result.resolved_moi_exec_save_sgpr = effective_options.moi_exec_save_sgpr;
  if (!result.resolved_moi_owner_sgpr)
    result.resolved_moi_owner_sgpr = effective_options.moi_owner_sgpr;
  if (!result.resolved_moi_dispatch_id_sgpr)
    result.resolved_moi_dispatch_id_sgpr = effective_options.moi_dispatch_id_sgpr;
  if (effective_options.moi_engine != ConSanMoiEngine::Sampled)
    append_moi_sync_site_dispositions(effective_options, result);
  if (effective_options.moi_report_buffer_address &&
      effective_options.moi_report_buffer_size < sizeof(ConSanMoiReportHeader)) {
    result.warnings.emplace_back("ConSan MOI report buffer is smaller than the report ABI header");
  }
  bool owner_epoch_prologue_applied_early = false;
  const bool has_usable_atomic_plan =
      std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return plan.site_kind == ConSanResourceSiteKind::Atomic &&
               plan.source != ConSanRegisterAllocationSource::Unsupported;
      });
  if (result.errors.empty() && inline_atomic_without_access && has_usable_atomic_plan &&
      effective_options.moi_init_owner_epoch) {
    // Atomic-only objects do not need access-layout information in their
    // owner/epoch prologue. Emit it before the large atomic helpers so the
    // original kernel entry can reach it without consuming a scarce local
    // branch island.
    try_apply_owner_epoch_prologue_patch(code_object_bytes, effective_options, arch, result);
    owner_epoch_prologue_applied_early =
        std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
          return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
        });
  }
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::Sampled)
    try_apply_direct_sampled_watchpoint_patch(code_object_bytes, effective_options, arch,
                                              resource_planning_state, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::Sampled)
    try_apply_sampled_atomic_sync_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::Sampled)
    try_apply_sampled_barrier_sync_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::InlineShadow)
    try_apply_inline_shadow_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::RecordReplay)
    try_apply_first_light_access_record_patch(code_object_bytes, effective_options, arch,
                                              resource_planning_state, result);
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::RecordReplay &&
      !explicit_persistent_state &&
      std::ranges::none_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
               patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
      })) {
    // Planning may admit access sites whose bodies all fail placement. In
    // that case standalone barrier/atomic/fence records must use their proven
    // probe-local owner, just like a code object with no access candidates.
    // Leaving the automatic owner/epoch pair enabled would add an unconsumed
    // entry prologue; dynamic-stack kernels can have no reachable prologue
    // island and would transactionally lose otherwise valid barrier records.
    effective_options.moi_init_owner_epoch = false;
    effective_options.moi_owner_vgpr.reset();
    effective_options.moi_epoch_vgpr.reset();
    result.warnings.emplace_back(
        "ConSan MOI record/replay reverted to probe-local state after all access probes failed "
        "placement");
  }
  if (result.errors.empty() && effective_options.moi_engine == ConSanMoiEngine::RecordReplay)
    try_apply_atomic_record_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty())
    try_apply_barrier_epoch_patch(code_object_bytes, effective_options, arch,
                                  resource_planning_state, result);
  if (result.errors.empty())
    try_apply_inline_atomic_ordering_patch(code_object_bytes, effective_options, arch, result);
  if (result.errors.empty())
    try_apply_fence_record_patch(code_object_bytes, effective_options, arch, result);
  // Sampled and inline prologues only initialize state consumed by an emitted
  // access or sync probe. Do not turn their otherwise unsupported/inventory-
  // only transforms into vacuous modifications containing setup code alone.
  // Record/replay retains standalone prologue emission for its explicit
  // prologue controls and host-level lowering tests.
  const bool prologue_needs_consumer =
      effective_options.moi_engine == ConSanMoiEngine::Sampled ||
      effective_options.moi_engine == ConSanMoiEngine::InlineShadow;
  if (result.errors.empty() && !owner_epoch_prologue_applied_early &&
      (!prologue_needs_consumer || result.modified))
    try_apply_owner_epoch_prologue_patch(code_object_bytes, effective_options, arch, result);
  finalize_moi_site_lowering_outcomes(result);
  summarize_moi_resource_plans(result);
  if (result.modified) {
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

ConSanMoiAutoReportInventory
inventory_consan_moi_auto_report(const ConSanResult &result, const ConSanOptions &options,
                                 std::span<const uint8_t> code_object_bytes) {
  ConSanMoiAutoReportInventory inventory;
  inventory.engine = options.moi_engine;
  const rj_code_arch_t arch = result.arch_name == "cdna3"   ? ROCJITSU_CODE_ARCH_CDNA3
                              : result.arch_name == "cdna4" ? ROCJITSU_CODE_ARCH_CDNA4
                                                            : ROCJITSU_CODE_ARCH_INVALID;

  size_t selected_candidate_count = 0;
  bool selected_flat_candidate = false;
  bool selected_dynamic_lds_owner = false;
  uint64_t selected_native_lds_extent = 0;
  std::unordered_set<uint64_t> dynamic_lds_descriptors;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.has_dynamic_lds)
      dynamic_lds_descriptors.insert(kernel.descriptor_file_offset);
  }
  const size_t selected_candidate_limit =
      options.max_patches_is_expert_limit ? options.max_patches : result.moi_candidates.size();
  for (size_t candidate_index = 0; candidate_index < result.moi_candidates.size();
       ++candidate_index) {
    if (selected_candidate_count >= selected_candidate_limit)
      break;
    const ConSanMoiCandidate &candidate = result.moi_candidates[candidate_index];
    const bool supported =
        std::ranges::any_of(result.site_dispositions, [&](const ConSanSiteDispositionRecord &site) {
          return site.site_kind == ConSanResourceSiteKind::Access &&
                 site.disposition == ConSanSiteDisposition::Supported &&
                 site.container_name == candidate.container_name &&
                 site.text_offset == candidate.text_offset;
        });
    if (!supported)
      continue;
    const auto ranges = candidate_access_ranges(code_object_bytes, candidate, arch);
    if (!ranges || ranges->empty())
      continue;
    ++selected_candidate_count;
    selected_flat_candidate |= candidate.source == ConSanMoiCandidateSource::FlatGroup ||
                               candidate.source == ConSanMoiCandidateSource::FlatMaybeGroup;
    selected_dynamic_lds_owner |=
        std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
          return plan.site_kind == ConSanResourceSiteKind::Access &&
                 plan.candidate_index == candidate_index &&
                 std::ranges::any_of(plan.owner_descriptor_file_offsets, [&](uint64_t owner) {
                   return dynamic_lds_descriptors.contains(owner);
                 });
        });
    if (candidate.source == ConSanMoiCandidateSource::NativeLds) {
      for (const ConSanMoiAccessRange &range : *ranges) {
        selected_native_lds_extent =
            std::max<uint64_t>(selected_native_lds_extent,
                               static_cast<uint64_t>(range.static_byte_offset) + range.byte_count);
      }
    }
    inventory.access_range_count += ranges->size();
  }

  const auto supported_count = [&](ConSanResourceSiteKind kind) {
    return static_cast<uint64_t>(std::ranges::count_if(
        result.site_dispositions, [&](const ConSanSiteDispositionRecord &site) {
          return site.site_kind == kind && site.disposition == ConSanSiteDisposition::Supported;
        }));
  };
  inventory.barrier_event_count = supported_count(ConSanResourceSiteKind::Barrier);
  inventory.atomic_event_count = supported_count(ConSanResourceSiteKind::Atomic);
  inventory.fence_event_count = supported_count(ConSanResourceSiteKind::Fence);
  if (options.moi_engine == ConSanMoiEngine::RecordReplay) {
    // Keep report sizing coupled to operational candidates while the
    // persistent-state path still uses ownerless runtime helpers as a
    // conservative reachability signal. Coverage excludes those impossible
    // sites, but shrinking this buffer early changes prologue placement for
    // dynamic-stack pressure kernels.
    inventory.atomic_event_count = static_cast<uint64_t>(
        std::ranges::count_if(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
          return plan.site_kind == ConSanResourceSiteKind::Atomic;
        }));
    inventory.fence_event_count = static_cast<uint64_t>(
        std::ranges::count_if(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
          return plan.site_kind == ConSanResourceSiteKind::Fence;
        }));
  }
  inventory.diagnostic_count = std::max<uint64_t>(inventory.access_range_count, 1u);
  if (options.moi_engine == ConSanMoiEngine::InlineShadow)
    inventory.inline_compact_token_mapping_count = selected_candidate_count;

  if (options.moi_engine == ConSanMoiEngine::Sampled) {
    const uint64_t bank_count = options.moi_runtime_sample_stride > 1u ? 8u : 1u;
    const uint64_t access_banks =
        inventory.access_range_count > std::numeric_limits<uint64_t>::max() / bank_count
            ? std::numeric_limits<uint64_t>::max()
            : inventory.access_range_count * bank_count;
    // Synchronization metadata is attached to an already selected access
    // window; atomics do not create independent sampled windows. Sizing from
    // atomic inventory alone otherwise provisions vacuous capacity for code
    // objects with no admissible LDS access.
    inventory.sampled_range_bank_count = access_banks;
    inventory.sampled_watchpoint_count = inventory.sampled_range_bank_count;
    inventory.sampled_bank_count_adaptive = bank_count > 1u;
    if (options.moi_track_barriers) {
      inventory.barrier_event_count = 0;
      for (const ConSanSyncSequence &sequence : result.sync_sequences) {
        if (consan_moi_sampled_qualifies_barrier_sequence(sequence))
          inventory.barrier_event_count += sequence.member_event_identities.size();
      }
    }
  }

  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    std::unordered_set<uint64_t> descriptor_offsets;
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      descriptor_offsets.insert(plan.owner_descriptor_file_offsets.begin(),
                                plan.owner_descriptor_file_offsets.end());
    }
    for (const ConSanMoiCandidate &candidate : result.moi_candidates) {
      if (candidate.kernel_descriptor_file_offset)
        descriptor_offsets.insert(*candidate.kernel_descriptor_file_offset);
    }
    for (uint64_t descriptor_offset : descriptor_offsets) {
      if (descriptor_offset > code_object_bytes.size() ||
          sizeof(KD) > code_object_bytes.size() - descriptor_offset)
        continue;
      KD descriptor{};
      std::memcpy(&descriptor, code_object_bytes.data() + descriptor_offset, sizeof(descriptor));
      inventory.inline_lds_bytes =
          std::max<uint64_t>(inventory.inline_lds_bytes, descriptor.group_segment_fixed_size);
    }
    // A native LDS instruction can name storage that is not reflected in the
    // kernel descriptor's fixed-LDS field (for example, target-managed tensor
    // transfers or dynamically sized LDS). Such a descriptor cannot host a
    // descriptor-grown local mirror. A nonzero field is not sufficient proof:
    // some dynamic-LDS objects retain a one-byte placeholder even though a
    // decoded access itself spans more bytes. Provision the external
    // full-aperture fallback just as we do for group-FLAT accesses.
    const bool descriptor_opaque_lds = inventory.access_range_count != 0u &&
                                       inventory.inline_lds_bytes < selected_native_lds_extent;
    if (selected_flat_candidate || selected_dynamic_lds_owner || descriptor_opaque_lds) {
      const uint64_t external_lds_bytes = options.moi_max_workgroup_lds_bytes.value_or(
          result.arch_name == "gfx1250"
              ? consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_GFX1250)
              : static_cast<uint32_t>(kConSanMoiInlineShadowConservativeExactShadowEntries *
                                      consan_moi_exact_shadow::granule_bytes));
      inventory.inline_lds_bytes =
          std::max<uint64_t>(inventory.inline_lds_bytes, external_lds_bytes);
    }
    const uint64_t ordering_capacity = std::max<uint64_t>(
        inventory.atomic_event_count, kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
    inventory.inline_atomic_release_count = ordering_capacity;
    inventory.inline_causal_snapshot_count = ordering_capacity;
    inventory.inline_acquired_epoch_token_count =
        std::max<uint64_t>(ordering_capacity, kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity);
    inventory.diagnostic_count = std::max<uint64_t>(
        inventory.diagnostic_count, kConSanMoiInlineShadowDefaultDiagnosticCapacity);
  }
  if (options.moi_engine == ConSanMoiEngine::RecordReplay)
    return fit_consan_moi_record_replay_auto_report_inventory(inventory);
  return fit_consan_moi_sampled_auto_report_inventory(inventory);
}

} // namespace rocjitsu

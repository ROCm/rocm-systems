// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Record/Replay owns event selection and evidence semantics. Shared resource,
// placement, descriptor, and target emission mechanics enter through declared
// contracts rather than through the MOI textual include order.

#include "rocjitsu/code/patch/consan/consan_moi_record_replay.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_apply.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_local_island_allocator.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/trampoline_builder.h"

#include <algorithm>
#include <cstring>
#include <ranges>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_detail::MoiFenceEvidenceSitePlan;
using consan_detail::MoiSpecialStateSgprs;
using consan_moi_detail::append_words_bytes;
using consan_moi_detail::count_nop_padding;
using consan_moi_detail::decode_relocatable_entry_instruction;
using consan_moi_detail::moi_has_runtime_hardware_dispatch_id;
using consan_moi_detail::record_replay_uses_automatic_banked_capture;
using consan_moi_detail::resolve_moi_report_layout;

namespace consan_moi_impl {

MoiObjectModePlan plan_record_replay_object_mode(const ConSanRequest &request,
                                                 const ConSanMoiOperatingPoint &point,
                                                 const MoiObjectFacts &facts,
                                                 const ConSanObservationPlan &) {
  MoiObjectModePlan plan =
      make_moi_object_mode_plan(request, point, ConSanMoiOwnerSource::WorkitemId);
  plan.atomic_or_fence_relevant =
      plan.track_atomics && (facts.has_admitted_atomic || facts.has_admitted_fence);
  // A bufferless automatic pass only sizes the report. Do not perturb its
  // semantic inventory with a prologue before the allocated-buffer retry can
  // emit an actual consumer. Explicit register controls retain their
  // standalone prologue behavior for host-level lowering tests.
  plan.prologue_requires_consumer =
      !facts.has_report_buffer && !facts.has_explicit_persistent_state;

  constexpr size_t kCompactBarrierMemberLimit = 32u;
  plan.dense_barrier_router =
      plan.dense_barrier_router || (facts.target_supports_dense_barrier_router &&
                                    (facts.admitted_barrier_count > kCompactBarrierMemberLimit ||
                                     facts.has_stranded_admitted_barrier));

  if (!facts.has_access_candidate && !facts.has_explicit_persistent_state &&
      !facts.has_admitted_barrier && !plan.atomic_or_fence_relevant) {
    plan.initialize_owner_epoch = false;
    plan.track_barriers = false;
    plan.warnings.emplace_back(
        "ConSan MOI record/replay skipped persistent state for a code object "
        "with no admitted access, barrier, atomic, or fence sites");
  }
  return plan;
}

void apply_record_replay_mode_patches(std::span<const uint8_t> bytes, MoiOptions &options,
                                      rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                                      std::span<const ConSanMoiCandidate> candidates,
                                      const MoiObjectFacts &facts,
                                      ConSanTransformArtifacts &result) {
  MoiRecordReplayAccessOutput access_output;
  if (const ConSanTargetProfile *target = consan_target_profile(arch)) {
    try_apply_first_light_access_record_patch(bytes, options, *target, resource_state,
                                              access_output, candidates, result);
  } else if (options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI first-light probe does not support this architecture");
  }
  if (!result.errors.empty())
    return;

  const bool emitted_access =
      std::ranges::any_of(result.patches, [](const ConSanPatchLoweringProduct &patch) {
        return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
               patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
      });
  const bool atomic_or_fence_relevant =
      options.moi_track_atomics && (facts.has_admitted_atomic || facts.has_admitted_fence);
  if (!facts.has_explicit_persistent_state && !emitted_access && !facts.has_admitted_barrier &&
      !atomic_or_fence_relevant) {
    // Planning may admit access sites whose bodies all fail placement. Drop
    // automatic state only when no standalone record can consume it.
    options.moi_initialize_owner_epoch = false;
    options.moi_owner_vgpr.reset();
    options.moi_epoch_vgpr.reset();
    options.moi_record_replay_workgroup_vgprs = {};
    options.moi_persistent_sgprs.record_replay_workgroup = {};
    options.moi_dispatch_id_vgpr.reset();
    options.owner_persistent_vgprs.clear();
    result.moi_operating_point = options;
    result.warnings.emplace_back(
        "ConSan MOI record/replay dropped unconsumed automatic state after all access probes "
        "failed placement");
  }
  try_apply_atomic_record_patch(bytes, options, arch, result);
  if (result.errors.empty())
    try_apply_record_replay_barrier_patch(bytes, options, arch, resource_state, access_output,
                                          result);
  if (result.errors.empty())
    try_apply_fence_record_patch(bytes, options, arch, result);
}

uint16_t record_replay_access_scratch_vgpr_count(const ConSanRequest &request,
                                                 const BoundRuntimeResources &resources,
                                                 const ConSanMoiOperatingPoint &,
                                                 const ConSanMoiCandidate &candidate,
                                                 rj_code_arch_t arch) {
  const uint16_t address_count = flat_access_address_scratch_count(candidate);
  return static_cast<uint16_t>(
      (record_replay_uses_automatic_banked_capture(request, resources) ? 10u : 6u) +
      (address_count != 0u ? address_count
       : candidate.is_direct_to_lds() || moi_load_clobbers_address(candidate) ||
               moi_access_requires_high_bank_address_capture(candidate, arch)
           ? 1u
           : 0u) +
      (consan_uses_gfx12_cdna_execution(arch) && candidate.is_native_two_range() &&
               candidate.encoded_offset_scale_bytes() > 8u
           ? 1u
           : 0u));
}

MoiPersistentStateDemand plan_record_replay_persistent_state_demand(
    const ConSanRequest &request, const BoundRuntimeResources &resources,
    const ConSanMoiOperatingPoint &point, const MoiPersistentStateFacts &facts) {
  MoiPersistentStateDemand demand =
      make_exact_workgroup_capture_demand(request, resources, point, facts);
  demand.needs_persistent_state =
      moi_initializes_owner_epoch(request, point) || demand.needs_entry_workgroup_tuple;
  demand.needs_persistent_dispatch_capture =
      facts.access_count && record_replay_uses_automatic_banked_capture(request, resources) &&
      !consan_moi_detail::moi_has_runtime_hardware_dispatch_id(point);
  // This is an automatic operating-point choice, not a user-facing limit.
  // Small barrier inventories benefit from compact persistent-epoch barriers;
  // larger barrier-dense objects benefit more from private-epoch access
  // coalescing. The access-heavy case avoids millions of dynamic records.
  constexpr size_t kCompactBarrierSiteLimit = 32u;
  demand.prefer_compact_barriers = request.moi_track_barriers && facts.barrier_count &&
                                   (facts.barrier_count <= kCompactBarrierSiteLimit ||
                                    facts.access_count >= 2u * facts.barrier_count);
  demand.private_state_supported = true;
  demand.scalar_state_required_for_private_or_overflow = true;
  demand.prefer_private_epoch_for_descriptor_growth = true;
  return demand;
}

MoiOperandOverlapSpillPolicy
record_replay_operand_overlap_spill(const MoiOperandOverlapSpillContext &context) {
  MoiOperandOverlapSpillPolicy policy;
  policy.supported =
      context.access_candidate != nullptr && context.site_kind == ConSanResourceSiteKind::Access &&
      consan_is_capability_arch(context.arch) && !context.request.moi_dynamic_access_records &&
      !context.guest_replay_requires_disjoint_address_scratch;
  return policy;
}

const MoiModeOperations kRecordReplayModeOperations = {
    plan_record_replay_object_mode,
    apply_record_replay_mode_patches,
    record_replay_access_scratch_vgpr_count,
    plan_record_replay_persistent_state_demand,
    false,
    true,
    record_replay_operand_overlap_spill,
    nullptr,
};

#include "rocjitsu/code/patch/consan/consan_moi_record_replay.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_atomic.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_fence.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu

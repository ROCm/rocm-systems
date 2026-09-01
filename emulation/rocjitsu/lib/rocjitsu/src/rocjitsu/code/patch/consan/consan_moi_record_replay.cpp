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
#include "rocjitsu/code/patch/consan/consan_moi_report_planning.h"
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

MoiObjectModePlan
plan_record_replay_object_mode(const ConSanRequest &request, const BoundRuntimeResources &resources,
                               const TransformPolicy &, const ConSanMoiOperatingPoint &point,
                               const MoiObjectFacts &facts, const ConSanObservationPlan &) {
  MoiObjectModePlan plan =
      make_moi_object_mode_plan(request, point, ConSanMoiOwnerSource::WorkitemId);
  const bool atomic_or_fence_relevant =
      plan.track_atomics && (facts.has_admitted_atomic || facts.has_admitted_fence);
  // A bufferless automatic pass only sizes the report. Do not perturb its
  // semantic inventory with a prologue before the allocated-buffer retry can
  // emit an actual consumer. Explicit register controls retain their
  // standalone prologue behavior for host-level lowering tests.
  plan.prologue_requires_consumer =
      !facts.has_report_buffer && !facts.has_explicit_persistent_state;

  constexpr size_t kCompactBarrierMemberLimit = 32u;
  plan.semantics.dense_barrier_router =
      facts.target_supports_dense_barrier_router &&
      (facts.admitted_barrier_count > kCompactBarrierMemberLimit ||
       facts.has_stranded_admitted_barrier);
  plan.semantics.report_layout =
      resolve_moi_report_layout(resources, ConSanMoiEngine::RecordReplay,
                                consan_moi_report_buffer_layout_for_bytes(
                                    resources.moi_report_buffer_size, request.moi_track_barriers,
                                    request.moi_track_atomics, request.moi_track_atomics));

  if (!facts.has_access_candidate && !facts.has_explicit_persistent_state &&
      !facts.has_admitted_barrier && !atomic_or_fence_relevant) {
    plan.initialize_owner_epoch = false;
    plan.track_barriers = false;
    plan.warnings.emplace_back(
        "ConSan MOI record/replay skipped persistent state for a code object "
        "with no admitted access, barrier, atomic, or fence sites");
  }
  return plan;
}

void apply_record_replay_mode_patches(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                      ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
                                      MoiResourcePlanningState &resource_state,
                                      std::span<const ConSanMoiCandidate> candidates,
                                      const MoiObjectFacts &facts,
                                      const MoiObjectModeSemantics &semantics,
                                      ConSanTransformArtifacts &result) {
  MoiRecordReplayAccessOutput access_output;
  if (const ConSanTargetProfile *target = consan_target_profile(arch)) {
    try_apply_first_light_access_record_patch(bytes, options, operating_point, *target,
                                              resource_state, access_output, candidates, semantics,
                                              result);
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
    operating_point.moi_initialize_owner_epoch = false;
    operating_point.reset_moi_owner_epoch_vgprs();
    operating_point.moi_record_replay_workgroup_vgprs = {};
    operating_point.moi_persistent_sgprs.record_replay_workgroup = {};
    operating_point.moi_dispatch_identity.reset_vgpr();
    operating_point.owner_persistent_vgprs.clear();
    result.moi_operating_point = operating_point;
    result.warnings.emplace_back(
        "ConSan MOI record/replay dropped unconsumed automatic state after all access probes "
        "failed placement");
  }
  try_apply_atomic_record_patch(bytes, options, operating_point, semantics, arch, result);
  if (result.errors.empty())
    try_apply_record_replay_barrier_patch(bytes, options, operating_point, arch, resource_state,
                                          access_output, semantics, result);
  if (result.errors.empty())
    try_apply_fence_record_patch(bytes, options, operating_point, semantics, arch, result);
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
      !consan_moi_detail::moi_has_runtime_hardware_dispatch_id(point) &&
      !(point.automatic_moi_private_epoch && point.moi_dispatch_identity.private_fallback());
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
  demand.cdna_overflow_strategy = MoiCdnaPersistentOverflowStrategy::ExactWorkgroupState;
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

MoiDispatchIdentityPlan plan_record_replay_dispatch_identity(const ConSanRequest &,
                                                             const MoiDispatchIdentityFacts &) {
  return {
      .needs_dispatch_id = true,
      .permits_private_entry_capture = true,
      .fallback_kind = ConSanMoiFallbackKind::RecordReplayZeroGeneration,
      .fallback_diagnostic =
          "ConSan MOI selected owner-local zero-generation records where the hardware "
          "dispatch-ID pair overlaps guest scalar state",
  };
}

MoiScalarAbiPlan plan_record_replay_scalar_abi(const ConSanRequest &,
                                               const ConSanMoiOperatingPoint &point) {
  std::optional<consan_detail::MoiSpecialStateSgprs> special_state;
  if (point.moi_exec_save_sgpr) {
    const uint16_t base = *point.moi_exec_save_sgpr;
    special_state = consan_detail::MoiSpecialStateSgprs{
        .vcc_save_sgpr = static_cast<uint16_t>(base + 2u),
        .scc_save_sgpr = point.has_compact_moi_scalar_spill() && point.moi_router_jump
                             ? point.moi_router_jump->scc_save_sgpr
                             : static_cast<uint16_t>(base + 4u),
    };
  }
  return make_moi_scalar_abi_plan(point, special_state, 0u, false);
}

uint16_t record_replay_exec_save_sgpr_count(const MoiExecSaveRequirement &requirement,
                                            const MoiExecSaveTargetFacts &) {
  if (requirement.automatic_banked_record_capture)
    return 14u;

  constexpr uint16_t kRuntimeWorkgroupGateSgprCount = 7u;
  const uint16_t runtime_workgroup_gate_count =
      requirement.runtime_sample_stride > 1u ? kRuntimeWorkgroupGateSgprCount : 0u;
  if (requirement.dense_record_barrier_router)
    return std::max<uint16_t>(8u, runtime_workgroup_gate_count);
  if (requirement.scalar_spill)
    return std::max<uint16_t>(4u, runtime_workgroup_gate_count);
  if (requirement.dynamic_stack_spill)
    return std::max<uint16_t>(6u, runtime_workgroup_gate_count);
  return requirement.has_report_buffer ? std::max<uint16_t>(5u, runtime_workgroup_gate_count) : 0u;
}

const MoiModeOperations kRecordReplayModeOperations = {
    .plan = plan_record_replay_object_mode,
    .apply = apply_record_replay_mode_patches,
    .access_scratch_vgpr_count = record_replay_access_scratch_vgpr_count,
    .operational_evidence = {ConSanProbeIntentKind::BarrierRecord,
                             ConSanProbeIntentKind::AtomicRecord, true},
    .dynamic_stack_frame_save_sgpr_offset = 5u,
    .exec_save_sgpr_count = record_replay_exec_save_sgpr_count,
    .prologue = {.backup_compact_spill_for_runtime_sampling = true},
    .persistent_state_demand = plan_record_replay_persistent_state_demand,
    .transient_scalar_placement = {ConSanMoiScalarSpillLayout::Compact, false, true, 0u},
    .dynamic_stack_spill_without_target_backend = false,
    .dynamic_stack_spill_requires_every_owner_dynamic = true,
    .operand_overlap_spill = record_replay_operand_overlap_spill,
    .access_spill_fallback = nullptr,
    .dispatch_identity = plan_record_replay_dispatch_identity,
    .scalar_abi = plan_record_replay_scalar_abi,
    .plan_evidence = plan_record_replay_evidence_requirements,
    .plan_report_layout = plan_record_replay_report_layout,
    .reconstruct_report_inventory = reconstruct_record_replay_report_inventory,
};

#include "rocjitsu/code/patch/consan/consan_moi_record_replay.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_atomic.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_fence.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu

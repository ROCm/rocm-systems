// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sampled access selection and synchronization metadata lowering share one
// engine-private translation unit. The sequencing between these phases is an
// explicit orchestration concern; their representation and route machinery
// remain private to the Sampled component.

#include "rocjitsu/code/patch/consan/consan_moi_sampled.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_apply.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_local_island_allocator.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled_access_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled_atomic_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/consan/consan_vgpr_bank_state.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cstring>
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

using consan_detail::append_moi_workitem_owner_derivation;
using consan_detail::build_moi_relocated_guest_access_words;
using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_detail::MoiBarrierEvidenceSitePlan;
using consan_detail::MoiSpecialStateSgprs;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
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
using consan_moi_detail::append_moi_prepare_scc_preserving_indirect_jump;
using consan_moi_detail::append_moi_report_dispatch_id_pair;
using consan_moi_detail::append_moi_report_dispatch_id_word;
using consan_moi_detail::append_moi_scc_preserving_indirect_jump;
using consan_moi_detail::append_publish_visible_evidence_if_zero;
using consan_moi_detail::append_store_moi_report_dispatch_id_pair;
using consan_moi_detail::append_store_u32_literal;
using consan_moi_detail::append_store_u32_sgpr;
using consan_moi_detail::append_store_u32_vgpr;
using consan_moi_detail::append_store_u32_vgpr_at_offset;
using consan_moi_detail::append_word_bytes;
using consan_moi_detail::append_words_bytes;
using consan_moi_detail::ConSanMoiRecordEmitter;
using consan_moi_detail::count_nop_padding;
using consan_moi_detail::decode_relocatable_entry_instruction;
using consan_moi_detail::DynamicRecordLayout;
using consan_moi_detail::kAccessRecordLayout;
using consan_moi_detail::kAtomicRecordLayout;
using consan_moi_detail::kBarrierRecordLayout;
using consan_moi_detail::kDiagnosticRecordLayout;
using consan_moi_detail::kFenceRecordLayout;
using consan_moi_detail::moi_has_runtime_hardware_dispatch_id;
using consan_moi_detail::note_moi_persistent_vgpr_state;
using consan_moi_detail::resolve_moi_report_layout;

namespace consan_moi_impl {

/// Returns the SCC snapshot that remains valid after a sampled access body.
///
/// Fixed layouts return after the sampled body has overwritten the runtime
/// gate's residue field with the guest SCC snapshot. Spill-backed layouts
/// restore the complete transient window before returning, so their separately
/// allocated indirect-jump SCC register remains the stable snapshot.
std::optional<uint16_t> moi_sampled_access_return_scc_sgpr(const ConSanRequest &request,
                                                           const ConSanMoiOperatingPoint &point) {
  if (request.moi_engine != ConSanMoiEngine::Sampled || !point.moi_exec_save_sgpr)
    return std::nullopt;
  if (point.has_compact_moi_scalar_spill()) {
    const auto indirect = moi_indirect_jump_sgprs(request, point);
    return indirect ? std::optional<uint16_t>(indirect->scc_save_sgpr) : std::nullopt;
  }
  const auto publication = moi_sampled_publication_state_sgprs(request, point);
  return publication ? std::optional<uint16_t>(publication->guest_scc_snapshot_sgpr) : std::nullopt;
}

MoiObjectModePlan plan_sampled_object_mode(const ConSanRequest &request,
                                           const ConSanMoiOperatingPoint &point,
                                           const MoiObjectFacts &facts,
                                           const ConSanObservationPlan &) {
  MoiObjectModePlan plan =
      make_moi_object_mode_plan(request, point, ConSanMoiOwnerSource::WorkitemId);
  plan.reserve_dynamic_stack_prologue_entry = true;
  plan.prologue_requires_consumer = true;
  if (plan.track_atomics && !facts.has_access_candidate) {
    // Sampled atomics publish ordering only into a selected LDS watchpoint's
    // causal window. Without that consumer they do not improve coverage.
    plan.track_atomics = false;
    plan.warnings.emplace_back(
        "ConSan MOI sampled engine skipped atomic ordering in a code object with no selected "
        "LDS access candidates");
  }
  plan.atomic_or_fence_relevant = plan.track_atomics && facts.has_admitted_atomic;
  return plan;
}

void apply_sampled_mode_patches(std::span<const uint8_t> bytes, MoiOptions &options,
                                rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                                std::span<const ConSanMoiCandidate> candidates,
                                const MoiObjectFacts &, ConSanTransformArtifacts &result) {
  try_apply_direct_sampled_watchpoint_patch(bytes, options, arch, resource_state, candidates,
                                            result);
  if (result.errors.empty())
    try_apply_sampled_atomic_sync_patch(bytes, options, arch, resource_state, result);
  if (result.errors.empty())
    try_apply_sampled_barrier_sync_patch(bytes, options, arch, resource_state, candidates, result);
}

uint16_t sampled_access_scratch_vgpr_count(const ConSanRequest &request,
                                           const BoundRuntimeResources &,
                                           const ConSanMoiOperatingPoint &point,
                                           const ConSanMoiCandidate &candidate,
                                           rj_code_arch_t arch) {
  return direct_sampled_scratch_count(request, point, candidate, arch);
}

MoiPersistentStateDemand plan_sampled_persistent_state_demand(
    const ConSanRequest &request, const BoundRuntimeResources &resources,
    const ConSanMoiOperatingPoint &point, const MoiPersistentStateFacts &facts) {
  MoiPersistentStateDemand demand =
      make_exact_workgroup_capture_demand(request, resources, point, facts);
  // A synchronization-aware Sampled probe must preserve one owner identity
  // from kernel entry through both access and sync sites. Access-only Sampled
  // objects retain the cheaper private-state choice.
  demand.needs_persistent_state = moi_initializes_owner_epoch(request, point) ||
                                  demand.needs_entry_workgroup_tuple || request.moi_track_atomics ||
                                  request.moi_track_barriers ||
                                  request.moi_runtime_sample_stride > 1u;
  demand.synchronization_requires_persistent_owner = facts.atomic_count || facts.barrier_count;
  demand.scalar_state_supported = true;
  demand.private_state_supported = request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId;
  demand.scalar_state_required_for_private_or_overflow = facts.has_operational_dynamic_stack_owner;
  demand.prefer_private_epoch_for_descriptor_growth = true;
  demand.cdna_overflow_strategy = MoiCdnaPersistentOverflowStrategy::OwnerSnapshot;
  return demand;
}

MoiOperandOverlapSpillPolicy
sampled_operand_overlap_spill(const MoiOperandOverlapSpillContext &context) {
  if (!consan_is_capability_arch(context.arch))
    return {};
  if (context.site_kind == ConSanResourceSiteKind::Atomic) {
    MoiOperandOverlapSpillPolicy policy;
    policy.supported = true;
    return policy;
  }
  if (context.site_kind != ConSanResourceSiteKind::Access || context.access_candidate == nullptr ||
      context.guest_replay_requires_disjoint_address_scratch) {
    return {};
  }
  const ConSanMoiCandidate &candidate = *context.access_candidate;
  const bool spill_backed_recovery = sampled_access_supports_spill_backed_operand_recovery(
      context.request, candidate, context.arch);
  MoiOperandOverlapSpillPolicy policy;
  policy.supported = sampled_access_can_plan_spill_over_guest_operands(context.request,
                                                                       context.point, candidate) ||
                     spill_backed_recovery;
  if (spill_backed_recovery && candidate.lowering.form &&
      candidate.lowering.form->destination_vgpr) {
    policy.protected_vgpr = candidate.lowering.form->destination_vgpr;
    policy.protected_vgpr_count = static_cast<uint8_t>(candidate_payload_vgpr_count(candidate));
  }
  return policy;
}

std::optional<uint16_t>
sampled_access_spill_fallback(const MoiAccessSpillFallbackContext &context) {
  if (!sampled_access_supports_spill_backed_operand_recovery(context.request, context.candidate,
                                                             context.arch) ||
      (!context.no_ordinary_window && !context.initial_spill_overlaps_guest)) {
    return std::nullopt;
  }
  return sampled_spill_backed_scratch_count(context.request, context.point, context.candidate,
                                            context.arch);
}

MoiDispatchIdentityPlan plan_sampled_dispatch_identity(const ConSanRequest &request,
                                                       const MoiDispatchIdentityFacts &facts) {
  return {
      .needs_dispatch_id =
          request.moi_runtime_sample_stride > 1u && !facts.target_uses_gfx12_cdna_execution,
      .fallback_kind = ConSanMoiFallbackKind::SampledLiteralDispatchId,
      .fallback_replans_dispatch_only = true,
      .fallback_diagnostic =
          "ConSan MOI selected owner-local literal dispatch IDs where the hardware pair "
          "overlaps guest scalar state",
  };
}

MoiScalarAbiPlan plan_sampled_scalar_abi(const ConSanRequest &request,
                                         const ConSanMoiOperatingPoint &point) {
  const auto publication = moi_sampled_publication_state_sgprs(request, point);
  const std::optional<consan_detail::MoiSpecialStateSgprs> special_state =
      publication
          ? std::optional{consan_detail::MoiSpecialStateSgprs{
                .vcc_save_sgpr = publication->selection_vcc_save_sgpr,
                .scc_save_sgpr = point.has_compact_moi_scalar_spill() && point.moi_router_jump
                                     ? point.moi_router_jump->scc_save_sgpr
                                     : publication->publication_exec_save_sgpr,
            }}
          : std::nullopt;
  return make_moi_scalar_abi_plan(point, special_state, 0u, false);
}

const MoiModeOperations kSampledModeOperations = {
    plan_sampled_object_mode,
    apply_sampled_mode_patches,
    sampled_access_scratch_vgpr_count,
    plan_sampled_persistent_state_demand,
    {ConSanMoiScalarSpillLayout::Compact, true, false, 8u},
    false,
    true,
    sampled_operand_overlap_spill,
    sampled_access_spill_fallback,
    plan_sampled_dispatch_identity,
    plan_sampled_scalar_abi,
    plan_sampled_evidence_requirements,
    plan_sampled_report_layout,
    reconstruct_sampled_report_inventory,
};

#include "rocjitsu/code/patch/consan/consan_moi_sampled_access.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_sync.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu

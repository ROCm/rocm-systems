// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ConSan access selection and synchronization metadata lowering share one
// mode-private translation unit. The sequencing between these phases is an
// explicit orchestration concern; their representation and route machinery
// remain private to the ConSan component.

#include "rocjitsu/code/patch/consan/consan_probe_lowering.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/kernel_scope.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_access_apply.h"
#include "rocjitsu/code/patch/consan/consan_access_emission.h"
#include "rocjitsu/code/patch/consan/consan_access_target.h"
#include "rocjitsu/code/patch/consan/consan_atomic_emission.h"
#include "rocjitsu/code/patch/consan/consan_contracts.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_device_primitives.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_identity_contracts.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_probe_planning.h"
#include "rocjitsu/code/patch/consan/consan_prologue.h"
#include "rocjitsu/code/patch/consan/consan_relocation.h"
#include "rocjitsu/code/patch/consan/consan_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_report_planning.h"
#include "rocjitsu/code/patch/consan/consan_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_sync_emission.h"
#include "rocjitsu/code/patch/consan/consan_text_relocation.h"
#include "rocjitsu/code/patch/consan/consan_window_emission.h"
#include "rocjitsu/code/patch/consan/targets/consan_vgpr_bank_state.h"
#include "rocjitsu/code/patch/consan/targets/consan_wave_sched_state.h"
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

namespace rocjitsu::consan {

using detail::append_atomic_fetch_add_one_u32;
using detail::append_atomic_load_u32;
using detail::append_compare_report_dispatch_id_word;
using detail::append_load_u32_vgpr_at_offset;
using detail::append_report_dispatch_id_pair;
using detail::append_report_dispatch_id_word;
using detail::append_select_first_lane_in_exec_mask;
using detail::append_store_report_dispatch_id_pair;
using detail::append_store_u32_literal;
using detail::append_store_u32_sgpr;
using detail::append_store_u32_vgpr;
using detail::append_store_u32_vgpr_at_offset;
using detail::append_word_bytes;
using detail::append_words_bytes;
using detail::append_workitem_owner_derivation;
using detail::AtomicEvidenceSitePlan;
using detail::BarrierEvidenceSitePlan;
using detail::decode_relocatable_entry_instruction;
using detail::has_runtime_hardware_dispatch_id;
using detail::range_overlaps;
using detail::RecordEmitter;
using detail::reject_optional_scratch_range_overlap;
using detail::resolve_report_layout;
using detail::SpecialStateSgprs;
using detail::WorkitemOwnerDerivationPlan;

namespace detail {

bool append_private_owner_epoch_load(std::vector<uint32_t> &words, std::span<const uint8_t> bytes,
                                     uint64_t descriptor_file_offset, bool automatic_private_epoch,
                                     const OwnerEpochVgprSources &owner_epoch_vgprs,
                                     const PrivateStateLayout &layout, rj_code_arch_t arch,
                                     std::vector<std::string> &errors) {
  if (!automatic_private_epoch || !owner_epoch_vgprs.owner || !owner_epoch_vgprs.epoch ||
      !layout.owner_offset) {
    errors.emplace_back("ConSan sync has an invalid private owner/epoch plan");
    return false;
  }
  const auto owner_shift = descriptor_owner_shift(bytes, descriptor_file_offset, arch, errors);
  const auto load_epoch =
      instrumentation::build_private_load_b32(*owner_epoch_vgprs.epoch, layout.epoch_offset, arch);
  const auto load_owner =
      instrumentation::build_private_load_b32(*owner_epoch_vgprs.owner, *layout.owner_offset, arch);
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  const auto owner =
      owner_shift ? instrumentation::build_v_lshrrev_b32(*owner_epoch_vgprs.owner,
                                                         scalar_positive_inline_u32(*owner_shift),
                                                         *owner_epoch_vgprs.owner, arch)
                  : std::nullopt;
  if (!owner_shift || !load_epoch || !load_owner || !wait || !owner) {
    errors.emplace_back("ConSan sync could not load private owner/epoch state");
    return false;
  }
  words.insert(words.end(), load_epoch->begin(), load_epoch->end());
  words.insert(words.end(), load_owner->begin(), load_owner->end());
  words.push_back(*wait);
  words.push_back(*owner);
  return true;
}

static std::optional<StaticAccessMappings>
make_access_runtime_mapping(StaticAccessAttribution access, LdsAccessKind access_kind,
                            uint32_t first_slot, uint32_t range_count, uint32_t bank_count,
                            const PatchLoweringProduct &patch) {
  if (range_count == 0u || bank_count == 0u)
    return std::nullopt;
  return StaticAccessMappings{{
      .access = std::move(access),
      .access_kind = access_kind,
      .first_slot = first_slot,
      .range_count = range_count,
      .bank_count = bank_count,
      .emitted_probe_text_offset = patch.trampoline_offset,
      .relocated_guest_text_offset = patch.relocated_guest_instruction_offset,
      .scratch_vgpr = patch.scratch_vgpr,
  }};
}

[[nodiscard]] StaticAccessMappings
runtime_mapping_including_staged(const TransformArtifacts &result) {
  StaticAccessMappings mapping = result.coverage_ledger.runtime_static_mapping();
  for (const TextFragment &fragment : result.staged_text_fragments) {
    mapping.insert(mapping.end(), fragment.runtime_mapping.begin(), fragment.runtime_mapping.end());
  }
  return mapping;
}

ObjectModePlan plan_object_mode(const Request &request, const BoundRuntimeResources &resources,
                                const TransformPolicy &policy, const ObjectFacts &facts) {
  ObjectModePlan plan;
  plan.owner_source = request.owner_source == OwnerSource::Automatic ? OwnerSource::WorkitemId
                                                                     : request.owner_source;
  plan.track_atomics = request.track_atomics;
  plan.semantics.report_layout = resolve_report_layout(
      resources, direct_report_buffer_layout_for_bytes(resources.report_buffer_size));
  if (plan.track_atomics && !facts.has_access_candidate) {
    // ConSan atomics publish ordering only into a selected LDS watchpoint's
    // causal window. Without that consumer they do not improve coverage.
    plan.track_atomics = false;
    plan.warning = "ConSan mode skipped atomic ordering in a code object with no selected "
                   "LDS access candidates";
  }
  if (plan.track_atomics) {
    const uint32_t patch_budget = policy.max_patches_is_expert_limit
                                      ? policy.max_patches
                                      : std::numeric_limits<uint32_t>::max();
    plan.semantics.reserved_atomic_patch_count = static_cast<uint32_t>(
        std::min({static_cast<size_t>(patch_budget),
                  static_cast<size_t>(plan.semantics.report_layout.watchpoint_capacity),
                  facts.admitted_atomic_count}));
  }
  return plan;
}

void apply_mode_patches(std::span<const uint8_t> bytes, const Options &options,
                        OperatingPoint &operating_point, rj_code_arch_t arch,
                        ResourcePlanningState &resource_state,
                        std::span<const Candidate> candidates,
                        const ObjectModeSemantics &mode_semantics, TransformArtifacts &result) {
  try_apply_direct_watchpoint_patch(bytes, options, operating_point, arch, candidates,
                                    mode_semantics, result);
  if (result.errors.empty())
    try_apply_atomic_sync_patch(bytes, options, operating_point, arch, resource_state,
                                mode_semantics, result);
  if (result.errors.empty())
    try_apply_barrier_sync_patch(bytes, options, operating_point, arch, resource_state, candidates,
                                 mode_semantics, result);
}

uint16_t barrier_scratch_vgpr_count(const BarrierScratchFacts &facts) {
  return facts.automatic_private_epoch || facts.persistent_scalar_state_complete ? 9u : 7u;
}

PersistentStateDemand plan_persistent_state_demand(const Request &request,
                                                   const OperatingPoint &point,
                                                   const PersistentStateFacts &facts) {
  PersistentStateDemand demand;
  demand.needs_entry_workgroup_tuple =
      (facts.access_count || facts.atomic_count || facts.barrier_count) &&
      !detail::has_exact_entry_workgroup_capture(point);
  demand.private_workgroup_tuple_supported = demand.needs_entry_workgroup_tuple;
  // A synchronization-aware ConSan probe must preserve one owner identity
  // from kernel entry through both access and sync sites. Access-only ConSan
  // objects retain the cheaper private-state choice.
  demand.needs_persistent_state =
      point.initialize_owner_epoch || demand.needs_entry_workgroup_tuple || request.track_atomics ||
      request.track_barriers ||
      (request.runtime_sample_stride > 1u || request.cell_selector().stride > 1u);
  demand.synchronization_requires_persistent_owner = facts.atomic_count || facts.barrier_count;
  demand.private_state_supported = request.owner_source == OwnerSource::WorkitemId;
  demand.scalar_state_required_for_private_or_overflow = facts.has_operational_dynamic_stack_owner;
  return demand;
}

OperandOverlapSpillPolicy operand_overlap_spill(const OperandOverlapSpillContext &context) {
  if (!context.resource_facts.target_available)
    return {};
  if (context.site_kind == ResourceSiteKind::Atomic) {
    OperandOverlapSpillPolicy policy;
    policy.supported = true;
    return policy;
  }
  if (context.site_kind != ResourceSiteKind::Access || context.access_candidate == nullptr ||
      context.resource_facts.guest_replay_requires_disjoint_address_scratch) {
    return {};
  }
  const Candidate &candidate = *context.access_candidate;
  const bool spill_backed_recovery = context.resource_facts.supports_native_lds_spill_recovery;
  OperandOverlapSpillPolicy policy;
  policy.supported = access_can_plan_spill_over_guest_operands(context.request,
                                                               context.resource_facts, candidate) ||
                     spill_backed_recovery;
  if (spill_backed_recovery && candidate.site().lowering.form &&
      candidate.site().lowering.form->destination_vgpr) {
    policy.protected_vgpr = candidate.site().lowering.form->destination_vgpr;
    policy.protected_vgpr_count = static_cast<uint8_t>(candidate_payload_vgpr_count(candidate));
  }
  return policy;
}

std::optional<uint16_t> access_spill_fallback(const AccessSpillFallbackContext &context) {
  if (!context.resource_facts.supports_native_lds_spill_recovery ||
      (!context.no_ordinary_window && !context.initial_spill_overlaps_guest)) {
    return std::nullopt;
  }
  return direct_scratch_count(context.request, context.resource_facts);
}

bool requires_dispatch_identity(const Request &request, const DispatchIdentityFacts &facts) {
  return facts.access_reports_need_explicit_identity &&
         (request.runtime_sample_stride > 1u || facts.has_access_or_atomic_consumer);
}

ScalarAbiPlan plan_scalar_abi(const ScalarPreservationState &preservation_state) {
  const auto publication = publication_state_sgprs(preservation_state.exec_save_sgpr);
  const std::optional<detail::SpecialStateSgprs> special_state =
      publication
          ? std::optional{detail::SpecialStateSgprs{
                .vcc_save_sgpr = publication->selection_vcc_save_sgpr,
                .scc_save_sgpr =
                    preservation_state.has_compact_spill() && preservation_state.scalar_spill_setup
                        ? preservation_state.scalar_spill_setup->temporaries.scc_save_sgpr
                        : publication->publication_exec_save_sgpr,
            }}
          : std::nullopt;
  return {.special_state = special_state};
}

#include "rocjitsu/code/patch/consan/consan_access.inc"

#include "rocjitsu/code/patch/consan/consan_sync.inc"

} // namespace detail
} // namespace rocjitsu::consan

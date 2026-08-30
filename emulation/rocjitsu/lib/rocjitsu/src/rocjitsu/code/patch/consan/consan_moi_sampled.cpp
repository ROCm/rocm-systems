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
using consan_moi_detail::resolve_moi_report_layout;

namespace consan_moi_impl {

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

const MoiModeOperations kSampledModeOperations = {
    plan_sampled_object_mode,
    apply_sampled_mode_patches,
    sampled_access_scratch_vgpr_count,
};

#include "rocjitsu/code/patch/consan/consan_moi_sampled_access.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_sync.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu

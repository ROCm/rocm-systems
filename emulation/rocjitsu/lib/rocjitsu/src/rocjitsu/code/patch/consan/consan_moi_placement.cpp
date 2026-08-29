// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include "rocjitsu/code/patch/consan/consan_moi_access_apply.h"

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
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"
#include "rocjitsu/code/patch/consan/consan_moi_candidate_projection.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.h"
#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_local_island_allocator.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_pipeline.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_replay.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled_access_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled_atomic_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
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
using consan_moi_detail::plan_prebuilt_appended_cave;
using consan_moi_detail::record_replay_entry_workgroup_capture_is_unambiguous;
using consan_moi_detail::record_replay_has_entry_workgroup_capture;
using consan_moi_detail::record_replay_requires_entry_workgroup_capture;
using consan_moi_detail::record_replay_uses_automatic_banked_capture;
using consan_moi_detail::resolve_moi_report_layout;

namespace consan_moi_impl {

#include "rocjitsu/code/patch/consan/consan_moi_placement.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu

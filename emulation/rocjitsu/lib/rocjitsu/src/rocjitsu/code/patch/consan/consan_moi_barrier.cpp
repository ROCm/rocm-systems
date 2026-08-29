// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Barrier lowering is a named synchronization component shared by the three
// MOI engines. It consumes engine-selected evidence intents and retains the
// exact fallback invariant that a Record/Replay route may reuse the common
// dense epoch mechanism without changing evidence meaning.

#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_apply.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_local_island_allocator.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
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
using consan_detail::MoiBarrierEvidenceSitePlan;
using consan_detail::MoiSpecialStateSgprs;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_moi_detail::append_moi_scc_preserving_indirect_jump;
using consan_moi_detail::append_publish_visible_evidence_if_zero;
using consan_moi_detail::append_word_bytes;
using consan_moi_detail::append_words_bytes;
using consan_moi_detail::count_nop_padding;
using consan_moi_detail::decode_relocatable_entry_instruction;
using consan_moi_detail::plan_prebuilt_appended_cave;
using consan_moi_detail::resolve_moi_report_layout;

namespace consan_moi_impl {

#include "rocjitsu/code/patch/consan/consan_moi_barrier.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu

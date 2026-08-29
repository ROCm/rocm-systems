// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Record/Replay owns event selection and evidence semantics. Shared resource,
// placement, descriptor, and target emission mechanics enter through declared
// contracts rather than through the MOI textual include order.

#include "rocjitsu/code/patch/consan/consan_moi_record_replay.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_local_island_allocator.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/trampoline_builder.h"

#include <algorithm>
#include <cstring>
#include <ranges>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {

using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_moi_detail::append_words_bytes;
using consan_moi_detail::resolve_moi_report_layout;

namespace consan_moi_impl {

#include "rocjitsu/code/patch/consan/consan_moi_record_atomic.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu

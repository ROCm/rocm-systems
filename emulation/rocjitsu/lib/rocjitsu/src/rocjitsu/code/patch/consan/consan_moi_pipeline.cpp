// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_pipeline.h"

#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"
#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled_atomic_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {

using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_detail::MoiBarrierEvidenceSitePlan;
using consan_detail::MoiFenceEvidenceSitePlan;

namespace consan_moi_impl {

#include "rocjitsu/code/patch/consan/consan_moi_pipeline.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu

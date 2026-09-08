// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Common barrier analysis projects admitted evidence into a mode-neutral
// lowering contract. Each mode owns the transaction that consumes it.

#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"

#include <algorithm>
#include <ranges>

namespace rocjitsu {

using consan_detail::MoiBarrierEvidenceSitePlan;

namespace consan_moi_impl {

#include "rocjitsu/code/patch/consan/consan_moi_barrier.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu

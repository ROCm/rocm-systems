// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Barrier analysis projects admitted evidence into the lowering contract
// consumed by ConSan synchronization instrumentation.

#include "rocjitsu/code/patch/consan/consan_barrier.h"

#include <algorithm>
#include <ranges>

namespace rocjitsu::consan {

using detail::BarrierEvidenceSitePlan;

namespace detail {

#include "rocjitsu/code/patch/consan/consan_barrier.inc"

} // namespace detail
} // namespace rocjitsu::consan

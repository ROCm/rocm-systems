// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// Constrains data collection to configurable time windows.
/// Each window is defined by a delay before collection begins, a duration
/// for how long it runs, and an optional repeat count.
///
/// The clock governing all windows is set once via
/// ROCPROFSYS_TRACE_PERIOD_CLOCK_ID:
///   "realtime" (default) — wall-clock time (std::chrono::steady_clock)
///   "cputime"            — process CPU time (CLOCK_PROCESS_CPUTIME_ID)
///
/// @todo Migrate delay/duration for process sampling and causal profiling
///       to this model (sampling delay/duration already wired; causal deferred).

#include "common/defines.h"

#include <cstdint>
#include <ctime>
#include <vector>

namespace rocprofsys
{
namespace constraint
{
struct spec
{
    double        delay    = 0.0;
    double        duration = 0.0;
    std::uint64_t repeat   = 1;
};

std::vector<spec>
get_trace_specs();

/// Returns CLOCK_PROCESS_CPUTIME_ID when ROCPROFSYS_TRACE_PERIOD_CLOCK_ID is
/// "cputime"; returns CLOCK_REALTIME (routed to clocks::steady) otherwise.
[[nodiscard]] clockid_t
get_trace_period_clock_id();

/// True iff the first configured trace window starts with a delay > 0.
[[nodiscard]] bool
trace_has_initial_delay();
}  // namespace constraint
}  // namespace rocprofsys

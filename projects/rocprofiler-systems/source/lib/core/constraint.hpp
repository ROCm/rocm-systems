// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// Constrains data collection to configurable time windows.
/// Each window is defined by a delay before collection begins and a
/// duration for how long it runs. `spec::repeat` is parsed from
/// ROCPROFSYS_TRACE_PERIODS but not yet consumed - only the first
/// configured window is currently wired into a running time_window trigger;
/// repeating/multi-window scheduling is not yet implemented.
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
#include <string_view>
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

[[nodiscard]] spec
parse_trace_period_entry(std::string_view entry, double default_delay,
                         double default_duration);

[[nodiscard]] std::vector<spec>
parse_trace_periods(std::string_view periods, double default_delay,
                    double default_duration);

[[nodiscard]] clockid_t
parse_trace_period_clock_id(std::string_view clock_id_str);

std::vector<spec>
get_trace_specs();

[[nodiscard]] clockid_t
get_trace_period_clock_id();
}  // namespace constraint
}  // namespace rocprofsys

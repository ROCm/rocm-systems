// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// Constrains data collection to configurable time windows.
/// Each window is defined by a delay before collection begins and a
/// duration for how long it runs. `trace_period::repeat` is parsed from
/// ROCPROFSYS_TRACE_PERIODS but not yet consumed - only the first
/// configured window is currently wired into a running time_window trigger;
/// repeating/multi-window scheduling is not yet implemented.
///
/// The clock governing all windows is set once via
/// ROCPROFSYS_TRACE_PERIOD_CLOCK_ID:
///   "realtime" (default) — wall-clock time (std::chrono::steady_clock)
///   "cputime"            — process CPU time (CLOCK_PROCESS_CPUTIME_ID)
///
/// Configuration is read through an Externals policy so the translation from
/// raw settings to trace_periods can be exercised without the global config
/// singleton; see constraint_deps.hpp for the production policy.
///
/// @todo Migrate delay/duration for process sampling and causal profiling
///       to this model (sampling delay/duration already wired; causal deferred).

#include "common/defines.h"
#include "common/delimit.hpp"
#include "common/env_vars.hpp"
#include "utility.hpp"

#include <spdlog/fmt/fmt.h>

#include <concepts>
#include <cstdint>
#include <ctime>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace constraint
{
struct trace_period
{
    double        delay    = 0.0;
    double        duration = 0.0;
    std::uint64_t repeat   = 1;
};

namespace concepts
{
template <typename Externals>
concept trace_config_externals = requires {
    { Externals::get_trace_delay() } -> std::convertible_to<double>;
    { Externals::get_trace_duration() } -> std::convertible_to<double>;
    { Externals::get_trace_periods() } -> std::convertible_to<std::string>;
    { Externals::get_trace_period_clock() } -> std::convertible_to<std::string>;
};
}  // namespace concepts

/// Translates the trace-window settings supplied by @p Externals into the
/// value types the control plane consumes.
template <typename Externals>
    requires concepts::trace_config_externals<Externals>
class trace_config
{
public:
    [[nodiscard]] static std::vector<trace_period> get_trace_specs();

    [[nodiscard]] static clockid_t get_trace_period_clock_id();

private:
    [[nodiscard]] static trace_period parse_period_entry(std::string_view entry,
                                                         double           default_delay,
                                                         double default_duration);

    [[nodiscard]] static std::vector<trace_period> parse_periods(std::string_view periods,
                                                                 double default_delay,
                                                                 double default_duration);

    [[nodiscard]] static clockid_t parse_period_clock_id(std::string_view clock_id_str);
};

template <typename Externals>
    requires concepts::trace_config_externals<Externals>
trace_period
trace_config<Externals>::parse_period_entry(std::string_view entry, double default_delay,
                                            double default_duration)
{
    const auto   _parts = rocprofsys::delimit(std::string{ entry }, ":");
    trace_period _s{ default_delay, default_duration, 1 };
    if(!_parts.empty()) _s.delay = utility::convert<double>(_parts.at(0));
    if(_parts.size() > 1) _s.duration = utility::convert<double>(_parts.at(1));
    if(_parts.size() > 2) _s.repeat = utility::convert<std::uint64_t>(_parts.at(2));
    return _s;
}

template <typename Externals>
    requires concepts::trace_config_externals<Externals>
std::vector<trace_period>
trace_config<Externals>::parse_periods(std::string_view periods, double default_delay,
                                       double default_duration)
{
    auto _v = std::vector<trace_period>{};
    for(const auto& _entry : rocprofsys::delimit(std::string{ periods }, " ;\t\n"))
        _v.push_back(parse_period_entry(_entry, default_delay, default_duration));
    return _v;
}

template <typename Externals>
    requires concepts::trace_config_externals<Externals>
clockid_t
trace_config<Externals>::parse_period_clock_id(std::string_view clock_id_str)
{
    if(clock_id_str == "cputime") return CLOCK_PROCESS_CPUTIME_ID;
    if(clock_id_str == "realtime") return CLOCK_REALTIME;
    throw std::runtime_error(
        fmt::format("Unknown {}: '{}'. Valid choices: realtime, cputime",
                    env_vars::TRACE_PERIOD_CLOCK_ID, clock_id_str));
}

template <typename Externals>
    requires concepts::trace_config_externals<Externals>
std::vector<trace_period>
trace_config<Externals>::get_trace_specs()
{
    auto _v = std::vector<trace_period>{};

    const auto _delay_v    = Externals::get_trace_delay();
    const auto _duration_v = Externals::get_trace_duration();

    if(_delay_v > 0.0 || _duration_v > 0.0)
        _v.push_back(trace_period{ _delay_v, _duration_v, 1 });

    // Clock for all TRACE_PERIODS entries is set via
    // ROCPROFSYS_TRACE_PERIOD_CLOCK_ID.
    const auto _periods_v = Externals::get_trace_periods();
    if(!_periods_v.empty())
    {
        const auto _periods = parse_periods(_periods_v, _delay_v, _duration_v);
        _v.insert(_v.end(), _periods.begin(), _periods.end());
    }

    return _v;
}

template <typename Externals>
    requires concepts::trace_config_externals<Externals>
clockid_t
trace_config<Externals>::get_trace_period_clock_id()
{
    return parse_period_clock_id(Externals::get_trace_period_clock());
}
}  // namespace constraint
}  // namespace rocprofsys

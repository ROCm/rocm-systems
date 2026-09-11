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
/// ROCPROFSYS_TRACE_DELAY/ROCPROFSYS_TRACE_DURATION and ROCPROFSYS_TRACE_PERIODS
/// are mutually exclusive: get_trace_specs() throws if both are configured, and
/// also throws if either delay or duration is negative.
///
/// The clock governing all windows is set once via
/// ROCPROFSYS_TRACE_PERIOD_CLOCK_ID:
///   "realtime" (default) — wall-clock time (std::chrono::steady_clock)
///   "cputime"            — process CPU time (CLOCK_PROCESS_CPUTIME_ID)
///
/// Configuration is read through an Externals policy - a single
/// get_trace_period_settings() call - so the translation from raw settings to
/// trace_periods can be exercised without the global config singleton;
/// default_trace_config_externals below is the production policy.

#include "common/delimit.hpp"
#include "common/env_vars.hpp"
#include "core/config.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/fmt.h>  // NOLINT(misc-include-cleaner)

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <ctime>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::inline config
{
struct trace_period
{
    double        delay    = 0.0;
    double        duration = 0.0;
    std::uint64_t repeat   = 1;
};

/// Raw trace-window settings as read from configuration, before parsing
/// ROCPROFSYS_TRACE_PERIODS into trace_period entries.
struct trace_period_settings
{
    double      delay    = 0.0;
    double      duration = 0.0;
    std::string periods;
    std::string period_clock = "realtime";
};

template <typename Externals>
concept trace_config_externals = requires {
    {
        Externals::get_trace_period_settings()
    } -> std::convertible_to<trace_period_settings>;
};

/// Production Externals policy: reads settings directly. Tests inject a mock instead.
struct default_trace_config_externals
{
    static trace_period_settings get_trace_period_settings()
    {
        return trace_period_settings{
            .delay = get_setting_value<double>(std::string{ env_vars::TRACE_DELAY })
                         .value_or(0.0),
            .duration = get_setting_value<double>(std::string{ env_vars::TRACE_DURATION })
                            .value_or(0.0),
            .periods =
                get_setting_value<std::string>(std::string{ env_vars::TRACE_PERIODS })
                    .value_or(std::string{}),
            .period_clock = get_setting_value<std::string>(
                                std::string{ env_vars::TRACE_PERIOD_CLOCK_ID })
                                .value_or(std::string{ "realtime" })
        };
    }
};

/// Translates the trace-window settings supplied by @p Externals into the
/// value types the control plane consumes.
template <typename Externals>
    requires trace_config_externals<Externals>
class trace_config
{
public:
    [[nodiscard]] static std::vector<trace_period> get_trace_specs();

    // NOLINTNEXTLINE(misc-include-cleaner)
    [[nodiscard]] static clockid_t get_trace_period_clock_id();

private:
    [[nodiscard]] static trace_period parse_period_entry(std::string_view entry);

    [[nodiscard]] static std::vector<trace_period> parse_periods(
        std::string_view periods);
};

template <typename Externals>
    requires trace_config_externals<Externals>
trace_period
trace_config<Externals>::parse_period_entry(std::string_view entry)
{
    const auto   fields = rocprofsys::delimit(std::string{ entry }, ":");
    trace_period period{};

    if(fields.size() > 3)
    {
        LOG_WARNING("{} entry '{}' has {} colon-separated fields; only the first 3 "
                    "(delay:duration:repeat) are used, the rest are ignored",
                    env_vars::TRACE_PERIODS, entry, fields.size());
    }

    switch(std::min<std::size_t>(fields.size(), 3))
    {
        case 3:
            period.repeat = utility::convert<std::uint64_t>(fields[2]);
            [[fallthrough]];

        case 2: period.duration = utility::convert<double>(fields[1]); [[fallthrough]];

        case 1: period.delay = utility::convert<double>(fields[0]); [[fallthrough]];

        case 0:
        default: break;
    }
    return period;
}

template <typename Externals>
    requires trace_config_externals<Externals>
std::vector<trace_period>
trace_config<Externals>::parse_periods(std::string_view periods)
{
    const auto entries = rocprofsys::delimit(std::string{ periods }, " ;\t\n");

    auto parsed_periods = std::vector<trace_period>{};
    parsed_periods.reserve(entries.size());
    std::transform(entries.begin(), entries.end(), std::back_inserter(parsed_periods),
                   [](const auto& entry) { return parse_period_entry(entry); });
    return parsed_periods;
}

template <typename Externals>
    requires trace_config_externals<Externals>
std::vector<trace_period>
trace_config<Externals>::get_trace_specs()
{
    const auto trace_control_settings = Externals::get_trace_period_settings();

    if(trace_control_settings.delay < 0.0 || trace_control_settings.duration < 0.0)
    {
        throw std::runtime_error(
            fmt::format("{}/{} must not be negative (delay={}, duration={})",
                        env_vars::TRACE_DELAY, env_vars::TRACE_DURATION,
                        trace_control_settings.delay, trace_control_settings.duration));
    }

    const bool has_delay_duration =
        trace_control_settings.delay > 0.0 || trace_control_settings.duration > 0.0;
    const bool has_periods = !trace_control_settings.periods.empty();

    if(has_delay_duration && has_periods)
    {
        // NOLINTNEXTLINE(misc-include-cleaner)
        throw std::runtime_error(fmt::format(
            "{}/{} and {} are mutually exclusive; configure one or the other, not both",
            env_vars::TRACE_DELAY, env_vars::TRACE_DURATION, env_vars::TRACE_PERIODS));
    }

    if(has_periods)
    {
        return parse_periods(trace_control_settings.periods);
    }
    if(has_delay_duration)
    {
        return { trace_period{ trace_control_settings.delay,
                               trace_control_settings.duration, 1 } };
    }
    return {};
}

template <typename Externals>
    requires trace_config_externals<Externals>
clockid_t
trace_config<Externals>::get_trace_period_clock_id()
{
    const auto clock_id_str = Externals::get_trace_period_settings().period_clock;

    clockid_t result;
    if(clock_id_str == "cputime")
    {
        result = CLOCK_PROCESS_CPUTIME_ID;  // NOLINT(misc-include-cleaner)
    }
    else if(clock_id_str == "realtime")
    {
        result = CLOCK_REALTIME;  // NOLINT(misc-include-cleaner)
    }
    else
    {
        throw std::runtime_error(
            fmt::format("Unknown {}: '{}'. Valid choices: realtime, cputime",
                        env_vars::TRACE_PERIOD_CLOCK_ID, clock_id_str));
    }
    return result;
}
}  // namespace rocprofsys::inline config

// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "constraint.hpp"
#include "common/delimit.hpp"
#include "common/env_vars.hpp"
#include "config.hpp"
#include "utility.hpp"

#include <spdlog/fmt/fmt.h>

#include <cstdint>
#include <ctime>
#include <stdexcept>
#include <string>

namespace rocprofsys
{
namespace constraint
{
spec
parse_trace_period_entry(std::string_view entry, double default_delay,
                         double default_duration)
{
    const auto _parts = rocprofsys::delimit(std::string{ entry }, ":");
    spec       _s{ default_delay, default_duration, 1 };
    if(!_parts.empty()) _s.delay = utility::convert<double>(_parts.at(0));
    if(_parts.size() > 1) _s.duration = utility::convert<double>(_parts.at(1));
    if(_parts.size() > 2) _s.repeat = utility::convert<std::uint64_t>(_parts.at(2));
    return _s;
}

std::vector<spec>
parse_trace_periods(std::string_view periods, double default_delay,
                    double default_duration)
{
    auto _v = std::vector<spec>{};
    for(const auto& _entry : rocprofsys::delimit(std::string{ periods }, " ;\t\n"))
        _v.push_back(parse_trace_period_entry(_entry, default_delay, default_duration));
    return _v;
}

clockid_t
parse_trace_period_clock_id(std::string_view clock_id_str)
{
    if(clock_id_str == "cputime") return CLOCK_PROCESS_CPUTIME_ID;
    if(clock_id_str == "realtime") return CLOCK_REALTIME;
    throw std::runtime_error(
        fmt::format("Unknown {}: '{}'. Valid choices: realtime, cputime",
                    env_vars::TRACE_PERIOD_CLOCK_ID, clock_id_str));
}

std::vector<spec>
get_trace_specs()
{
    auto _v = std::vector<constraint::spec>{};

    const auto _delay_v =
        config::get_setting_value<double>(std::string{ env_vars::TRACE_DELAY })
            .value_or(0.0);
    const auto _duration_v =
        config::get_setting_value<double>(std::string{ env_vars::TRACE_DURATION })
            .value_or(0.0);

    if(_delay_v > 0.0 || _duration_v > 0.0)
        _v.push_back(spec{ _delay_v, _duration_v, 1 });

    // Clock for all TRACE_PERIODS entries is set via
    // ROCPROFSYS_TRACE_PERIOD_CLOCK_ID.
    const auto _periods_v =
        config::get_setting_value<std::string>(std::string{ env_vars::TRACE_PERIODS })
            .value_or("");
    if(!_periods_v.empty())
    {
        auto _periods = parse_trace_periods(_periods_v, _delay_v, _duration_v);
        _v.insert(_v.end(), std::make_move_iterator(_periods.begin()),
                  std::make_move_iterator(_periods.end()));
    }

    return _v;
}

clockid_t
get_trace_period_clock_id()
{
    const auto _str = config::get_setting_value<std::string>(
                          std::string{ env_vars::TRACE_PERIOD_CLOCK_ID })
                          .value_or("realtime");
    return parse_trace_period_clock_id(_str);
}
}  // namespace constraint
}  // namespace rocprofsys

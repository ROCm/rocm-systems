// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "constraint.hpp"
#include "common/delimit.hpp"
#include "common/env_vars.hpp"
#include "config.hpp"
#include "utility.hpp"

#include <cstdint>
#include <ctime>
#include <string>

namespace rocprofsys
{
namespace constraint
{
std::vector<spec>
get_trace_specs()
{
    auto _v = std::vector<constraint::spec>{};

    {
        const auto _delay_v =
            config::get_setting_value<double>(std::string{ env_vars::TRACE_DELAY })
                .value_or(0.0);
        const auto _duration_v =
            config::get_setting_value<double>(std::string{ env_vars::TRACE_DURATION })
                .value_or(0.0);
        if(_delay_v > 0.0 || _duration_v > 0.0)
            _v.push_back(spec{ _delay_v, _duration_v, 1 });
    }

    // Each ROCPROFSYS_TRACE_PERIODS entry: delay[:duration[:repeat]]
    // Clock for all entries is set via ROCPROFSYS_TRACE_PERIOD_CLOCK_ID.
    if(auto _periods_v =
           config::get_setting_value<std::string>(std::string{ env_vars::TRACE_PERIODS })
               .value_or("");
       !_periods_v.empty())
    {
        const auto _default_delay =
            config::get_setting_value<double>(std::string{ env_vars::TRACE_DELAY })
                .value_or(0.0);
        const auto _default_dur =
            config::get_setting_value<double>(std::string{ env_vars::TRACE_DURATION })
                .value_or(0.0);
        for(const auto& _entry : rocprofsys::common::delimit(_periods_v, " ;\t\n"))
        {
            const auto _parts = rocprofsys::common::delimit(_entry, ":");
            spec       _s{ _default_delay, _default_dur, 1 };
            if(!_parts.empty()) _s.delay = utility::convert<double>(_parts.at(0));
            if(_parts.size() > 1) _s.duration = utility::convert<double>(_parts.at(1));
            if(_parts.size() > 2)
                _s.repeat = utility::convert<std::uint64_t>(_parts.at(2));
            _v.push_back(_s);
        }
    }

    return _v;
}

clockid_t
get_trace_period_clock_id()
{
    const auto _str =
        config::get_setting_value<std::string>(
            std::string{ env_vars::TRACE_PERIOD_CLOCK_ID })
            .value_or("realtime");
    // "cputime" is the only value that changes runtime behaviour — it routes
    // delay/duration scheduling to clocks::posix(CLOCK_PROCESS_CPUTIME_ID) so
    // windows tick in process CPU time rather than wall-clock time.
    // Any other value (including the default "realtime") uses clocks::steady.
    return (_str == "cputime") ? CLOCK_PROCESS_CPUTIME_ID : CLOCK_REALTIME;
}

bool
trace_has_initial_delay()
{
    const auto specs = get_trace_specs();
    return !specs.empty() && specs.front().delay > 0.0;
}
}  // namespace constraint
}  // namespace rocprofsys

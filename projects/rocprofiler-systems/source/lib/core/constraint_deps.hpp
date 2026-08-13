// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/config.hpp"

#include <string>

namespace rocprofsys::constraint
{
// Default Externals for trace_config: reads the trace-window settings from the
// global config singleton. Replaced by a mock in tests.
struct default_trace_config_externals
{
    static double get_trace_delay() { return ::rocprofsys::config::get_trace_delay(); }

    static double get_trace_duration()
    {
        return ::rocprofsys::config::get_trace_duration();
    }

    static std::string get_trace_periods()
    {
        return ::rocprofsys::config::get_trace_periods();
    }

    static std::string get_trace_period_clock()
    {
        return ::rocprofsys::config::get_trace_period_clock();
    }
};
}  // namespace rocprofsys::constraint

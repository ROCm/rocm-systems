// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "input_parameters.h"

#include "environ_cache.h"

#include <iostream>
#include <utility>

using namespace rocprofiler_compute_tool;

EnvInputParameters::EnvInputParameters(std::shared_ptr<const EnvironCache> environ)
    : m_environ{std::move(environ)}
{
}

std::string_view EnvInputParameters::get_or_warn(std::string_view env_var_name,
                                                 std::string_view default_value)
{
    const auto v = m_environ->get(env_var_name);
    if (v && !v->empty())
        return *v;

#ifndef NDEBUG
    const auto reason = v ? std::string_view{"empty"} : std::string_view{"unset"};
    std::clog << "[rocprofiler-compute] DEBUG: " << env_var_name << " is " << reason
              << "; defaulting to \"" << default_value << "\"" << std::endl;
#endif
    return default_value;
}

std::string_view EnvInputParameters::get_output_path()
{
    return get_or_warn("ROCPROF_OUTPUT_PATH", kDefaultOutputPath);
}

std::string_view EnvInputParameters::get_requested_counters()
{
    return get_or_warn("ROCPROF_COUNTERS", kDefaultRequestedCounters);
}

std::string_view EnvInputParameters::get_iteration_multiplexing_mode()
{
    return get_or_warn("ROCPROF_ITERATION_MULTIPLEXING", kDefaultIterationMultiplexingMode);
}

std::string_view EnvInputParameters::get_kernel_filter_include_regex()
{
    return get_or_warn("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX", kDefaultKernelFilterIncludeRegex);
}

std::string_view EnvInputParameters::get_kernel_filter_range()
{
    return get_or_warn("ROCPROF_KERNEL_FILTER_RANGE", kDefaultKernelFilterRange);
}

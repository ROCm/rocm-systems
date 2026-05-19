// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "input_parameters.h"

#include "environ_cache.h"

#include <utility>

using namespace rocprofiler_compute_tool;

EnvInputParameters::EnvInputParameters(std::shared_ptr<const EnvironCache> environ)
    : m_environ{std::move(environ)}
{
}

std::optional<std::string_view> EnvInputParameters::get_output_path()
{
    return m_environ->get("ROCPROF_OUTPUT_PATH");
}

std::optional<std::string_view> EnvInputParameters::get_requested_counters()
{
    return m_environ->get("ROCPROF_COUNTERS");
}

std::optional<std::string_view> EnvInputParameters::get_iteration_multiplexing_mode()
{
    return m_environ->get("ROCPROF_ITERATION_MULTIPLEXING");
}

std::optional<std::string_view> EnvInputParameters::get_kernel_filter_include_regex()
{
    return m_environ->get("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX");
}

std::optional<std::string_view> EnvInputParameters::get_kernel_filter_range()
{
    return m_environ->get("ROCPROF_KERNEL_FILTER_RANGE");
}

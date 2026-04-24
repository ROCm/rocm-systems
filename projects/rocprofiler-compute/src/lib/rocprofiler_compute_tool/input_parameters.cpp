// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "input_parameters.h"

#include <stdlib.h>

using namespace rocprofiler_compute_tool;

const char* env_input_parameters_t::get_output_path()
{
    return getenv("ROCPROF_OUTPUT_PATH");
}

const char* env_input_parameters_t::get_requested_counters()
{
    return getenv("ROCPROF_COUNTERS");
}

const char* env_input_parameters_t::get_iteration_multiplexing_mode()
{
    return getenv("ROCPROF_ITERATION_MULTIPLEXING");
}

const char* env_input_parameters_t::get_kernel_filter_include_regex()
{
    return getenv("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX");
}

const char* env_input_parameters_t::get_kernel_filter_range()
{
    return getenv("ROCPROF_KERNEL_FILTER_RANGE");
}

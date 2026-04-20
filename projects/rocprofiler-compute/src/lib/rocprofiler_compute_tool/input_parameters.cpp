// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "input_parameters.h"

#include <stdlib.h>

using namespace rocprofiler_compute_tool;

const char* EnvInputParameters::get_output_path()
{
    return getenv("ROCPROF_OUTPUT_PATH");
}

const char* EnvInputParameters::get_requested_counters()
{
    return getenv("ROCPROF_COUNTERS");
}

const char* EnvInputParameters::get_iteration_multiplexing_mode()
{
    return getenv("ROCPROF_ITERATION_MULTIPLEXING");
}

const char* EnvInputParameters::get_kernel_filter_include_regex()
{
    return getenv("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX");
}

const char* EnvInputParameters::get_kernel_filter_range()
{
    return getenv("ROCPROF_KERNEL_FILTER_RANGE");
}

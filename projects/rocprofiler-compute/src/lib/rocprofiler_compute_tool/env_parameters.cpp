// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "env_parameters.h"

#include <string>

using namespace rocm_compute;

std::string env_parameters_impl_t::get_output_path()
{
    const char* val = getenv("ROCPROF_OUTPUT_PATH");
    return val ? std::string(val) : std::string();
}

std::string env_parameters_impl_t::get_requested_counters()
{
    const char* val = getenv("ROCPROF_COUNTERS");
    return val ? std::string(val) : std::string();
}

std::string env_parameters_impl_t::get_iteration_multiplexing_mode()
{
    const char* val = getenv("ROCPROF_ITERATION_MULTIPLEXING");
    return val ? std::string(val) : std::string();
}

std::string env_parameters_impl_t::get_kernel_filter_include_regex()
{
    const char* val = getenv("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX");
    return val ? std::string(val) : std::string();
}

std::string env_parameters_impl_t::get_kernel_filter_range()
{
    const char* val = getenv("ROCPROF_KERNEL_FILTER_RANGE");
    return val ? std::string(val) : std::string();
}

std::string env_parameters_impl_t::get_pc_sampling_mode() const
{
    return "stochastic";
    // const char* val = getenv("ROCPROF_PC_SAMPLING_METHOD");
    // return val ? std::string(val) : std::string();
}

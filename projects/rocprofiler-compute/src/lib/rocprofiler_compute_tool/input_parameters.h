// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "pc_sampling_collector.h"

namespace rocprofiler_compute_tool
{
class input_parameters_t
{
public:
    virtual ~input_parameters_t()                         = default;
    virtual const char* get_output_path()                 = 0;
    virtual const char* get_requested_counters()          = 0;
    virtual const char* get_iteration_multiplexing_mode() = 0;
    virtual const char* get_kernel_filter_include_regex() = 0;
    virtual const char* get_kernel_filter_range()         = 0;
    virtual const char* get_pc_sampling_mode() const      = 0;
};

class env_input_parameters_t : public input_parameters_t
{
public:
    const char* get_output_path() override;
    const char* get_requested_counters() override;
    const char* get_iteration_multiplexing_mode() override;
    const char* get_kernel_filter_include_regex() override;
    const char* get_kernel_filter_range() override;
    const char* get_pc_sampling_mode() const;
};
}  // namespace rocprofiler_compute_tool

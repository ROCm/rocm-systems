// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

namespace rocprofiler_compute_tool
{
class InputParameters
{
public:
    virtual const char* get_output_path()                 = 0;
    virtual const char* get_requested_counters()          = 0;
    virtual const char* get_iteration_multiplexing_mode() = 0;
    virtual const char* get_kernel_filter_include_regex() = 0;
    virtual const char* get_kernel_filter_range()         = 0;
};

class EnvInputParameters : public InputParameters
{
public:
    const char* get_output_path() override;
    const char* get_requested_counters() override;
    const char* get_iteration_multiplexing_mode() override;
    const char* get_kernel_filter_include_regex() override;
    const char* get_kernel_filter_range() override;
};
}  // namespace rocprofiler_compute_tool

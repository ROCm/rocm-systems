#pragma once

namespace rocprof_compute_tool
{
class InputParameters
{
public:
    virtual char* get_output_path()                 = 0;
    virtual char* get_requested_counters()          = 0;
    virtual char* get_iteration_multiplexing_mode() = 0;
    virtual char* get_kernel_filter_include_regex() = 0;
    virtual char* get_kernel_filter_range()         = 0;
};

class EnvInputParameters : public InputParameters
{
public:
    char* get_output_path() override;
    char* get_requested_counters() override;
    char* get_iteration_multiplexing_mode() override;
    char* get_kernel_filter_include_regex() override;
    char* get_kernel_filter_range() override;
};
}  // namespace rocprof_compute_tool
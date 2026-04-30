// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <string>

namespace rocm_compute
{
class env_parameters_t
{
public:
    virtual ~env_parameters_t()                         = default;
    virtual std::string get_output_path() = 0;
    virtual std::string get_requested_counters() = 0;
    virtual std::string get_iteration_multiplexing_mode() = 0;
    virtual std::string get_kernel_filter_include_regex() = 0;
    virtual std::string get_kernel_filter_range() = 0;
    virtual std::string get_pc_sampling_mode() const = 0;
};

class env_parameters_impl_t : public env_parameters_t
{
public:
    std::string get_output_path() override;
    std::string get_requested_counters() override;
    std::string get_iteration_multiplexing_mode() override;
    std::string get_kernel_filter_include_regex() override;
    std::string get_kernel_filter_range() override;
    std::string get_pc_sampling_mode() const override;
};
}  // namespace rocm_compute

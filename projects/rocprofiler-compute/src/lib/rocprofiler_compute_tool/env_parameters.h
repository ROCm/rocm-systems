// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <string>

namespace rocm_compute
{
class env_parameters_t
{
public:
    virtual ~env_parameters_t()                                 = default;
    virtual std::string get_output_path() const                 = 0;
    virtual std::string get_requested_counters() const          = 0;
    virtual std::string get_iteration_multiplexing_mode() const = 0;
    virtual std::string get_kernel_filter_include_regex() const = 0;
    virtual std::string get_kernel_filter_range() const         = 0;
    virtual std::string get_pc_sampling_mode() const            = 0;
};

class env_parameters_impl_t : public env_parameters_t
{
public:
    std::string get_output_path() const override;
    std::string get_requested_counters() const override;
    std::string get_iteration_multiplexing_mode() const override;
    std::string get_kernel_filter_include_regex() const override;
    std::string get_kernel_filter_range() const override;
    std::string get_pc_sampling_mode() const override;
};
}  // namespace rocm_compute

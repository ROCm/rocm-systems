// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "environ_cache.h"

#include <memory>
#include <string_view>

namespace rocprofiler_compute_tool
{
class InputParameters
{
public:
    virtual ~InputParameters() = default;

    virtual std::string_view get_output_path()                 = 0;
    virtual std::string_view get_requested_counters()          = 0;
    virtual std::string_view get_iteration_multiplexing_mode() = 0;
    virtual std::string_view get_kernel_filter_include_regex() = 0;
    virtual std::string_view get_kernel_filter_range()         = 0;
    virtual std::string_view get_pc_sampling_method()          = 0;
};

class EnvInputParameters : public InputParameters
{
public:
    static constexpr std::string_view kDefaultOutputPath{"./"};
    static constexpr std::string_view kDefaultRequestedCounters{""};
    static constexpr std::string_view kDefaultIterationMultiplexingMode{""};
    static constexpr std::string_view kDefaultKernelFilterIncludeRegex{""};
    static constexpr std::string_view kDefaultKernelFilterRange{""};
    static constexpr std::string_view kDefaultPcSamplingMethod{""};

    explicit EnvInputParameters(std::shared_ptr<const EnvironCache> environ = EnvironCache::instance());
    std::string_view get_output_path() override;
    std::string_view get_requested_counters() override;
    std::string_view get_iteration_multiplexing_mode() override;
    std::string_view get_kernel_filter_include_regex() override;
    std::string_view get_kernel_filter_range() override;
    std::string_view get_pc_sampling_method() override;

private:
    std::string_view get(std::string_view env_var_name, std::string_view default_value);

    std::shared_ptr<const EnvironCache> m_environ;
};
}  // namespace rocprofiler_compute_tool

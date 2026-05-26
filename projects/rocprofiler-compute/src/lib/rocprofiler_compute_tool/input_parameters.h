// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "environ_cache.h"

#include <memory>
#include <optional>
#include <string_view>

namespace rocprofiler_compute_tool
{
class InputParameters
{
public:
    virtual ~InputParameters() = default;

    virtual std::optional<std::string_view> get_output_path()                 = 0;
    virtual std::optional<std::string_view> get_requested_counters()          = 0;
    virtual std::optional<std::string_view> get_iteration_multiplexing_mode() = 0;
    virtual std::optional<std::string_view> get_kernel_filter_include_regex() = 0;
    virtual std::optional<std::string_view> get_kernel_filter_range()         = 0;
};

class EnvInputParameters : public InputParameters
{
public:
    static constexpr std::string_view kDefaultOutputPath{"./workloads/"};
    static constexpr std::string_view kDefaultRequestedCounters{""};
    static constexpr std::string_view kDefaultIterationMultiplexingMode{""};
    static constexpr std::string_view kDefaultKernelFilterIncludeRegex{""};
    static constexpr std::string_view kDefaultKernelFilterRange{""};

    static std::string_view get_default_output_path() { return kDefaultOutputPath; }

    static std::string_view get_default_requested_counters() { return kDefaultRequestedCounters; }

    static std::string_view get_default_iteration_multiplexing_mode()
    {
        return kDefaultIterationMultiplexingMode;
    }

    static std::string_view get_default_kernel_filter_include_regex()
    {
        return kDefaultKernelFilterIncludeRegex;
    }

    static std::string_view get_default_kernel_filter_range() { return kDefaultKernelFilterRange; }

    explicit EnvInputParameters(std::shared_ptr<const EnvironCache> environ = EnvironCache::instance());
    std::optional<std::string_view> get_output_path() override;
    std::optional<std::string_view> get_requested_counters() override;
    std::optional<std::string_view> get_iteration_multiplexing_mode() override;
    std::optional<std::string_view> get_kernel_filter_include_regex() override;
    std::optional<std::string_view> get_kernel_filter_range() override;

private:
    std::shared_ptr<const EnvironCache> m_environ;
};
}  // namespace rocprofiler_compute_tool

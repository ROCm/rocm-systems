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

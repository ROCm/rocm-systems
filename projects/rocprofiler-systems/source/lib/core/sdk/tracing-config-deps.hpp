// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/env_vars.hpp"
#include "core/config.hpp"
#include "core/state.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace rocprofsys::rocprofiler_sdk
{

// Default Externals for tracing_config: reads settings and config from the global
// singletons. Replaced by a mock in tests.
struct default_externals
{
    static bool get_use_rcclp() { return ::rocprofsys::config::get_use_rcclp(); }
    static bool get_use_ompt() { return ::rocprofsys::config::get_use_ompt(); }
    static bool get_use_unified_memory_profiling()
    {
        return ::rocprofsys::config::get_use_unified_memory_profiling();
    }
    static std::string get_rocm_domains()
    {
        return ::rocprofsys::get_setting_value<std::string>(
                   std::string{ ::rocprofsys::env_vars::ROCM_DOMAINS })
            .value_or(std::string{});
    }

    static std::optional<std::string> get_setting_value(std::string_view setting_name)
    {
        return ::rocprofsys::get_setting_value<std::string>(std::string{ setting_name });
    }

    using ProcessState = ::rocprofsys::state::process;
};

}  // namespace rocprofsys::rocprofiler_sdk

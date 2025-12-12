// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/timemory.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
struct version_info
{
    std::uint32_t major     = 0;
    std::uint32_t minor     = 0;
    std::uint32_t patch     = 0;
    std::uint32_t formatted = 0;  // major * 10000 + minor * 100 + patch
};

void
config_settings(const std::shared_ptr<settings>&);

version_info&
get_version();

std::unordered_set<rocprofiler_callback_tracing_kind_t>
get_callback_domains();

std::unordered_set<rocprofiler_buffer_tracing_kind_t>
get_buffered_domains();

std::vector<std::int32_t>
get_operations(rocprofiler_callback_tracing_kind_t kindv);

std::vector<std::int32_t>
get_operations(rocprofiler_buffer_tracing_kind_t kindv);

std::vector<std::string>
get_rocm_events();

std::unordered_set<std::int32_t>
get_backtrace_operations(rocprofiler_callback_tracing_kind_t kindv);

std::unordered_set<std::int32_t>
get_backtrace_operations(rocprofiler_buffer_tracing_kind_t kindv);

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

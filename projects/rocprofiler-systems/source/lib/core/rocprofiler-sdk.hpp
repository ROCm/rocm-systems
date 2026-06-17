// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/rocprofiler_sdk_backend.hpp"
#include "core/timemory.hpp"

#include <cstdint>
#include <memory>
#include <string>
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

template <typename Backend>
class sdk_core
{
public:
    static void config_settings(const std::shared_ptr<settings>&);

    static version_info& get_version();

    static std::unordered_set<typename Backend::callback_tracing_kind>
    get_callback_domains();

    static std::unordered_set<typename Backend::buffer_tracing_kind>
    get_buffered_domains();

    static std::vector<std::int32_t> get_operations(Backend::callback_tracing_kind kindv);

    static std::vector<std::int32_t> get_operations(Backend::buffer_tracing_kind kindv);

    static std::vector<std::string> get_rocm_events();

    static std::unordered_set<std::int32_t> get_backtrace_operations(
        Backend::callback_tracing_kind kindv);

    static std::unordered_set<std::int32_t> get_backtrace_operations(
        Backend::buffer_tracing_kind kindv);

private:
    struct operation_options
    {
        std::string operations_include            = {};
        std::string operations_exclude            = {};
        std::string operations_annotate_backtrace = {};
    };

    static std::unordered_map<typename Backend::callback_tracing_kind, operation_options>
        callback_operation_option_names;
    static std::unordered_map<typename Backend::buffer_tracing_kind, operation_options>
        buffered_operation_option_names;

    static std::unordered_set<std::int32_t> get_operations_impl(
        Backend::callback_tracing_kind kindv, const std::string& optname = {});

    static std::unordered_set<std::int32_t> get_operations_impl(
        Backend::buffer_tracing_kind kindv, const std::string& optname = {});

    static std::vector<std::int32_t> get_operations_impl(
        const std::unordered_set<std::int32_t>& complete,
        const std::unordered_set<std::int32_t>& include,
        const std::unordered_set<std::int32_t>& exclude);
};

extern template class sdk_core<backend>;

using core_sdk = sdk_core<backend>;

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

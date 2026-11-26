// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocsys/schema.hpp"
#include "parser/backend.hpp"
#include "parser/schema.hpp"

#include "common/env_vars.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::rocsys
{
namespace
{
/// A --traces name that adds a domain to ROCPROFSYS_ROCM_DOMAINS.
struct rocm_trace
{
    std::string_view name;
    std::string_view alias;
    std::string_view domain;
};

/// A --traces name that turns on a host runtime through its ROCPROFSYS_USE_* switch.
struct runtime_trace
{
    std::string_view name;
    std::string_view use_switch;
};

constexpr auto k_rocm_traces = std::to_array<rocm_trace>({
    { .name = "hip-api", .alias = "hip", .domain = "hip_api" },
    { .name = "hsa-api", .alias = "hsa", .domain = "hsa_api" },
    { .name = "roctx", .alias = {}, .domain = "marker_api" },
    { .name = "kernel-dispatch", .alias = {}, .domain = "kernel_dispatch" },
    { .name = "memory-copy", .alias = {}, .domain = "memory_copy" },
    { .name = "memory-allocation", .alias = {}, .domain = "memory_allocation" },
    { .name = "rocshmem", .alias = {}, .domain = "rocshmem_api" },
});

constexpr auto k_runtime_traces = std::to_array<runtime_trace>({
    { .name = "rccl", .use_switch = env_vars::USE_RCCLP },
    { .name = "mpi", .use_switch = env_vars::USE_MPIP },
    { .name = "openmp", .use_switch = env_vars::USE_OMPT },
    { .name = "ucx", .use_switch = env_vars::USE_UCX },
    { .name = "openshmem", .use_switch = env_vars::USE_OPENSHMEM },
});

constexpr std::string_view k_trace_backend = "rocprofiler-sdk";

class trace_backend final : public parser::backend_interface
{
public:
    explicit trace_backend(std::vector<std::string> supported_domains)
    : m_supported_domains{ std::move(supported_domains) }
    {}

    [[nodiscard]] std::string_view name() const override { return k_trace_backend; }

    [[nodiscard]] std::vector<std::string> choices(
        const parser::schema_param&) const override
    {
        const auto supported = [this](const rocm_trace& item) {
            return std::ranges::find(m_supported_domains, item.domain) !=
                   m_supported_domains.end();
        };
        std::vector<std::string> names;
        for(const auto& item : k_rocm_traces | std::views::filter(supported))
        {
            names.emplace_back(item.name);
            if(!item.alias.empty())
            {
                names.emplace_back(item.alias);
            }
        }
        return names;
    }

private:
    std::vector<std::string> m_supported_domains;
};

parser::schema_param
traces()
{
    auto param        = parser::schema_param{};
    param.id          = "traces";
    param.category    = "tracing";
    param.description = "APIs and runtimes to trace; names left out are not traced";
    param.type        = parser::param_type::enumeration_list;
    param.arg         = "--traces";
    param.backend     = std::string{ k_trace_backend };
    for(const auto& item : k_runtime_traces)
    {
        param.choices.emplace_back(item.name);
    }
    return param;
}

parser::schema_param
output_path()
{
    auto param        = parser::schema_param{};
    param.id          = "output_path";
    param.category    = "output";
    param.description = "Directory for output files";
    param.type        = parser::param_type::path;
    param.env         = env_vars::OUTPUT_PATH;
    param.arg         = "--output-path";
    return param;
}

parser::schema_param
log_level()
{
    auto param        = parser::schema_param{};
    param.id          = "log_level";
    param.category    = "logging";
    param.description = "Log level";
    param.type        = parser::param_type::enumeration;
    param.env         = env_vars::LOG_LEVEL;
    param.arg         = "--log-level";
    // Every spelling the tool's logger reads: warning is warn, and err is error.
    param.choices = { "trace", "debug", "info",     "warn", "warning",
                      "error", "err",   "critical", "off" };
    return param;
}
}  // namespace

parser::schema
make_schema()
{
    auto catalog = parser::schema{};
    for(auto& param : std::vector{ traces(), output_path(), log_level() })
    {
        const auto param_id = param.id;
        if(!catalog.add_param(std::move(param)))
        {
            throw std::logic_error(
                fmt::format("rocsys option '{}' is not unique", param_id));
        }
    }
    return catalog;
}

std::unique_ptr<parser::backend_interface>
make_trace_backend(std::vector<std::string> supported_domains)
{
    return std::make_unique<trace_backend>(std::move(supported_domains));
}

parser::value_map_t
traces_environment(const std::vector<std::string>& names)
{
    const auto listed = [&names](std::string_view name) {
        return std::ranges::find(names, name) != names.end();
    };
    const auto selected = [&listed](const rocm_trace& item) {
        return listed(item.name) || (!item.alias.empty() && listed(item.alias));
    };
    auto domains = k_rocm_traces | std::views::filter(selected) |
                   std::views::transform(&rocm_trace::domain);
    auto environment =
        parser::value_map_t{ { env_vars::ROCM_DOMAINS,
                               fmt::format("{}", fmt::join(domains, ",")) } };
    for(const auto& item : k_runtime_traces)
    {
        environment.emplace(item.use_switch, listed(item.name) ? "true" : "false");
    }
    return environment;
}
}  // namespace rocprofsys::rocsys

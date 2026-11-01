// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"
#include "parser/value_provider.hpp"

#include "common/string_utility.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
class catalog_provider final : public value_provider_interface
{
public:
    catalog_provider(std::string name, std::vector<std::string> choices,
                     std::optional<std::string> fallback)
    : m_name(std::move(name))
    , m_choices(std::move(choices))
    , m_default(std::move(fallback))
    {}

    [[nodiscard]] std::string_view name() const override { return m_name; }

    [[nodiscard]] std::vector<std::string> choices(const schema_param&) const override
    {
        return m_choices;
    }

    [[nodiscard]] std::optional<std::string> default_value(
        const schema_param&) const override
    {
        return m_default;
    }

private:
    std::string                m_name;
    std::vector<std::string>   m_choices;
    std::optional<std::string> m_default;
};

std::unique_ptr<value_provider_interface>
catalog(std::string name, std::vector<std::string> choices,
        std::optional<std::string> fallback = std::nullopt)
{
    return std::make_unique<catalog_provider>(std::move(name), std::move(choices),
                                              std::move(fallback));
}

bool
is_name_list(const schema_param& param, std::string_view value)
{
    if(param.type == param_type::enumeration_list)
    {
        return true;
    }
    if(param.type != param_type::string)
    {
        return false;
    }
    return value.find(list_separator(param)) != std::string_view::npos;
}

[[nodiscard]] bool
open_name_allowed(const schema_param& param, std::string_view value)
{
    if(!is_name_list(param, value))
    {
        const bool present = !utility::string::trim(value).empty();
        return present || param.type != param_type::enumeration;
    }
    const auto separator = list_separator(param);
    if(separator.empty())
    {
        return false;
    }
    const auto parts = split_list(value, separator).value_or(std::vector<std::string>{});
    return std::ranges::all_of(
        parts, [](const std::string& element) { return !element.empty(); });
}

// Hardware-counter names are not a fixed catalog. Any non-empty name is accepted.
class open_name_provider final : public value_provider_interface
{
public:
    explicit open_name_provider(std::string name)
    : m_name(std::move(name))
    {}

    [[nodiscard]] std::string_view name() const override { return m_name; }

    [[nodiscard]] std::vector<std::string> choices(const schema_param&) const override
    {
        return {};
    }

    [[nodiscard]] std::optional<std::string> default_value(
        const schema_param&) const override
    {
        return std::nullopt;
    }

    [[nodiscard]] bool accepts(const schema_param& param,
                               std::string_view    value) const override
    {
        return open_name_allowed(param, value);
    }

private:
    std::string m_name;
};

std::unique_ptr<value_provider_interface>
open_catalog(std::string name)
{
    return std::make_unique<open_name_provider>(std::move(name));
}

// Snapshot of tracing_config::get_domain_choices() for the current SDK.
std::vector<std::string>
rocm_domains()
{
    return { "hip_api",
             "hip_compiler_api",
             "hip_runtime_api",
             "hip_stream",
             "hipfile_api",
             "hsa_amd_ext_api",
             "hsa_api",
             "hsa_core_api",
             "hsa_finalize_ext_api",
             "hsa_image_ext_api",
             "kernel_dispatch",
             "kfd_event_dropped_events",
             "kfd_event_page_fault",
             "kfd_event_page_migrate",
             "kfd_event_queue",
             "kfd_event_unmap_from_gpu",
             "kfd_events",
             "kfd_page_fault",
             "kfd_page_migrate",
             "kfd_queue",
             "marker_api",
             "memory_allocation",
             "memory_copy",
             "ompt",
             "rccl_api",
             "rocdecode_api",
             "rocjpeg_api",
             "rocshmem_api",
             "roctx",
             "scratch_memory" };
}

std::vector<std::string>
gpu_metric_names()
{
    return { "busy",  "gfx_clock",  "jpeg_activity", "mem_clock",    "mem_usage", "pcie",
             "power", "sdma_usage", "temp",          "vcn_activity", "xgmi" };
}
}  // namespace

void
value_provider_registry::add_builtins()
{
    static_cast<void>(add(catalog(
        "rocm_domains_sdk", rocm_domains(),
        std::string{
            "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory" })));
    static_cast<void>(add(catalog("parallel_runtimes", { "mpi", "openmp", "pthread" })));
    static_cast<void>(add(catalog("gpu_metrics", gpu_metric_names())));
    static_cast<void>(add(open_catalog("hw_counters")));
}

}  // namespace rocprofsys::parser

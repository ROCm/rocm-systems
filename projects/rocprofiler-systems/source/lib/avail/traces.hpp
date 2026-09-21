// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/records.hpp"

#include "common/string_utility.hpp"

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys::avail
{
/// One SDK tracing kind: the domain token plus the operation names the SDK
/// reports for it (HIP/HSA API functions, KFD events, and so on).
struct trace_kind_entry
{
    std::string              name       = {};
    std::vector<std::string> operations = {};
};

/// Supplies the domain names rocprofiler-sdk reports for callback and buffered
/// tracing. Names are the SDK's own strings; aliases and defaults are added
/// here so the listing matches what ROCPROFSYS_ROCM_DOMAINS accepts.
template <typename T>
concept trace_inventory_source = requires(T& source) {
    { source.callback_domains() } -> std::same_as<std::vector<trace_kind_entry>>;
    { source.buffered_domains() } -> std::same_as<std::vector<trace_kind_entry>>;
};

/// SDK kinds that are not user-facing domain tokens (internal or covered by
/// the marker_api / roctx aliases).
[[nodiscard]] inline bool
is_internal_trace_domain(std::string_view name) noexcept
{
    return name.empty() || name == "none" || name == "correlation_id_retirement" ||
           name == "marker_core_api" || name == "marker_control_api" ||
           name == "marker_name_api" || name == "code_object";
}

[[nodiscard]] inline bool
is_default_trace_domain(std::string_view name) noexcept
{
    return name == "hip_runtime_api" || name == "marker_api" ||
           name == "kernel_dispatch" || name == "memory_copy" || name == "scratch_memory";
}

[[nodiscard]] inline bool
is_listed_operation(std::string_view name) noexcept
{
    auto lower = utility::string::to_lower(name);
    return !lower.empty() && lower != "none";
}

enum class trace_section : std::uint8_t
{
    gpu_rocm,
    host,
    other
};

[[nodiscard]] inline bool
is_host_runtime_trace(std::string_view name) noexcept
{
    return name == "mpi" || name == "ucx" || name == "oshmem" || name == "openshmem" ||
           name == "kokkos" || name == "ompt";
}

/// GOTCHA VA-API wrapping and pthread/OS-runtime tracing (listed as osrt).
[[nodiscard]] inline bool
is_other_runtime_trace(std::string_view name) noexcept
{
    return name == "vaapi" || name == "osrt";
}

/// GPU / ROCm domains this tool actually traces (see tracing_config
/// get_supported_{callback,buffer}_domains). SDK kinds outside this set are
/// omitted from `--traces`.
[[nodiscard]] inline bool
is_gpu_rocm_trace(std::string_view name) noexcept
{
    constexpr std::string_view k_tokens[] = {
        "hip_api",
        "hip_compiler_api",
        "hip_runtime_api",
        "hipfile_api",
        "hsa_api",
        "hsa_amd_ext_api",
        "hsa_core_api",
        "hsa_finalize_ext_api",
        "hsa_image_ext_api",
        "marker_api",
        "roctx",
        "kernel_dispatch",
        "memory_copy",
        "memory_allocation",
        "scratch_memory",
        "page_migration",
        "kfd_events",
        "rccl_api",
        "rocshmem_api",
        "rocdecode_api",
        "rocjpeg_api",
    };
    if(!name.empty() && name.compare(0, 3, "kfd") == 0) return true;
    for(auto token : k_tokens)
        if(name == token) return true;
    return false;
}

[[nodiscard]] inline bool
is_collectable_trace(std::string_view name) noexcept
{
    return is_gpu_rocm_trace(name) || is_host_runtime_trace(name) ||
           is_other_runtime_trace(name);
}

[[nodiscard]] inline trace_section
section_for_trace(std::string_view name) noexcept
{
    if(is_host_runtime_trace(name)) return trace_section::host;
    if(is_other_runtime_trace(name) || !is_gpu_rocm_trace(name))
        return trace_section::other;
    return trace_section::gpu_rocm;
}

/// Alias members stay in the catalog for `--list-operations` but are omitted
/// from `--traces` so `hip_api` is not followed by `hip_runtime_api`.
[[nodiscard]] inline bool
is_hidden_alias_member(const trace_domain_record&              record,
                       const std::vector<trace_domain_record>& traces) noexcept
{
    if(record.alias) return false;
    for(const auto& trace : traces)
    {
        if(!trace.alias) continue;
        for(const auto& member : trace.alias_members)
        {
            if(member == record.name) return true;
        }
    }
    return false;
}

inline void
append_operations(std::vector<std::string>& dest, const std::vector<std::string>& src)
{
    for(const auto& name : src)
    {
        if(is_listed_operation(name)) dest.push_back(name);
    }
}

inline void
sort_unique(std::vector<std::string>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

/// Merges callback and buffered name tables, keeps only domains this tool
/// traces, and adds hip_api / hsa_api / marker_api / roctx / kfd_events
/// aliases plus host runtimes. Operation names stay attached so
/// `--list-operations` can expand a domain or an alias.
template <trace_inventory_source Source>
[[nodiscard]] trace_query_result
collect_traces(Source& source)
{
    auto result      = trace_query_result{};
    auto by_name     = std::unordered_map<std::string, trace_domain_record>{};
    auto ops_by_name = std::unordered_map<std::string, std::vector<std::string>>{};

    auto ingest = [&](std::vector<trace_kind_entry> entries, bool callback,
                      bool buffered) {
        for(auto& entry : entries)
        {
            auto name = utility::string::to_lower(entry.name);
            if(name.empty()) continue;

            append_operations(ops_by_name[name], entry.operations);

            if(is_internal_trace_domain(name)) continue;
            if(!is_collectable_trace(name)) continue;

            auto& record = by_name[name];
            record.name  = name;
            if(callback) record.callback = true;
            if(buffered) record.buffered = true;
            append_operations(record.operations, entry.operations);
        }
    };

    try
    {
        ingest(source.callback_domains(), true, false);
        ingest(source.buffered_domains(), false, true);
    } catch(const std::exception& e)
    {
        result.diagnostics.push_back({ source_id::rocprofiler_sdk, e.what() });
        return result;
    }

    auto add_alias = [&](std::string name, std::vector<std::string> members) {
        auto& record         = by_name[name];
        record.name          = std::move(name);
        record.alias         = true;
        record.alias_members = std::move(members);
        record.is_default    = is_default_trace_domain(record.name);
    };

    // Membership matches tracing_config::get_{callback,buffered}_domain_map.
    add_alias("hip_api", { "hip_compiler_api", "hip_runtime_api" });
    add_alias("hsa_api", { "hsa_amd_ext_api", "hsa_core_api", "hsa_finalize_ext_api",
                           "hsa_image_ext_api" });
    add_alias("marker_api", { "marker_core_api" });
    add_alias("roctx", { "marker_core_api" });

    auto kfd_members = std::vector<std::string>{};
    for(const auto& [name, record] : by_name)
    {
        if(!record.alias && name.compare(0, 3, "kfd") == 0) kfd_members.push_back(name);
    }
    if(!kfd_members.empty())
    {
        std::sort(kfd_members.begin(), kfd_members.end());
        add_alias("kfd_events", std::move(kfd_members));
    }

    // Host GOTCHA runtimes and OS/pthread tracing are not SDK domains.
    constexpr std::string_view k_static_tokens[] = {
        "mpi", "ucx", "oshmem", "kokkos", "ompt", "vaapi", "osrt",
    };
    for(auto token : k_static_tokens)
    {
        auto name = std::string{ token };
        if(by_name.find(name) != by_name.end()) continue;
        auto& record = by_name[name];
        record.name  = std::move(name);
    }

    result.traces.reserve(by_name.size());
    for(auto& [name, record] : by_name)
    {
        record.is_default = record.is_default || is_default_trace_domain(name);
        if(record.alias)
        {
            record.operations.clear();
            for(const auto& member : record.alias_members)
            {
                auto itr = ops_by_name.find(member);
                if(itr != ops_by_name.end())
                    append_operations(record.operations, itr->second);
            }
        }
        sort_unique(record.operations);
        result.traces.push_back(std::move(record));
    }

    std::sort(result.traces.begin(), result.traces.end(),
              [](const trace_domain_record& lhs, const trace_domain_record& rhs) {
                  return lhs.name < rhs.name;
              });

    return result;
}

/// Tracing domains from the live rocprofiler-sdk backend name tables.
[[nodiscard]] trace_query_result
query_traces();
}  // namespace rocprofsys::avail

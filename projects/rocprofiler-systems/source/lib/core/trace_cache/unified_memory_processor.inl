// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Template definitions for unified_memory_processor_t. Include only from TUs
// that need to instantiate the template (production .cpp + tests). Other
// callers include unified_memory_processor.hpp and rely on extern template.

#pragma once

#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/trace_cache/unified_memory_processor.hpp"
#include "logger/debug.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <spdlog/fmt/fmt.h>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <utility>

namespace rocprofsys
{
namespace trace_cache
{

namespace detail
{
inline constexpr uint64_t kKiB = 1024ULL;
inline constexpr uint64_t kMiB = kKiB * 1024;
inline constexpr uint64_t kGiB = kMiB * 1024;

inline constexpr uint64_t kNsPerUs  = 1000ULL;
inline constexpr uint64_t kNsPerMs  = kNsPerUs * 1000;
inline constexpr uint64_t kNsPerSec = kNsPerMs * 1000;

// Largest double < 2^64; equality would overflow on uint64_t cast (UB,
// [conv.fpint]). nextafter isn't constexpr until C++23.
inline const double kMaxSafeUint64 =
    std::nextafter(static_cast<double>(std::numeric_limits<uint64_t>::max()), 0.0);

[[nodiscard]] inline std::string
format_size(uint64_t bytes)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    if(bytes >= kGiB)
        oss << (static_cast<double>(bytes) / kGiB) << "GB";
    else if(bytes >= kMiB)
        oss << (static_cast<double>(bytes) / kMiB) << "MB";
    else if(bytes >= kKiB)
        oss << (static_cast<double>(bytes) / kKiB) << "KB";
    else
        oss << bytes << "B";

    return oss.str();
}

[[nodiscard]] inline std::string
generate_unified_memory_output_path(int pid, const std::string& output_dir,
                                    std::string_view ext)
{
    return rocprofsys::get_output_absolute_path("unified_memory", ext,
                                                std::to_string(pid), output_dir);
}

[[nodiscard]] inline bool
is_known_agent_type(uint8_t raw_type) noexcept
{
    using agent_type_underlying = std::underlying_type_t<agent_type>;
    return raw_type <= static_cast<agent_type_underlying>(agent_type::NIC);
}

[[nodiscard]] inline std::string
format_time(uint64_t nanoseconds)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    if(nanoseconds >= kNsPerSec)
        oss << (static_cast<double>(nanoseconds) / kNsPerSec) << "s";
    else if(nanoseconds >= kNsPerMs)
        oss << (static_cast<double>(nanoseconds) / kNsPerMs) << "ms";
    else if(nanoseconds >= kNsPerUs)
        oss << (static_cast<double>(nanoseconds) / kNsPerUs) << "us";
    else
        oss << nanoseconds << "ns";

    return oss.str();
}

}  // namespace detail

template <typename AgentManagerT, typename OutputRegistryT>
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::unified_memory_processor_t(
    std::shared_ptr<AgentManagerT> agent_mgr, int pid, std::string output_dir,
    OutputRegistryT& output_registry)
: processor_t<unified_memory_processor_t<AgentManagerT, OutputRegistryT>>()
, m_agent_manager(std::move(agent_mgr))
, m_pid(pid)
, m_output_dir(std::move(output_dir))
, m_output_registry(output_registry)
{
    const char* xnack    = std::getenv("HSA_XNACK");
    m_data.xnack_enabled = (xnack && std::strcmp(xnack, "1") == 0);

    const auto& all_agents = m_agent_manager->get_agents();
    for(const auto& agent_ptr : all_agents)
    {
        if(!agent_ptr) continue;

        m_node_type_cache[agent_ptr->node_id] = agent_ptr->type;

        if(agent_ptr->type == agent_type::GPU)
        {
            m_gpu_name_cache[agent_ptr->node_id] = agent_ptr->name;
        }
    }
}

template <typename AgentManagerT, typename OutputRegistryT>
void
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::prepare_for_processing()
{
    LOG_DEBUG("Preparing unified memory processor for processing");

    if(!m_data.xnack_enabled)
    {
        LOG_WARNING("HSA_XNACK is not set to 1. Unified memory profiling may show "
                    "limited data. Set HSA_XNACK=1 for page fault-driven migration.");
    }
}

template <typename AgentManagerT, typename OutputRegistryT>
void
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::finalize_processing()
{
    LOG_DEBUG("Finalizing unified memory processor");

    bool has_migrations = false;
    for(const auto& [device_id, summary] : m_data.devices)
    {
        if(summary.host_to_device.count > 0 || summary.device_to_host.count > 0 ||
           summary.device_to_device.count > 0)
        {
            has_migrations = true;
            break;
        }
    }

    if(!has_migrations && m_data.total_page_faults == 0)
    {
        LOG_INFO("No unified memory events captured (no migrations, no page faults). "
                 "Skipping output generation.");
        return;
    }

    std::string txt_path =
        detail::generate_unified_memory_output_path(m_pid, m_output_dir, "txt");
    std::ofstream txt_file(txt_path);
    if(!txt_file.is_open())
    {
        LOG_ERROR("Failed to open unified memory text output: {}", txt_path);
    }
    else
    {
        write_text_output(txt_file);
        txt_file.close();
        m_output_registry.register_file(txt_path, output_format::text);
        LOG_INFO("Unified memory text report written to: {}", txt_path);
    }

    std::string json_path =
        detail::generate_unified_memory_output_path(m_pid, m_output_dir, "json");
    std::ofstream json_file(json_path);
    if(!json_file.is_open())
    {
        LOG_ERROR("Failed to open unified memory JSON output: {}", json_path);
    }
    else
    {
        write_json_output(json_file);
        json_file.close();
        m_output_registry.register_file(json_path, output_format::json);
        LOG_INFO("Unified memory JSON report written to: {}", json_path);
    }

    LOG_INFO("Unified memory processor finalized successfully");
}

template <typename AgentManagerT, typename OutputRegistryT>
void
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::handle(
    const kfd_sample& sample)
{
    if(sample.category == "rocm_kfd_page_migrate")
        handle_page_migrate(sample);
    else if(sample.category == "rocm_kfd_page_fault")
        m_data.total_page_faults++;
}

template <typename AgentManagerT, typename OutputRegistryT>
void
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::handle_page_migrate(
    const kfd_sample& sample)
{
    auto agent_ids = parse_agent_ids_from_args(sample.args_str);
    if(!agent_ids.has_value())
    {
        LOG_TRACE("Failed to parse agent IDs from KFD page migration event");
        return;
    }

    auto [src_label, dst_label] = std::move(*agent_ids);
    auto direction              = classify_direction(src_label, dst_label);

    uint32_t device_id  = sample.device_id;
    auto [it, inserted] = m_data.devices.try_emplace(device_id);
    if(inserted)
        it->second.device_name = resolve_device_label(sample, src_label, dst_label);

    auto& device_summary = it->second;

    // Float-to-int overflow is UB ([conv.fpint]); guard NaN/inf/sign/2^64.
    uint64_t size_bytes = 0;
    if(std::isfinite(sample.value) && sample.value > 0.0 &&
       sample.value < detail::kMaxSafeUint64)
        size_bytes = static_cast<uint64_t>(sample.value);

    // Guard against non-monotonic KFD timestamps to avoid unsigned wrap.
    uint64_t duration_ns = (sample.end_timestamp >= sample.start_timestamp)
                               ? sample.end_timestamp - sample.start_timestamp
                               : 0;

    switch(direction)
    {
        case migration_direction::HOST_TO_DEVICE:
            device_summary.host_to_device.add_migration(size_bytes, duration_ns);
            break;
        case migration_direction::DEVICE_TO_HOST:
            device_summary.device_to_host.add_migration(size_bytes, duration_ns);
            break;
        case migration_direction::DEVICE_TO_DEVICE:
            device_summary.device_to_device.add_migration(size_bytes, duration_ns);
            break;
        case migration_direction::UNKNOWN:
            LOG_TRACE("Unknown migration direction for device {}", device_id);
            break;
    }

    // Initialize to sentinel; loop falls through to it on no match.
    const detail::trigger_entry* entry = &detail::kTriggerTable.back();
    for(const auto& row : detail::kTriggerTable)
    {
        if(row.kfd_name != nullptr && sample.name == row.kfd_name)
        {
            entry = &row;
            break;
        }
    }
    ++(m_data.triggers.*(entry->member));
}

template <typename AgentManagerT, typename OutputRegistryT>
std::optional<std::pair<uint32_t, uint32_t>>
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::parse_node_id_pair(
    const std::string& src_label, const std::string& dst_label) const
{
    auto parse_one = [](const std::string& s, uint32_t& out) -> bool {
        const char* first = s.data();
        const char* last  = s.data() + s.size();
        auto        res   = std::from_chars(first, last, out);
        return res.ec == std::errc{} && res.ptr == last;
    };

    uint32_t src_id = 0;
    uint32_t dst_id = 0;
    if(!parse_one(src_label, src_id) || !parse_one(dst_label, dst_id))
    {
        LOG_TRACE("Failed to parse node IDs from labels: src='{}', dst='{}'", src_label,
                  dst_label);
        return std::nullopt;
    }
    return std::pair{ src_id, dst_id };
}

template <typename AgentManagerT, typename OutputRegistryT>
typename unified_memory_processor_t<AgentManagerT, OutputRegistryT>::migration_direction
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::classify_direction(
    const std::string& src_label, const std::string& dst_label) const
{
    auto ids = parse_node_id_pair(src_label, dst_label);
    if(!ids.has_value()) return migration_direction::UNKNOWN;
    const auto [src_node_id, dst_node_id] = *ids;

    auto src_it = m_node_type_cache.find(src_node_id);
    auto dst_it = m_node_type_cache.find(dst_node_id);

    if(src_it == m_node_type_cache.end() || dst_it == m_node_type_cache.end())
    {
        LOG_TRACE("Node IDs not found in cache: src={}, dst={}", src_node_id,
                  dst_node_id);
        return migration_direction::UNKNOWN;
    }

    bool src_is_cpu = (src_it->second == agent_type::CPU);
    bool dst_is_cpu = (dst_it->second == agent_type::CPU);

    if(src_is_cpu && !dst_is_cpu)
        return migration_direction::HOST_TO_DEVICE;
    else if(!src_is_cpu && dst_is_cpu)
        return migration_direction::DEVICE_TO_HOST;
    else if(!src_is_cpu && !dst_is_cpu)
        return migration_direction::DEVICE_TO_DEVICE;
    else
        return migration_direction::UNKNOWN;
}

template <typename AgentManagerT, typename OutputRegistryT>
std::optional<std::pair<std::string, std::string>>
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::parse_agent_ids_from_args(
    const std::string& args_str) const
{
    std::string src_agent;
    std::string dst_agent;
    try
    {
        const auto args = process_arguments_string(args_str);
        for(const auto& a : args)
        {
            if(a.arg_name == "src_agent")
                src_agent = a.arg_value;
            else if(a.arg_name == "dst_agent")
                dst_agent = a.arg_value;
        }
    } catch(const std::exception& e)
    {
        LOG_TRACE("Failed to parse KFD args_str: {}", e.what());
        return std::nullopt;
    }

    if(src_agent.empty() || dst_agent.empty()) return std::nullopt;

    return std::make_pair(std::move(src_agent), std::move(dst_agent));
}

template <typename AgentManagerT, typename OutputRegistryT>
std::string
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::resolve_device_label(
    const kfd_sample& sample, const std::string& src_label,
    const std::string& dst_label) const
{
    // Fallback shape "CPU {device_id}" pinned by AgentLookupThrowFallsBackSafely.
    std::string cpu_name = fmt::format("CPU {}", sample.device_id);
    if(detail::is_known_agent_type(sample.device_type))
    {
        try
        {
            const auto& cpu_agent = m_agent_manager->get_agent_by_type_index(
                sample.device_id, static_cast<agent_type>(sample.device_type));
            if(!cpu_agent.name.empty()) cpu_name = cpu_agent.name;
        } catch(const std::exception& e)
        {
            LOG_TRACE("CPU agent lookup failed for device_id={}: {}", sample.device_id,
                      e.what());
        }
    }

    return fmt::format("{} (via {})", extract_gpu_name(src_label, dst_label), cpu_name);
}

template <typename AgentManagerT, typename OutputRegistryT>
std::string
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::extract_gpu_name(
    const std::string& src_label, const std::string& dst_label) const
{
    auto ids = parse_node_id_pair(src_label, dst_label);
    if(!ids.has_value()) return "GPU";
    const auto [src_node_id, dst_node_id] = *ids;

    auto src_it = m_gpu_name_cache.find(src_node_id);
    if(src_it != m_gpu_name_cache.end())
    {
        return src_it->second.empty() ? fmt::format("GPU {}", src_node_id)
                                      : src_it->second;
    }

    auto dst_it = m_gpu_name_cache.find(dst_node_id);
    if(dst_it != m_gpu_name_cache.end())
    {
        return dst_it->second.empty() ? fmt::format("GPU {}", dst_node_id)
                                      : dst_it->second;
    }

    return "GPU";
}

template <typename AgentManagerT, typename OutputRegistryT>
void
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::write_text_output(
    std::ostream& out) const
{
    out << "==" << m_pid << "== Unified Memory profiling result:\n";

    for(const auto& [device_id, summary] : m_data.devices)
    {
        out << " Device \"" << summary.device_name << " (" << device_id << ")\"\n";
        out << "    Count  Avg Size  Min Size  Max Size  Total Size  Total Time    "
               "Bandwidth  Name\n";

        auto print_stats = [&](const migration_stats& stats, const char* name) {
            if(stats.count > 0)
            {
                out << std::setw(9) << stats.count << "  " << std::setw(8)
                    << detail::format_size(static_cast<uint64_t>(stats.avg_size_bytes()))
                    << "  " << std::setw(8) << detail::format_size(stats.min_size_bytes)
                    << "  " << std::setw(8) << detail::format_size(stats.max_size_bytes)
                    << "  " << std::setw(10)
                    << detail::format_size(stats.total_size_bytes) << "  "
                    << std::setw(11) << detail::format_time(stats.total_time_ns) << "  "
                    << std::setw(9) << std::fixed << std::setprecision(2)
                    << stats.bandwidth_gbps() << " GB/s  " << name << "\n";
            }
        };

        print_stats(summary.host_to_device, "Host To Device");
        print_stats(summary.device_to_host, "Device To Host");
        print_stats(summary.device_to_device, "Device To Device");

        out << "\n";
    }

    out << " Total Page Faults: " << m_data.total_page_faults << "\n";

    if(m_data.triggers.total() > 0)
    {
        out << "\n Migration Triggers:\n";
        for(const auto& row : detail::kTriggerTable)
        {
            const auto count = m_data.triggers.*(row.member);
            if(count > 0)
                out << fmt::format("   {:<16}{:>10}\n", std::string(row.text_label) + ":",
                                   count);
        }
    }
}

template <typename AgentManagerT, typename OutputRegistryT>
void
unified_memory_processor_t<AgentManagerT, OutputRegistryT>::write_json_output(
    std::ostream& out) const
{
    nlohmann::json root;
    nlohmann::json devices_array = nlohmann::json::array();

    for(const auto& [device_id, summary] : m_data.devices)
    {
        nlohmann::json device;
        device["device_id"]   = device_id;
        device["device_name"] = summary.device_name;

        auto create_migration_json = [](const migration_stats& stats) -> nlohmann::json {
            nlohmann::json obj;
            obj["count"]            = stats.count;
            obj["avg_size_bytes"]   = stats.avg_size_bytes();
            obj["min_size_bytes"]   = (stats.count == 0) ? 0 : stats.min_size_bytes;
            obj["max_size_bytes"]   = stats.max_size_bytes;
            obj["total_size_bytes"] = stats.total_size_bytes;
            obj["total_time_ns"]    = stats.total_time_ns;
            obj["bandwidth_gbps"]   = stats.bandwidth_gbps();
            return obj;
        };

        nlohmann::json migrations;
        migrations["host_to_device"]   = create_migration_json(summary.host_to_device);
        migrations["device_to_host"]   = create_migration_json(summary.device_to_host);
        migrations["device_to_device"] = create_migration_json(summary.device_to_device);

        device["migrations"] = migrations;

        devices_array.push_back(device);
    }

    root["devices"] = devices_array;

    nlohmann::json summary;
    summary["total_page_faults"] = m_data.total_page_faults;
    summary["xnack_enabled"]     = m_data.xnack_enabled;

    nlohmann::json triggers;
    for(const auto& row : detail::kTriggerTable)
        triggers[row.json_key] = m_data.triggers.*(row.member);
    summary["migration_triggers"] = triggers;

    root["summary"] = summary;

    out << root.dump(2);
}

}  // namespace trace_cache
}  // namespace rocprofsys

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

#include "core/trace_cache/unified_memory_processor.hpp"
#include "core/config.hpp"
#include "core/defines.hpp"
#include "logger/debug.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <spdlog/fmt/fmt.h>
#include <sstream>

namespace rocprofsys
{
namespace trace_cache
{

namespace
{
constexpr size_t SRC_AGENT_PREFIX_LEN = 11;
constexpr size_t DST_AGENT_PREFIX_LEN = 11;

std::string
format_size(uint64_t bytes) noexcept
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    if(bytes >= 1024ULL * 1024 * 1024)
        oss << (bytes / (1024.0 * 1024 * 1024)) << "GB";
    else if(bytes >= 1024ULL * 1024)
        oss << (bytes / (1024.0 * 1024)) << "MB";
    else if(bytes >= 1024ULL)
        oss << (bytes / 1024.0) << "KB";
    else
        oss << bytes << "B";

    return oss.str();
}

std::string
generate_unified_memory_output_path(int pid, const std::string& output_dir,
                                    std::string_view ext)
{
    return rocprofsys::get_output_absolute_path("unified_memory", ext,
                                                std::to_string(pid), output_dir);
}

std::string
format_time(uint64_t nanoseconds) noexcept
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    if(nanoseconds >= 1000000000ULL)
        oss << (nanoseconds / 1000000000.0) << "s";
    else if(nanoseconds >= 1000000ULL)
        oss << (nanoseconds / 1000000.0) << "ms";
    else if(nanoseconds >= 1000ULL)
        oss << (nanoseconds / 1000.0) << "us";
    else
        oss << nanoseconds << "ns";

    return oss.str();
}

}  // namespace

unified_memory_processor_t::unified_memory_processor_t(
    const std::shared_ptr<metadata_registry>& metadata,
    const std::shared_ptr<agent_manager>& agent_mgr, int pid,
    const std::string& output_dir, output_file_registry& output_registry)
: processor_t<unified_memory_processor_t>()
, m_metadata(metadata)
, m_agent_manager(agent_mgr)
, m_pid(pid)
, m_output_dir(output_dir)
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

void
unified_memory_processor_t::prepare_for_processing()
{
    LOG_DEBUG("Preparing unified memory processor for processing");

    if(!m_data.xnack_enabled)
    {
        LOG_WARNING("HSA_XNACK is not set to 1. Unified memory profiling may show "
                    "limited data. Set HSA_XNACK=1 for page fault-driven migration.");
    }
}

void
unified_memory_processor_t::finalize_processing()
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
        generate_unified_memory_output_path(m_pid, m_output_dir, "txt");
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
        generate_unified_memory_output_path(m_pid, m_output_dir, "json");
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

void
unified_memory_processor_t::handle(const kfd_sample& sample)
{
    if(sample.category == "rocm_kfd_page_migrate")
    {
        auto agent_ids = parse_agent_ids_from_args(sample.args_str);
        if(!agent_ids.has_value())
        {
            LOG_TRACE("Failed to parse agent IDs from KFD page migration event");
            return;
        }

        auto [src_label, dst_label] = agent_ids.value();
        auto direction              = classify_direction(src_label, dst_label);

        uint32_t device_id = sample.device_id;
        if(m_data.devices.find(device_id) == m_data.devices.end())
        {
            device_migration_summary summary;
            summary.device_id = device_id;

            std::string cpu_name = fmt::format("CPU {}", device_id);
            try
            {
                const auto& cpu_agent = m_agent_manager->get_agent_by_type_index(
                    device_id, static_cast<agent_type>(sample.device_type));
                if(!cpu_agent.name.empty()) cpu_name = cpu_agent.name;
            } catch(const std::out_of_range& e)
            {
                LOG_DEBUG("Could not resolve CPU agent for device_id={}: {}", device_id,
                          e.what());
            }

            std::string gpu_name = extract_gpu_name(src_label, dst_label);

            summary.device_name = fmt::format("{} (via {})", gpu_name, cpu_name);

            m_data.devices[device_id] = summary;
        }

        auto& device_summary = m_data.devices[device_id];

        uint64_t size_bytes  = static_cast<uint64_t>(sample.value);
        uint64_t duration_ns = sample.end_timestamp - sample.start_timestamp;

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

        auto trigger = classify_trigger(sample.name);
        switch(trigger)
        {
            case migration_trigger::GPU_PAGE_FAULT:
                m_data.triggers.gpu_page_fault++;
                break;
            case migration_trigger::CPU_PAGE_FAULT:
                m_data.triggers.cpu_page_fault++;
                break;
            case migration_trigger::PREFETCH: m_data.triggers.prefetch++; break;
            case migration_trigger::TTM_EVICTION: m_data.triggers.ttm_eviction++; break;
            case migration_trigger::UNKNOWN: m_data.triggers.unknown++; break;
        }
    }
    else if(sample.category == "rocm_kfd_page_fault")
    {
        uint32_t agent_id = sample.device_id;
        bool     is_read  = is_read_fault(sample.name);

        m_data.faults_by_agent[agent_id].add_fault(is_read);
        m_data.total_page_faults++;
    }
}

unified_memory_processor_t::migration_direction
unified_memory_processor_t::classify_direction(const std::string& src_label,
                                               const std::string& dst_label) const
{
    uint32_t src_node_id = 0;
    uint32_t dst_node_id = 0;

    try
    {
        src_node_id = std::stoul(src_label);
        dst_node_id = std::stoul(dst_label);
    } catch(const std::exception&)
    {
        LOG_TRACE("Failed to parse node IDs from labels: src='{}', dst='{}'", src_label,
                  dst_label);
        return migration_direction::UNKNOWN;
    }

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

unified_memory_processor_t::migration_trigger
unified_memory_processor_t::classify_trigger(const std::string& name) const
{
    if(name == "PAGE_MIGRATE_PAGEFAULT_GPU") return migration_trigger::GPU_PAGE_FAULT;
    if(name == "PAGE_MIGRATE_PAGEFAULT_CPU") return migration_trigger::CPU_PAGE_FAULT;
    if(name == "PAGE_MIGRATE_PREFETCH") return migration_trigger::PREFETCH;
    if(name == "PAGE_MIGRATE_TTM_EVICTION") return migration_trigger::TTM_EVICTION;
    return migration_trigger::UNKNOWN;
}

bool
unified_memory_processor_t::is_read_fault(const std::string& name) const
{
    return name.find("Read") != std::string::npos ||
           name.find("READ") != std::string::npos;
}

std::optional<std::pair<std::string, std::string>>
unified_memory_processor_t::parse_agent_ids_from_args(const std::string& args_str) const
{
    std::string src_agent, dst_agent;

    size_t src_pos = args_str.find("src_agent;;");
    if(src_pos != std::string::npos)
    {
        src_pos += SRC_AGENT_PREFIX_LEN;
        size_t end_pos = args_str.find(";;", src_pos);
        if(end_pos != std::string::npos)
        {
            src_agent = args_str.substr(src_pos, end_pos - src_pos);
        }
    }

    size_t dst_pos = args_str.find("dst_agent;;");
    if(dst_pos != std::string::npos)
    {
        dst_pos += DST_AGENT_PREFIX_LEN;
        size_t end_pos = args_str.find(";;", dst_pos);
        if(end_pos != std::string::npos)
        {
            dst_agent = args_str.substr(dst_pos, end_pos - dst_pos);
        }
    }

    if(src_agent.empty() || dst_agent.empty())
    {
        return std::nullopt;
    }

    return std::make_pair(src_agent, dst_agent);
}

std::string
unified_memory_processor_t::extract_gpu_name(const std::string& src_label,
                                             const std::string& dst_label) const
{
    uint32_t src_node_id = 0;
    uint32_t dst_node_id = 0;

    try
    {
        src_node_id = std::stoul(src_label);
        dst_node_id = std::stoul(dst_label);
    } catch(const std::exception&)
    {
        return "GPU";
    }

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

void
unified_memory_processor_t::write_text_output(std::ostream& out)
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
                    << format_size(static_cast<uint64_t>(stats.avg_size_bytes())) << "  "
                    << std::setw(8) << format_size(stats.min_size_bytes) << "  "
                    << std::setw(8) << format_size(stats.max_size_bytes) << "  "
                    << std::setw(10) << format_size(stats.total_size_bytes) << "  "
                    << std::setw(11) << format_time(stats.total_time_ns) << "  "
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
        if(m_data.triggers.gpu_page_fault > 0)
            out << "   GPU page fault: " << std::setw(10)
                << m_data.triggers.gpu_page_fault << "\n";
        if(m_data.triggers.cpu_page_fault > 0)
            out << "   CPU page fault: " << std::setw(10)
                << m_data.triggers.cpu_page_fault << "\n";
        if(m_data.triggers.prefetch > 0)
            out << "   Prefetch:       " << std::setw(10) << m_data.triggers.prefetch
                << "\n";
        if(m_data.triggers.ttm_eviction > 0)
            out << "   TTM eviction:   " << std::setw(10) << m_data.triggers.ttm_eviction
                << "\n";
        if(m_data.triggers.unknown > 0)
            out << "   Unknown:        " << std::setw(10) << m_data.triggers.unknown
                << "\n";
    }
}

void
unified_memory_processor_t::write_json_output(std::ostream& out)
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
    triggers["gpu_page_fault"]    = m_data.triggers.gpu_page_fault;
    triggers["cpu_page_fault"]    = m_data.triggers.cpu_page_fault;
    triggers["prefetch"]          = m_data.triggers.prefetch;
    triggers["ttm_eviction"]      = m_data.triggers.ttm_eviction;
    triggers["unknown"]           = m_data.triggers.unknown;
    summary["migration_triggers"] = triggers;

    root["summary"] = summary;

    out << root.dump(2);  // Pretty print with 2-space indent
}

}  // namespace trace_cache
}  // namespace rocprofsys

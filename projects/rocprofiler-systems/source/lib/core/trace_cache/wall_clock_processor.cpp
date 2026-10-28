// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/wall_clock_processor.hpp"
#include "common/path.hpp"
#include "core/agent_manager.hpp"
#include "core/demangler.hpp"
#include "core/output/output_summary.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/wall_clock_report.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <utility>

namespace rocprofsys::trace_cache
{
namespace
{
template <typename Sample>
std::optional<wall_clock_interval>
make_interval(const Sample& sample, std::string name)
{
    if(sample.end_timestamp < sample.start_timestamp)
    {
        LOG_WARNING("wall_clock: skipping '{}' (thread {}), it ends at {} before it "
                    "begins at {}",
                    name, sample.thread_id, sample.end_timestamp, sample.start_timestamp);
        return std::nullopt;
    }
    return wall_clock_interval{ .thread_id      = sample.thread_id,
                                .name           = std::move(name),
                                .correlation_id = sample.correlation_id_internal,
                                .begin = wall_clock_nanoseconds{ sample.start_timestamp },
                                .end   = wall_clock_nanoseconds{ sample.end_timestamp } };
}

bool
is_application_worker_thread(const thread_info& info)
{
    return info.index_data.has_value() && !info.is_offset &&
           info.index_data->sequent_value != 0 && info.get_stop() > info.get_start();
}

void
write_file(const std::string& filepath, std::string_view content, pid_t pid)
{
    std::ofstream file;
    if(!path::create_parent_dirs_and_open_ofstream(file, filepath))
    {
        LOG_ERROR("Failed to open wall_clock output: {}", filepath);
        return;
    }
    file << content;
    file.close();
    if(file.fail())
    {
        LOG_ERROR("Failed to write wall_clock output: {}", filepath);
        return;
    }
    output::registry::instance().register_file(filepath, pid);
    LOG_INFO("wall_clock output written to: {}", filepath);
}
}  // namespace

wall_clock_processor_t::wall_clock_processor_t(
    std::shared_ptr<metadata_registry> metadata, std::shared_ptr<agent_manager> agents,
    pid_t pid, wall_clock_options options)
: m_metadata{ std::move(metadata) }
, m_pid{ pid }
, m_agents{ std::move(agents) }
, m_options{ std::move(options) }
, m_report{ m_options.settings }
{}

void
wall_clock_processor_t::handle(const region_sample& sample)
{
    auto interval = make_interval(sample, std::string{ sample.name });
    if(interval.has_value())
    {
        m_report.add_region(std::move(*interval));
    }
}

void
wall_clock_processor_t::handle(const kernel_dispatch_sample& sample)
{
    auto const symbol = m_metadata->get_kernel_symbol(sample.kernel_id);
    if(!symbol.has_value())
    {
        throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
    }
    auto interval =
        make_interval(sample, rocprofsys::utility::demangle(symbol->kernel_name));
    if(interval.has_value())
    {
        m_report.add_gpu_activity(std::move(*interval));
    }
}

void
wall_clock_processor_t::handle(const memory_copy_sample& sample)
{
    auto const& src = m_agents->get_agent_by_handle(sample.src_agent_id_handle);
    auto const& dst = m_agents->get_agent_by_handle(sample.dst_agent_id_handle);
    auto        interval =
        make_interval(sample, fmt::format("memory_copy: {} -> {}", src.logical_node_id,
                                          dst.logical_node_id));
    if(interval.has_value())
    {
        m_report.add_gpu_activity(std::move(*interval));
    }
}

void
wall_clock_processor_t::handle(const scratch_memory_sample& sample)
{
    auto interval = make_interval(sample, std::string{ sample.name });
    if(interval.has_value())
    {
        m_report.add_gpu_activity(std::move(*interval));
    }
}

void
wall_clock_processor_t::add_thread_lifetimes()
{
    for(const auto& registered : m_metadata->get_thread_info_list())
    {
        const auto& info =
            thread_info::get(static_cast<std::int64_t>(registered.thread_id), SystemTID);
        if(!info.has_value() || !is_application_worker_thread(*info))
        {
            continue;
        }
        m_report.add_region({ .thread_id      = registered.thread_id,
                              .name           = "start_thread",
                              .correlation_id = 0,
                              .begin = wall_clock_nanoseconds{ info->get_start() },
                              .end   = wall_clock_nanoseconds{ info->get_stop() } });
    }
}

void
wall_clock_processor_t::finalize_processing()
{
    if(m_report.empty())
    {
        LOG_DEBUG("No regions or kernels recorded for PID {}; no wall_clock report",
                  m_pid);
        return;
    }
    add_thread_lifetimes();

    if(m_options.cout)
    {
        std::cout << m_report.text();
    }
    if(m_options.text_path)
    {
        write_file(*m_options.text_path, m_report.text(), m_pid);
    }
    if(m_options.json_path)
    {
        write_file(*m_options.json_path, m_report.json(), m_pid);
    }
}

}  // namespace rocprofsys::trace_cache

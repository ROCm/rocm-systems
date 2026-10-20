// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/wall_clock_processor.hpp"
#include "core/agent_manager.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/output/output_summary.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/trace_cache/wall_clock_report.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <ostream>
#include <sstream>
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
wall_clock_interval
make_interval(const Sample& sample, std::string name)
{
    return { .thread_id      = sample.thread_id,
             .name           = std::move(name),
             .correlation_id = sample.correlation_id_internal,
             .begin          = wall_clock_nanoseconds{ sample.start_timestamp },
             .end            = wall_clock_nanoseconds{ sample.end_timestamp } };
}

/// True for a worker thread created by the application that has already stopped. The
/// main thread, internal threads and threads still running are excluded.
bool
is_finished_application_thread(const thread_info& info)
{
    return info.index_data.has_value() && !info.is_offset &&
           info.index_data->sequent_value != 0 && info.get_stop() > info.get_start();
}
}  // namespace

wall_clock_processor_t::wall_clock_processor_t(
    std::shared_ptr<metadata_registry> metadata, std::shared_ptr<agent_manager> agents,
    pid_t pid, wall_clock_options options)
: m_metadata{ std::move(metadata) }
, m_pid{ pid }
, m_agents{ std::move(agents) }
, m_options{ options }
, m_report{ options.layout }
{}

void
wall_clock_processor_t::handle(const region_sample& sample)
{
    m_report.add_region(make_interval(sample, std::string{ sample.name }));
}

void
wall_clock_processor_t::handle(const kernel_dispatch_sample& sample)
{
    auto const symbol = m_metadata->get_kernel_symbol(sample.kernel_id);
    if(!symbol.has_value())
    {
        throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
    }
    m_report.add_gpu_activity(
        make_interval(sample, rocprofsys::utility::demangle(symbol->kernel_name)));
}

void
wall_clock_processor_t::handle(const memory_copy_sample& sample)
{
    auto const& src = m_agents->get_agent_by_handle(sample.src_agent_id_handle);
    auto const& dst = m_agents->get_agent_by_handle(sample.dst_agent_id_handle);
    m_report.add_gpu_activity(
        make_interval(sample, fmt::format("memory_copy: {} -> {}", src.logical_node_id,
                                          dst.logical_node_id)));
}

void
wall_clock_processor_t::handle(const scratch_memory_sample& sample)
{
    m_report.add_gpu_activity(make_interval(sample, std::string{ sample.name }));
}

/// Threads created by the application are timed for their whole lifetime as
/// start_thread, as in the timemory report. They are not cached as regions, so their
/// lifetimes come from the thread table, as for the rocpd thread info.
void
wall_clock_processor_t::add_thread_lifetimes()
{
    for(const auto& registered : m_metadata->get_thread_info_list())
    {
        const auto& info =
            thread_info::get(static_cast<std::int64_t>(registered.thread_id), SystemTID);
        if(!info.has_value() || !is_finished_application_thread(*info))
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
wall_clock_processor_t::write_file(std::string_view type, const auto& write)
{
    auto const path = get_output_absolute_path("wall_clock", type, m_options.suffix, "");
    std::filesystem::create_directories(std::filesystem::path{ path }.parent_path());
    std::ofstream file{ path };
    if(!file.is_open())
    {
        LOG_ERROR("Failed to open wall_clock output: {}", path);
        return;
    }
    write(file);
    file.close();
    if(file.fail())
    {
        LOG_ERROR("Failed to write wall_clock output: {}", path);
        return;
    }
    output::registry::instance().register_file(path, m_pid);
    LOG_INFO("wall_clock output written to: {}", path);
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
        std::ostringstream table;
        m_report.write_text(table);
        LOG_INFO("\n{}", table.str());
    }
    if(m_options.text)
    {
        write_file("txt", [this](std::ostream& out) { m_report.write_text(out); });
    }
    if(m_options.json)
    {
        write_file("json", [this](std::ostream& out) {
            m_report.write_json(out, m_options.rank, m_options.mpi_size);
        });
    }
}

}  // namespace rocprofsys::trace_cache

// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/wall_clock_processor.hpp"
#include "common/env_vars.hpp"
#include "common/path.hpp"
#include "core/agent_manager.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/output/output_summary.hpp"
#include "core/trace_cache/data_types.hpp"
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
constexpr std::string_view k_thread_lifetime_region = "start_thread";

template <typename Sample>
[[nodiscard("a dropped interval loses the sample")]] std::optional<wall_clock_interval>
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

[[nodiscard]] bool
is_application_worker_thread(const thread_info& info)
{
    return info.index_data.has_value() && !info.is_offset &&
           info.index_data->sequent_value != k_main_thread_index &&
           info.get_stop() > info.get_start();
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

[[nodiscard]] wall_clock_settings
make_wall_clock_settings()
{
    constexpr wall_clock_settings k_defaults{};
    auto const                    units =
        config::get_setting_value<std::string>(env_vars::TIMING_UNITS).value_or("");
    auto const unit = find_wall_clock_unit(units);
    if(!unit.has_value())
    {
        LOG_WARNING("Unknown {} '{}'; the wall_clock report uses {}",
                    env_vars::TIMING_UNITS, units, k_defaults.unit.name);
    }
    // A negative precision is timemory's "use the default" value
    auto const precision =
        config::get_setting_value<std::int16_t>(env_vars::TIMING_PRECISION)
            .value_or(k_defaults.precision);
    return wall_clock_settings{
        .flat = config::get_setting_value<bool>(env_vars::FLAT_PROFILE)
                    .value_or(k_defaults.flat),
        .timeline = config::get_setting_value<bool>(env_vars::TIMELINE_PROFILE)
                        .value_or(k_defaults.timeline),
        .collapse_threads = config::get_setting_value<bool>(env_vars::COLLAPSE_THREADS)
                                .value_or(k_defaults.collapse_threads),
        .max_depth = config::get_setting_value<std::uint16_t>(env_vars::MAX_DEPTH)
                         .value_or(k_defaults.max_depth),
        .max_width = config::get_setting_value<std::int32_t>(env_vars::MAX_WIDTH)
                         .value_or(k_defaults.max_width),
        .unit      = unit.value_or(k_defaults.unit),
        .precision = precision >= 0 ? precision : k_defaults.precision,
        .rank      = config::get_mpi_rank_from_env().value_or(k_defaults.rank),
        .mpi_size  = config::get_mpi_world_size_from_env().value_or(k_defaults.mpi_size),
    };
}
}  // namespace

wall_clock_options
make_wall_clock_options(const data::processor_config_t& process)
{
    auto const settings     = make_wall_clock_settings();
    auto const is_displayed = settings.max_depth > 0;
    auto const should_write_files =
        is_displayed &&
        config::get_setting_value<bool>(env_vars::FILE_OUTPUT).value_or(true);
    auto const is_root_process = process._pid == process._ppid;
    auto const suffix          = is_root_process ? config::get_output_process_suffix()
                                                 : std::to_string(process._pid);
    // Without the suffix (ROCPROFSYS_USE_PID=OFF) child processes would share one file
    auto const basename       = is_root_process || settings::use_output_suffix()
                                    ? std::string{ "wall_clock" }
                                    : fmt::format("wall_clock-{}", process._pid);
    auto const output_path_if = [&](bool enabled, std::string_view extension) {
        return enabled ? std::optional{ get_output_absolute_path(basename, extension,
                                                                 suffix, "") }
                       : std::nullopt;
    };
    return wall_clock_options{
        .settings  = settings,
        .text_path = output_path_if(
            should_write_files &&
                config::get_setting_value<bool>(env_vars::TEXT_OUTPUT).value_or(true),
            "txt"),
        .json_path = output_path_if(
            should_write_files &&
                config::get_setting_value<bool>(env_vars::JSON_OUTPUT).value_or(true),
            "json"),
        .should_print_to_stdout =
            is_displayed &&
            config::get_setting_value<bool>(env_vars::COUT_OUTPUT).value_or(false),
    };
}

wall_clock_processor_t::wall_clock_processor_t(
    std::shared_ptr<metadata_registry> metadata, std::shared_ptr<agent_manager> agents,
    pid_t pid, wall_clock_options options)
: m_metadata{ std::move(metadata) }
, m_pid{ pid }
, m_agents{ std::move(agents) }
, m_options{ std::move(options) }
, m_report{ m_options.settings }
{}

template <typename Sample>
void
wall_clock_processor_t::add_region(const Sample& sample, std::string name)
{
    auto const interval = make_interval(sample, std::move(name));
    if(interval.has_value())
    {
        m_report.add_region(*interval);
    }
}

template <typename Sample>
void
wall_clock_processor_t::add_gpu_activity(const Sample& sample, std::string name)
{
    auto const interval = make_interval(sample, std::move(name));
    if(interval.has_value())
    {
        m_report.add_gpu_activity(*interval);
    }
}

const std::string&
wall_clock_processor_t::kernel_name(std::uint64_t kernel_id)
{
    auto const cached = m_kernel_names.find(kernel_id);
    if(cached != m_kernel_names.end())
    {
        return cached->second;
    }
    auto const symbol = m_metadata->get_kernel_symbol(kernel_id);
    if(!symbol.has_value())
    {
        throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
    }
    return m_kernel_names
        .emplace(kernel_id, rocprofsys::utility::demangle(symbol->kernel_name))
        .first->second;
}

void
wall_clock_processor_t::handle(const region_sample& sample)
{
    add_region(sample, std::string{ sample.name });
}

void
wall_clock_processor_t::handle(const kernel_dispatch_sample& sample)
{
    add_gpu_activity(sample, kernel_name(sample.kernel_id));
}

void
wall_clock_processor_t::handle(const memory_copy_sample& sample)
{
    auto const& src = m_agents->get_agent_by_handle(sample.src_agent_id_handle);
    auto const& dst = m_agents->get_agent_by_handle(sample.dst_agent_id_handle);
    add_gpu_activity(sample, fmt::format("memory_copy: {} -> {}", src.logical_node_id,
                                         dst.logical_node_id));
}

void
wall_clock_processor_t::handle(const scratch_memory_sample& sample)
{
    add_gpu_activity(sample, std::string{ sample.name });
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
                              .name           = std::string{ k_thread_lifetime_region },
                              .correlation_id = k_no_correlation,
                              .begin = wall_clock_nanoseconds{ info->get_start() },
                              .end   = wall_clock_nanoseconds{ info->get_stop() } });
    }
}

void
wall_clock_processor_t::finalize_processing()
{
    add_thread_lifetimes();
    if(m_report.empty())
    {
        LOG_DEBUG("No regions, kernels or worker threads recorded for PID {}; no "
                  "wall_clock report",
                  m_pid);
        return;
    }

    if(m_options.should_print_to_stdout || m_options.text_path)
    {
        auto const table = m_report.text();
        if(m_options.should_print_to_stdout)
        {
            std::cout << table;
        }
        if(m_options.text_path)
        {
            write_file(*m_options.text_path, table, m_pid);
        }
    }
    if(m_options.json_path)
    {
        write_file(*m_options.json_path, m_report.json(), m_pid);
    }
}

}  // namespace rocprofsys::trace_cache

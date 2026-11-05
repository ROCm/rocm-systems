// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/procfs/backend.hpp"
#include "common/clocks.hpp"
#include "common/units/data_size.hpp"
#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/collection/record.hpp"
#include "core/collection/scheduler.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "library/pmc/collectors/common/settings.hpp"
#include "library/pmc/collectors/cpu/cpu_traits.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/device_providers/procfs/provider.hpp"
#include "logger/debug.hpp"
#include "rocprofiler-systems/categories.h"
#include <fmt/format.h>
#include <utility>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::collection::sources
{
// The process-level metrics are written once, under the first socket.
template <typename Clock = std::chrono::steady_clock>
class cpu_source final : public source_interface
{
public:
    cpu_source(std::chrono::nanoseconds interval, scheduler_factory<Clock> make_scheduler)
    : m_interval(interval)
    , m_scheduler(make_scheduler("cpu"))
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return "cpu"; }

    void describe(source_id src_id, sink_interface& out) override
    {
        m_id  = src_id;
        m_out = &out;

        m_provider = provider_factory_t::create();
        auto entries =
            traits_t::template enumerate_devices<pmc::collectors::settings_policy>(
                m_provider);
        m_enabled = pmc::collectors::settings_policy::get_cpu_enabled_metrics();

        bool first_socket = true;
        for(auto& entry : entries)
        {
            m_sockets.push_back(make_socket(std::move(entry), first_socket));
            first_socket = false;
        }
    }

    void start() override
    {
        m_scheduler->start(m_interval,
                           [this](std::uint64_t timeline_ns) { poll_once(timeline_ns); });
    }

    void pause() override
    {
        m_gap_begin_ns = m_scheduler->pause();
        m_paused       = true;
    }

    void resume() override
    {
        write_gap();
        m_scheduler->resume();
    }

    void stop() override
    {
        m_scheduler->stop();
        if(m_paused)
        {
            write_gap();
        }
        if(m_provider)
        {
            m_provider->shutdown();
            m_provider.reset();
        }
    }

private:
    using provider_factory_t = pmc::device_providers::procfs::provider_factory<
        backends::procfs::backend_factory>;
    using provider_t       = provider_factory_t::provider_t;
    using traits_t         = pmc::collectors::cpu::cpu_traits<provider_t>;
    using device_entry_t   = traits_t::device_entry;
    using enabled_t        = pmc::collectors::cpu::enabled_metrics;
    using process_t        = pmc::collectors::cpu::process_metrics;
    using metrics_t        = pmc::collectors::cpu::metrics;
    using process_reader_t = double (*)(const process_t&);

    struct socket_stream
    {
        device_entry_t entry;
        schema_id      schema = 0;
        enabled_t      effective{};
        std::vector<process_reader_t>
            process_readers;               // one per process column, first socket only
        std::vector<std::size_t> cpu_ids;  // sorted, the cores of the per-core columns
        std::size_t              freq_first = 0;
        std::size_t              load_first = 0;
        bool                     has_freq   = false;
        bool                     has_load   = false;
        bool                     failed     = false;
        std::vector<double>      row;
    };

    struct process_metric
    {
        bool (*enabled)(const enabled_t&);
        process_reader_t read;
        std::uint32_t    category_id;
        const char*      pmc_name;
        const char*      perfetto_name;
        const char*      unit;
        const char*      symbol;
        const char*      description;
    };

    static double to_megabytes(std::int64_t bytes_value) noexcept
    {
        using rocprofsys::common::units::bytes;
        using rocprofsys::common::units::data_size_cast;
        using rocprofsys::common::units::megabytes;
        return data_size_cast<megabytes>(bytes{ static_cast<double>(bytes_value) })
            .count();
    }

    static double to_seconds(std::int64_t microseconds) noexcept
    {
        return std::chrono::duration<double>{
            std::chrono::microseconds{ microseconds }
        }.count();
    }

    static std::vector<process_metric> process_metrics()
    {
        return {
            { .enabled =
                  [](const enabled_t& enabled) { return enabled.bits.page_rss != 0; },
              .read = [](const process_t& info) { return to_megabytes(info.page_rss); },
              .category_id   = ROCPROFSYS_CATEGORY_PROCESS_PAGE,
              .pmc_name      = trait::name<category::process_page>::value,
              .perfetto_name = "CPU Memory Usage (S)",
              .unit          = "MB",
              .symbol        = "Page RSS",
              .description   = "Process Physical Memory (RSS)" },
            { .enabled =
                  [](const enabled_t& enabled) { return enabled.bits.virt_mem != 0; },
              .read = [](const process_t& info) { return to_megabytes(info.virt_mem); },
              .category_id   = ROCPROFSYS_CATEGORY_PROCESS_VIRT,
              .pmc_name      = trait::name<category::process_virt>::value,
              .perfetto_name = "CPU Virtual Memory (S)",
              .unit          = "MB",
              .symbol        = "Virt Mem",
              .description   = "Process Virtual Memory" },
            { .enabled =
                  [](const enabled_t& enabled) { return enabled.bits.peak_rss != 0; },
              .read = [](const process_t& info) { return to_megabytes(info.peak_rss); },
              .category_id   = ROCPROFSYS_CATEGORY_PROCESS_PEAK,
              .pmc_name      = trait::name<category::process_peak>::value,
              .perfetto_name = "CPU Peak Memory (S)",
              .unit          = "MB",
              .symbol        = "Peak RSS",
              .description   = "Process Peak Memory (HWM)" },
            { .enabled =
                  [](const enabled_t& enabled) { return enabled.bits.ctx_switches != 0; },
              .read =
                  [](const process_t& info) {
                      return static_cast<double>(info.context_switches);
                  },
              .category_id   = ROCPROFSYS_CATEGORY_PROCESS_CONTEXT_SWITCH,
              .pmc_name      = trait::name<category::process_context_switch>::value,
              .perfetto_name = "CPU Context Switches (S)",
              .unit          = "count",
              .symbol        = "Ctx Switches",
              .description   = "Context Switches" },
            { .enabled =
                  [](const enabled_t& enabled) { return enabled.bits.page_faults != 0; },
              .read =
                  [](const process_t& info) {
                      return static_cast<double>(info.page_faults);
                  },
              .category_id   = ROCPROFSYS_CATEGORY_PROCESS_PAGE_FAULT,
              .pmc_name      = trait::name<category::process_page_fault>::value,
              .perfetto_name = "CPU Page Faults (S)",
              .unit          = "count",
              .symbol        = "Page Faults",
              .description   = "Page Faults" },
            { .enabled =
                  [](const enabled_t& enabled) { return enabled.bits.user_time != 0; },
              .read =
                  [](const process_t& info) { return to_seconds(info.user_mode_time); },
              .category_id   = ROCPROFSYS_CATEGORY_PROCESS_USER_MODE_TIME,
              .pmc_name      = trait::name<category::process_user_mode_time>::value,
              .perfetto_name = "CPU User Time (S)",
              .unit          = "sec",
              .symbol        = "User Time",
              .description   = "Process CPU Time in User Mode" },
            { .enabled =
                  [](const enabled_t& enabled) { return enabled.bits.kernel_time != 0; },
              .read =
                  [](const process_t& info) { return to_seconds(info.kernel_mode_time); },
              .category_id   = ROCPROFSYS_CATEGORY_PROCESS_KERNEL_MODE_TIME,
              .pmc_name      = trait::name<category::process_kernel_mode_time>::value,
              .perfetto_name = "CPU Kernel Time (S)",
              .unit          = "sec",
              .symbol        = "Kernel Time",
              .description   = "Process CPU Time in Kernel Mode" },
        };
    }

    static column_spec core_column(std::size_t socket, std::size_t cpu_id, bool frequency)
    {
        const std::string base = frequency ? trait::name<category::cpu_freq>::value
                                           : trait::name<category::cpu_load>::value;
        const auto        pmc  = fmt::format("{} [{}] Core [{}]", base, socket, cpu_id);

        column_spec column;
        column.name = pmc;
        column.description =
            frequency ? "CPU Core Frequency" : "CPU Core Load Percentage";
        column.unit     = frequency ? "MHz" : "%";
        column.pmc_name = pmc;
        column.perfetto_track_name =
            fmt::format("CPU [{}] Core [{}] {} (S)", socket, cpu_id,
                        frequency ? "Frequency" : "Load");
        column.rocpd_track_name = pmc;
        column.category_id =
            frequency ? ROCPROFSYS_CATEGORY_CPU_FREQ : ROCPROFSYS_CATEGORY_CPU_LOAD;
        return column;
    }

    static void add_process_columns(schema& description, socket_stream& socket)
    {
        for(const auto& metric : process_metrics())
        {
            if(!metric.enabled(socket.effective))
            {
                continue;
            }
            column_spec column;
            column.name                = metric.symbol;
            column.description         = metric.description;
            column.unit                = metric.unit;
            column.pmc_name            = metric.pmc_name;
            column.perfetto_track_name = metric.perfetto_name;
            column.rocpd_track_name    = metric.pmc_name;
            column.category_id         = metric.category_id;
            description.columns.push_back(std::move(column));
            socket.process_readers.push_back(metric.read);
        }
    }

    socket_stream make_socket(device_entry_t entry, bool first_socket)
    {
        socket_stream socket;
        socket.entry           = std::move(entry);
        socket.effective.value = m_enabled.value & socket.entry.supported_metrics.value;
        const auto socket_id   = socket.entry.device->get_index();

        schema description;
        description.name        = fmt::format("CPU [{}]", socket_id);
        description.target_arch = "CPU";
        description.agent       = agent_type::cpu;
        description.device      = static_cast<std::uint32_t>(socket_id);

        if(first_socket)
        {
            add_process_columns(description, socket);
        }

        const auto& cores = socket.entry.device->get_monitored_cpus();
        socket.cpu_ids.assign(cores.begin(), cores.end());
        add_core_columns(description, socket, socket_id, true);
        add_core_columns(description, socket, socket_id, false);

        socket.row.assign(description.columns.size(),
                          std::numeric_limits<double>::quiet_NaN());
        socket.schema = m_out->register_schema(m_id, description);
        return socket;
    }

    static void add_core_columns(schema& description, socket_stream& socket,
                                 std::size_t socket_id, bool frequency)
    {
        const bool enabled = frequency ? socket.effective.bits.frequency != 0
                                       : socket.effective.bits.load != 0;
        if(!enabled || socket.cpu_ids.empty())
        {
            return;
        }

        (frequency ? socket.freq_first : socket.load_first) = description.columns.size();
        (frequency ? socket.has_freq : socket.has_load)     = true;
        for(const auto cpu_id : socket.cpu_ids)
        {
            description.columns.push_back(core_column(socket_id, cpu_id, frequency));
        }
    }

    void poll_once(std::uint64_t timeline_ns)
    {
        m_timestamp = timeline_ns;
        for(auto& socket : m_sockets)
        {
            if(!socket.failed)
            {
                poll_socket(socket);
            }
        }
    }

    void write_gap()
    {
        m_out->write_gap({ .source   = m_id,
                           .begin_ns = m_gap_begin_ns,
                           .end_ns   = clocks::timeline::now_ns() });
        m_paused = false;
    }

    void poll_socket(socket_stream& socket)
    {
        try
        {
            const auto reading = socket.entry.device->get_cpu_metrics(m_enabled);
            fill_row(socket, reading);
        } catch(const std::exception& error)
        {
            LOG_ERROR(
                "Reading metrics failed for CPU device {}. Error: {}. Disabling device!",
                socket.entry.device->get_index(), error.what());
            socket.failed = true;
            return;
        }

        m_out->write_batch({ .source    = m_id,
                             .schema    = socket.schema,
                             .timestamp = m_timestamp,
                             .values    = socket.row });
    }

    static void fill_row(socket_stream& socket, const metrics_t& reading)
    {
        std::fill(socket.row.begin(), socket.row.end(),
                  std::numeric_limits<double>::quiet_NaN());

        for(std::size_t idx = 0; idx < socket.process_readers.size(); ++idx)
        {
            socket.row[idx] = socket.process_readers[idx](reading.process_data);
        }

        for(const auto& core : reading.cpu_data)
        {
            const auto found = std::lower_bound(socket.cpu_ids.begin(),
                                                socket.cpu_ids.end(), core.cpu_id);
            if(found == socket.cpu_ids.end() || *found != core.cpu_id)
            {
                continue;
            }

            const auto pos = static_cast<std::size_t>(found - socket.cpu_ids.begin());
            if(socket.has_freq)
            {
                socket.row[socket.freq_first + pos] = static_cast<double>(core.frequency);
            }
            if(socket.has_load)
            {
                socket.row[socket.load_first + pos] = core.load;
            }
        }
    }

    std::chrono::nanoseconds          m_interval;
    std::unique_ptr<scheduler<Clock>> m_scheduler;
    std::uint64_t                     m_gap_begin_ns = 0;
    bool                              m_paused       = false;
    std::uint64_t                     m_timestamp    = 0;
    source_id                         m_id           = 0;
    sink_interface*                   m_out          = nullptr;
    enabled_t                         m_enabled{};
    std::shared_ptr<provider_t>       m_provider;
    std::vector<socket_stream>        m_sockets;
};
}  // namespace rocprofsys::collection::sources

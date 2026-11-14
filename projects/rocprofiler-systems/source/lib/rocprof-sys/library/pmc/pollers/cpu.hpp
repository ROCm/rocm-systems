// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "library/pmc/collectors/cpu/cpu_traits.hpp"
#include "library/pmc/pollers/catalog/cpu.hpp"
#include "library/pmc/pollers/row_stream.hpp"
#include "logger/debug.hpp"
#include "policies/pmc/device_policy.hpp"
#include "policies/pmc/settings_policy.hpp"
#include "rocprofiler-systems/categories.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::pollers
{
/** Polls the CPU sockets: one row per socket with its core metrics; the process metrics
 * are written once, under the first socket. */
template <typename Provider, typename Settings>
class cpu final : public collection::source_interface
{
    using traits_t  = pmc::collectors::cpu::cpu_traits<Provider>;
    using enabled_t = catalog::cpu::enabled_t;

    static_assert(policies::pmc::cpu_device<typename traits_t::device_t, enabled_t>);
    static_assert(policies::pmc::cpu_settings<Settings>);

public:
    /** @param provider the procfs provider, or null */
    explicit cpu(std::shared_ptr<Provider> provider)
    : m_provider(std::move(provider))
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return "cpu"; }

    [[nodiscard]] bool describe(collection::sink_interface& out) override
    {
        if(!m_provider)
        {
            return false;
        }
        m_enabled  = traits_t::template get_enabled_metrics<Settings>();
        bool first = true;
        for(auto& entry : traits_t::template enumerate_devices<Settings>(m_provider))
        {
            add_socket(std::move(entry), first, out);
            first = false;
        }
        if(m_sockets.empty())
        {
            release();
            return false;
        }
        return true;
    }

    void poll(std::uint64_t timeline_ns, collection::sink_interface& out) override
    {
        poll_streams(m_sockets, timeline_ns, out, [](socket_stream& socket) {
            fill_row(socket);
            return true;
        });
    }

    void release() override
    {
        m_sockets.clear();
        m_provider.reset();
    }

private:
    /** The columns of one core metric: one per monitored core, from column `first`. */
    struct core_block
    {
        /** The catalog row of the metric. */
        const catalog::cpu::core_row_t* metric = nullptr;
        /** Column of the first monitored core. */
        std::size_t first = 0;
    };

    /** One socket and the row it writes. */
    struct socket_stream
    {
        /** The device and its supported metrics. */
        traits_t::device_entry entry;
        /** The metrics read on each poll (no process metrics after the first socket). */
        enabled_t request{};
        /** Its row; k_no_value where not read. */
        row_stream rows;
        /** Rows of the process columns, which come first. */
        std::vector<const catalog::cpu::process_row_t*> process_rows;
        /** Monitored cores, ascending, as the device reports them. */
        std::vector<std::size_t> cpu_ids;
        /** One per enabled core metric. */
        std::vector<core_block> core_blocks;
    };

    /** Builds and registers the schema of one socket; a socket without columns is
     * skipped. */
    void add_socket(traits_t::device_entry entry, bool first,
                    collection::sink_interface& out)
    {
        socket_stream socket{ .entry        = std::move(entry),
                              .rows         = {},
                              .process_rows = {},
                              .cpu_ids      = {},
                              .core_blocks  = {} };
        enabled_t     effective{};
        effective.value      = m_enabled.value & socket.entry.supported_metrics.value;
        const auto socket_id = socket.entry.device->get_index();

        collection::schema description{
            .agent       = agent_type::cpu,
            .device      = static_cast<std::uint32_t>(socket_id),
            .target_arch = "CPU",
            .event_category =
                std::string{ category_info(ROCPROFSYS_CATEGORY_CPU_FREQ).name },
            .columns = {}
        };
        socket.request = first ? effective : core_metrics_of(effective);
        if(first)
        {
            add_process_columns(description, socket, effective);
        }
        add_core_columns(description, socket, effective);
        if(description.columns.empty())
        {
            LOG_DEBUG("CPU socket {} has no enabled metric, not sampled", socket_id);
            return;
        }
        socket.rows.open(description, out, name(), socket_id);
        m_sockets.push_back(std::move(socket));
    }

    /** `enabled` without the process metrics. */
    [[nodiscard]] static enabled_t core_metrics_of(const enabled_t& enabled)
    {
        enabled_t cores{};
        cores.bits.frequency = enabled.bits.frequency;
        cores.bits.load      = enabled.bits.load;
        return cores;
    }

    /** Adds a column per enabled process metric. */
    static void add_process_columns(collection::schema& description,
                                    socket_stream& socket, const enabled_t& effective)
    {
        for(const auto& metric : catalog::cpu::k_process_metrics)
        {
            if(metric.enabled(effective))
            {
                description.columns.push_back(catalog::cpu::make_process_column(metric));
                socket.process_rows.push_back(&metric);
            }
        }
    }

    /** Adds a column per monitored core for each enabled core metric. */
    static void add_core_columns(collection::schema& description, socket_stream& socket,
                                 const enabled_t& effective)
    {
        const auto& cores = socket.entry.device->get_monitored_cpus();
        socket.cpu_ids.assign(cores.begin(), cores.end());
        const auto socket_id = socket.entry.device->get_index();
        for(const auto& metric : catalog::cpu::k_core_metrics)
        {
            if(!metric.enabled(effective) || socket.cpu_ids.empty())
            {
                continue;
            }
            socket.core_blocks.push_back(
                { .metric = &metric, .first = description.columns.size() });
            for(const auto cpu_id : socket.cpu_ids)
            {
                description.columns.push_back(
                    catalog::cpu::make_core_column(metric, socket_id, cpu_id));
            }
        }
    }

    /** Reads the socket into its row. */
    static void fill_row(socket_stream& socket)
    {
        const auto reading = socket.entry.device->get_cpu_metrics(socket.request);
        std::ranges::fill(socket.rows.row, collection::k_no_value);
        std::ranges::transform(socket.process_rows, socket.rows.row.begin(),
                               [&reading](const catalog::cpu::process_row_t* metric) {
                                   return metric->read(reading.process_data, 0, 0);
                               });
        for(const auto& block : socket.core_blocks)
        {
            fill_cores(socket, block, reading.cpu_data);
        }
    }

    /** Copies one core metric into the row. The device lists its cores in the order of
     * cpu_ids; a core it did not report keeps k_no_value. */
    static void fill_cores(socket_stream& socket, const core_block& block,
                           const std::vector<catalog::cpu::core_t>& cores)
    {
        const auto count = std::min(socket.cpu_ids.size(), cores.size());
        for(std::size_t i = 0; i < count; ++i)
        {
            // A core the device skipped shifts the rest: never copy onto another core
            if(cores[i].cpu_id == socket.cpu_ids[i])
            {
                socket.rows.row[block.first + i] = block.metric->read(cores[i], 0, 0);
            }
        }
    }

    /** The procfs provider; null after release(). */
    std::shared_ptr<Provider> m_provider;
    /** The metrics the settings enable. */
    enabled_t m_enabled{};
    /** Sockets with at least one column. */
    std::vector<socket_stream> m_sockets;
};
}  // namespace rocprofsys::pmc::pollers

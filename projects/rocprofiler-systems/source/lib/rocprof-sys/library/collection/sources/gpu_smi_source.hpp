// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/clocks.hpp"
#include "core/agent.hpp"
#include "core/collection/record.hpp"
#include "core/collection/scheduler.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "library/collection/sources/gpu_smi_catalog.hpp"
#include "library/pmc/collectors/gpu/gpu_traits.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::collection::sources
{
/** The values a GPU does not report are NaN. The AMD SMI provider is shared with the NIC
 * source and shut down by its owner. */
template <typename Provider, typename Device, typename Settings,
          typename Clock = std::chrono::steady_clock>
class gpu_smi_source final : public source_interface
{
public:
    using provider_getter = std::function<std::shared_ptr<Provider>()>;

    /** @param get_provider returns the started provider, or null when AMD SMI is off */
    gpu_smi_source(std::chrono::nanoseconds interval, provider_getter get_provider,
                   scheduler_factory<Clock> make_scheduler)
    : m_interval(interval)
    , m_get_provider(std::move(get_provider))
    , m_scheduler(make_scheduler("gpu"))
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return "gpu"; }

    void describe(source_id src_id, sink_interface& out) override
    {
        m_id  = src_id;
        m_out = &out;

        auto provider = m_get_provider();
        if(!provider)
        {
            return;
        }

        m_enabled = traits_t::template get_enabled_metrics<Settings>();
        for(auto& entry : traits_t::template enumerate_devices<Settings>(provider))
        {
            m_devices.push_back(make_stream(std::move(entry)));
        }
        m_provider = std::move(provider);
        LOG_INFO("Enabled {} GPU devices for AMD SMI sampling", m_devices.size());
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
        m_devices.clear();
        m_provider.reset();
    }

private:
    using traits_t       = pmc::collectors::gpu::gpu_traits<Provider, Device>;
    using device_entry_t = traits_t::device_entry;

    struct device_stream
    {
        device_entry_t               entry;
        schema_id                    schema = 0;
        std::vector<gpu_smi::reader> readers;  // one per column, in column order
        std::vector<double>          row;
        bool                         failed = false;
    };

    template <typename Metric>
    static column_spec make_column(const Metric& metric, const std::string& name,
                                   const gpu_smi::column_names& names)
    {
        column_spec column;
        column.name                = name;
        column.description         = metric.description;
        column.unit                = metric.unit;
        column.pmc_name            = names.pmc;
        column.perfetto_track_name = names.perfetto;
        column.rocpd_track_name    = names.rocpd;
        column.category_id         = metric.category_id;
        return column;
    }

    static void add_scalar_columns(schema& description, device_stream& stream,
                                   const gpu_smi::enabled_t& effective,
                                   std::uint32_t             device_id)
    {
        for(const auto& metric : gpu_smi::scalar_metrics())
        {
            if(!metric.enabled(effective))
            {
                continue;
            }
            const gpu_smi::column_names names{ .pmc      = metric.pmc_name,
                                               .perfetto = fmt::format(
                                                   "GPU [{}] {} (S)", device_id,
                                                   metric.perfetto_label),
                                               .rocpd = metric.pmc_name };
            description.columns.push_back(make_column(metric, metric.symbol, names));
            stream.readers.push_back({ .read = metric.read });
        }
    }

    template <typename Metric>
    static void add_array_metric(schema& description, device_stream& stream,
                                 const Metric& metric, const gpu_smi::metrics_t& probe)
    {
        const auto device_id =
            static_cast<std::uint32_t>(stream.entry.device->get_index());
        for(std::size_t index = 0; index < metric.first_count * metric.second_count;
            ++index)
        {
            const auto first  = index / metric.second_count;
            const auto second = index % metric.second_count;
            if(std::isnan(metric.read(probe, first, second)))
            {
                continue;
            }
            const auto names = metric.names(device_id, metric, first, second);
            description.columns.push_back(make_column(metric, names.pmc, names));
            stream.readers.push_back(
                { .read = metric.read, .first = first, .second = second });
        }
    }

    static void add_array_columns(schema& description, device_stream& stream,
                                  const gpu_smi::enabled_t& effective,
                                  const gpu_smi::metrics_t& probe)
    {
        for(const auto& metric : gpu_smi::array_metrics())
        {
            if(metric.enabled(effective))
            {
                add_array_metric(description, stream, metric, probe);
            }
        }
    }

    // The first reading decides which array elements the device reports. SDMA is left
    // out: it is a delta against the previous reading, and the first real poll should be
    // it.
    [[nodiscard]] gpu_smi::metrics_t probe_reading(const device_entry_t& entry) const
    {
        auto probe_enabled            = m_enabled;
        probe_enabled.bits.sdma_usage = 0;
        return entry.device->get_metrics(probe_enabled, 0);
    }

    device_stream make_stream(device_entry_t entry)
    {
        device_stream stream;
        stream.entry = std::move(entry);

        gpu_smi::enabled_t effective;
        effective.value = m_enabled.value & stream.entry.supported_metrics.value;
        const auto device_id =
            static_cast<std::uint32_t>(stream.entry.device->get_index());

        schema description;
        description.name        = fmt::format("GPU [{}]", device_id);
        description.target_arch = "GPU";
        description.agent       = agent_type::gpu;
        description.device      = device_id;

        add_scalar_columns(description, stream, effective, device_id);
        add_array_columns(description, stream, effective, probe_reading(stream.entry));

        stream.row.assign(description.columns.size(), gpu_smi::detail::nan_value());
        stream.schema = m_out->register_schema(m_id, description);
        return stream;
    }

    void poll_once(std::uint64_t timeline_ns)
    {
        m_timestamp = timeline_ns;
        for(auto& device : m_devices)
        {
            if(!device.failed)
            {
                poll_device(device);
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

    void poll_device(device_stream& stream)
    {
        try
        {
            const auto reading = stream.entry.device->get_metrics(m_enabled, m_timestamp);
            std::transform(stream.readers.begin(), stream.readers.end(),
                           stream.row.begin(), [&reading](const gpu_smi::reader& item) {
                               return item.read(reading, item.first, item.second);
                           });
        } catch(const std::exception& error)
        {
            LOG_ERROR("Reading metrics failed for GPU device {}. Error: {}. Disabling "
                      "device!",
                      stream.entry.device->get_index(), error.what());
            stream.failed = true;
            return;
        }

        m_out->write_batch({ .source    = m_id,
                             .schema    = stream.schema,
                             .timestamp = m_timestamp,
                             .values    = stream.row });
    }

    std::chrono::nanoseconds          m_interval;
    provider_getter                   m_get_provider;
    std::unique_ptr<scheduler<Clock>> m_scheduler;
    std::uint64_t                     m_gap_begin_ns = 0;
    bool                              m_paused       = false;
    std::uint64_t                     m_timestamp    = 0;
    source_id                         m_id           = 0;
    sink_interface*                   m_out          = nullptr;
    gpu_smi::enabled_t                m_enabled{};
    std::shared_ptr<Provider>         m_provider;
    std::vector<device_stream>        m_devices;
};
}  // namespace rocprofsys::collection::sources

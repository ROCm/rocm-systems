// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/clocks.hpp"
#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/collection/record.hpp"
#include "core/collection/scheduler.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "library/pmc/collectors/hipfile/hipfile_traits.hpp"
#include "library/pmc/collectors/hipfile/types.hpp"
#include "logger/debug.hpp"
#include "rocprofiler-systems/categories.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace rocprofsys::collection::sources
{
/** A poll where hipFile cannot be queried (the target did no hipFile I/O yet) writes no
 * batch: that is not a measurement of zero. */
template <typename Provider, typename Device, typename Settings,
          typename Clock = std::chrono::steady_clock>
class hipfile_source final : public source_interface
{
public:
    using provider_getter = std::function<std::shared_ptr<Provider>()>;

    /** @param get_provider returns the started provider, or null when hipFile is off */
    hipfile_source(std::chrono::nanoseconds interval, provider_getter get_provider,
                   scheduler_factory<Clock> make_scheduler)
    : m_interval(interval)
    , m_get_provider(std::move(get_provider))
    , m_scheduler(make_scheduler("hipfile"))
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return "hipfile"; }

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
            m_gpus.push_back(make_stream(std::move(entry)));
        }
        m_provider = std::move(provider);
        LOG_INFO("Enabled {} GPU devices for hipFile sampling", m_gpus.size());
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
        m_gpus.clear();
        m_provider.reset();
    }

private:
    using traits_t       = pmc::collectors::hipfile::hipfile_traits<Provider, Device>;
    using device_entry_t = traits_t::device_entry;
    using value_fn       = double (*)(const pmc::collectors::hipfile::metrics&);

    struct gpu_stream
    {
        device_entry_t        entry;
        schema_id             schema = 0;
        std::vector<value_fn> readers;  // one per column, in column order
        std::vector<double>   row;
    };

    static column_spec make_column(const pmc::collectors::hipfile::metric_desc& metric,
                                   std::size_t                                  gpu_index)
    {
        namespace hipfile = pmc::collectors::hipfile;

        const auto track = hipfile::track_name(gpu_index, metric.suffix);

        column_spec column;
        column.name        = metric.suffix;
        column.description = tim::trait::name<tim::category::hipfile>::description;
        column.unit        = metric.unit;
        column.pmc_name    = hipfile::pmc_name(metric.suffix);
        column.perfetto_track_name = track;
        column.rocpd_track_name    = track;
        column.category_id         = ROCPROFSYS_CATEGORY_HIPFILE;
        return column;
    }

    gpu_stream make_stream(device_entry_t entry)
    {
        namespace hipfile = pmc::collectors::hipfile;

        gpu_stream stream;
        stream.entry = std::move(entry);

        hipfile::enabled_metrics effective;
        effective.value      = m_enabled.value & stream.entry.supported_metrics.value;
        const auto gpu_index = stream.entry.device->get_index();

        schema description;
        description.name        = fmt::format("GPU [{}] Storage", gpu_index);
        description.target_arch = "GPU";
        description.agent       = agent_type::gpu;
        description.device      = static_cast<std::uint32_t>(gpu_index);

        for(const auto& metric : hipfile::METRIC_TABLE)
        {
            if((effective.value & (1U << metric.bit)) == 0U)
            {
                continue;
            }
            description.columns.push_back(make_column(metric, gpu_index));
            stream.readers.push_back(metric.value);
        }

        stream.row.assign(description.columns.size(), 0.0);
        stream.schema = m_out->register_schema(m_id, description);
        return stream;
    }

    void poll_once(std::uint64_t timeline_ns)
    {
        m_timestamp = timeline_ns;
        for(auto& gpu : m_gpus)
        {
            poll_gpu(gpu);
        }
    }

    void write_gap()
    {
        m_out->write_gap({ .source   = m_id,
                           .begin_ns = m_gap_begin_ns,
                           .end_ns   = clocks::timeline::now_ns() });
        m_paused = false;
    }

    void poll_gpu(gpu_stream& stream)
    {
        try
        {
            const auto reading = stream.entry.device->get_metrics(m_enabled, m_timestamp);
            if(reading.query_failed)
            {
                return;
            }
            std::transform(stream.readers.begin(), stream.readers.end(),
                           stream.row.begin(),
                           [&reading](value_fn read) { return read(reading); });
        } catch(const std::exception& error)
        {
            LOG_ERROR("Reading hipFile metrics failed for GPU {}. Error: {}",
                      stream.entry.device->get_index(), error.what());
            return;
        }

        m_out->write_batch({ .source    = m_id,
                             .schema    = stream.schema,
                             .timestamp = m_timestamp,
                             .values    = stream.row });
    }

    std::chrono::nanoseconds                  m_interval;
    provider_getter                           m_get_provider;
    std::unique_ptr<scheduler<Clock>>         m_scheduler;
    std::uint64_t                             m_gap_begin_ns = 0;
    bool                                      m_paused       = false;
    std::uint64_t                             m_timestamp    = 0;
    source_id                                 m_id           = 0;
    sink_interface*                           m_out          = nullptr;
    pmc::collectors::hipfile::enabled_metrics m_enabled{};
    std::shared_ptr<Provider>                 m_provider;
    std::vector<gpu_stream>                   m_gpus;
};
}  // namespace rocprofsys::collection::sources

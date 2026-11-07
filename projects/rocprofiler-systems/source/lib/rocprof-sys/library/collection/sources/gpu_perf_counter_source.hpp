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
#include "library/pmc/collectors/gpu_perf_counter/gpu_perf_counter_traits.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "logger/debug.hpp"
#include "rocprofiler-systems/categories.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys::collection::sources
{
/** The provider is created while the SDK is configured, the only time its contexts can be
 * created. A counter the SDK did not report in a poll is NaN. */
template <typename Provider, typename Settings,
          typename Clock = std::chrono::steady_clock>
class gpu_perf_counter_source final : public source_interface
{
public:
    gpu_perf_counter_source(std::shared_ptr<Provider> provider,
                            std::chrono::nanoseconds  interval,
                            scheduler_factory<Clock>  make_scheduler)
    : m_interval(interval)
    , m_provider(std::move(provider))
    , m_scheduler(make_scheduler("gpu_perf_counter"))
    {}

    [[nodiscard]] std::string_view name() const noexcept override
    {
        return "gpu_perf_counter";
    }

    void describe(source_id src_id, sink_interface& out) override
    {
        m_id  = src_id;
        m_out = &out;

        for(auto& entry : traits_t::template enumerate_devices<Settings>(m_provider))
        {
            m_gpus.push_back(make_stream(std::move(entry)));
        }
        LOG_INFO("Enabled {} GPU devices for hardware counter sampling", m_gpus.size());
    }

    void start() override
    {
        // The contexts need HSA, which may not be live yet; the devices retry on every
        // poll.
        m_provider->start();
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
        m_provider->shutdown();
        m_gpus.clear();
    }

private:
    using traits_t = pmc::collectors::gpu_perf_counter::gpu_perf_counter_traits<Provider>;
    using device_entry_t = traits_t::device_entry;

    struct gpu_stream
    {
        device_entry_t                                 entry;
        schema_id                                      schema = 0;
        std::unordered_map<std::uint64_t, std::size_t> column_of;
        std::vector<double>                            row;
    };

    static column_spec make_column(
        const pmc::collectors::gpu_perf_counter::counter_metadata& meta,
        std::size_t                                                gpu_index)
    {
        namespace perf_counter = pmc::collectors::gpu_perf_counter;

        const auto qualified = perf_counter::make_qualified_name(meta);
        const auto track     = perf_counter::format_track_name(gpu_index, qualified);

        column_spec column;
        column.name = qualified;
        column.description =
            meta.description.empty() ? "SDK PMC hardware counter" : meta.description;
        column.unit                = "count";
        column.pmc_name            = qualified;
        column.perfetto_track_name = track;
        column.rocpd_track_name    = track;
        column.category_id         = ROCPROFSYS_CATEGORY_ROCM_COUNTER_COLLECTION;
        column.block               = meta.block;
        column.expression          = meta.expression;
        column.is_constant         = meta.is_constant;
        column.is_derived          = meta.is_derived;
        return column;
    }

    gpu_stream make_stream(device_entry_t entry)
    {
        gpu_stream stream;
        stream.entry         = std::move(entry);
        const auto gpu_index = stream.entry.device->get_index();

        schema description;
        description.name        = fmt::format("GPU [{}] Counters", gpu_index);
        description.target_arch = "GPU";
        description.agent       = agent_type::gpu;
        description.device      = static_cast<std::uint32_t>(gpu_index);

        for(const auto& meta : stream.entry.device->get_counter_metadata())
        {
            stream.column_of.emplace(meta.counter_id, description.columns.size());
            description.columns.push_back(make_column(meta, gpu_index));
        }

        stream.row.assign(description.columns.size(),
                          std::numeric_limits<double>::quiet_NaN());
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
            const auto& records =
                stream.entry.device->sample_metrics(m_enabled, m_timestamp);
            if(records.empty())
            {
                return;
            }
            fill_row(stream, records);
        } catch(const std::exception& error)
        {
            LOG_ERROR("Reading hardware counters failed for GPU {}. Error: {}",
                      stream.entry.device->get_index(), error.what());
            return;
        }

        m_out->write_batch({ .source    = m_id,
                             .schema    = stream.schema,
                             .timestamp = m_timestamp,
                             .values    = stream.row });
    }

    static void fill_row(gpu_stream&                                       stream,
                         const pmc::collectors::gpu_perf_counter::metrics& records)
    {
        std::fill(stream.row.begin(), stream.row.end(),
                  std::numeric_limits<double>::quiet_NaN());
        for(const auto& record : records)
        {
            const auto column = stream.column_of.find(record.counter_id);
            if(column != stream.column_of.end())
            {
                stream.row[column->second] = record.value;
            }
        }
    }

    std::chrono::nanoseconds                           m_interval;
    std::shared_ptr<Provider>                          m_provider;
    std::unique_ptr<scheduler<Clock>>                  m_scheduler;
    std::uint64_t                                      m_gap_begin_ns = 0;
    bool                                               m_paused       = false;
    std::uint64_t                                      m_timestamp    = 0;
    source_id                                          m_id           = 0;
    sink_interface*                                    m_out          = nullptr;
    pmc::collectors::gpu_perf_counter::enabled_metrics m_enabled;
    std::vector<gpu_stream>                            m_gpus;
};
}  // namespace rocprofsys::collection::sources

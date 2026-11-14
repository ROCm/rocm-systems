// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "library/pmc/collectors/gpu_perf_counter/gpu_perf_counter_traits.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/pollers/row_stream.hpp"
#include "logger/debug.hpp"
#include "policies/pmc/device_policy.hpp"
#include "rocprofiler-systems/categories.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::pollers
{
/** Polls the GPU hardware counters through the profiling SDK: one row per GPU. A counter
 * the SDK did not report in a poll is k_no_value; a poll without records writes no row.
 * The provider is created while the SDK is configured, the only time its contexts can be
 * created; release() shuts it down. */
template <typename Provider, typename Settings>
class gpu_perf_counter final : public collection::source_interface
{
    using traits_t = pmc::collectors::gpu_perf_counter::gpu_perf_counter_traits<Provider>;
    using enabled_t = pmc::collectors::gpu_perf_counter::enabled_metrics;
    using records_t = pmc::collectors::gpu_perf_counter::metrics;

    static_assert(policies::pmc::counter_device<typename traits_t::device_t, enabled_t>);
    static_assert(policies::pmc::stoppable_provider<Provider>);

public:
    /** @param provider the SDK counter provider, or null */
    explicit gpu_perf_counter(std::shared_ptr<Provider> provider)
    : m_provider(std::move(provider))
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return "gpu_pmc"; }

    [[nodiscard]] bool describe(collection::sink_interface& out) override
    {
        if(!m_provider)
        {
            return false;
        }
        // Settings is only part of the traits signature: the agent list given to the
        // provider is already filtered.
        for(auto& entry : traits_t::template enumerate_devices<Settings>(m_provider))
        {
            add_gpu(std::move(entry), out);
        }
        LOG_INFO("Enabled {} GPU devices for hardware counter sampling", m_gpus.size());
        if(m_gpus.empty())
        {
            release();
            return false;
        }
        return true;
    }

    void poll(std::uint64_t timeline_ns, collection::sink_interface& out) override
    {
        poll_streams(m_gpus, timeline_ns, out, [timeline_ns](gpu_stream& gpu) {
            return fill_row(gpu, timeline_ns);
        });
    }

    void release() override
    {
        m_gpus.clear();
        if(m_provider)
        {
            m_provider->shutdown();
            m_provider.reset();
        }
    }

private:
    /** One GPU and the row it writes. */
    struct gpu_stream
    {
        /** The device. */
        traits_t::device_entry entry;
        /** Its row. */
        row_stream rows;
        /** Column of each counter id. */
        std::unordered_map<pmc::collectors::gpu_perf_counter::counter_id_t, std::size_t>
            column_of;
    };

    /** The column of one counter on GPU `gpu_index`. */
    [[nodiscard]] static collection::column_spec make_column(
        const pmc::collectors::gpu_perf_counter::counter_metadata& meta,
        std::size_t                                                gpu_index)
    {
        namespace perf_counter = pmc::collectors::gpu_perf_counter;
        const auto qualified   = perf_counter::make_qualified_name(meta);
        const auto track       = perf_counter::format_track_name(gpu_index, qualified);
        return { .symbol      = qualified,
                 .description = meta.description.empty() ? "SDK PMC hardware counter"
                                                         : meta.description,
                 .unit        = "count",
                 .pmc_name    = qualified,
                 .perfetto_track_name = track,
                 .rocpd_track_name    = track,
                 .category_id         = ROCPROFSYS_CATEGORY_ROCM_COUNTER_COLLECTION,
                 .block               = meta.block,
                 .expression          = meta.expression,
                 .is_constant         = meta.is_constant,
                 .is_derived          = meta.is_derived };
    }

    /** Builds and registers the schema of one GPU; a GPU without counters is skipped. */
    void add_gpu(traits_t::device_entry entry, collection::sink_interface& out)
    {
        gpu_stream stream{ .entry = std::move(entry), .rows = {}, .column_of = {} };
        const auto gpu_index = stream.entry.device->get_index();

        collection::schema description{
            .agent       = agent_type::gpu,
            .device      = static_cast<std::uint32_t>(gpu_index),
            .target_arch = "GPU",
            .event_category =
                std::string{
                    category_info(ROCPROFSYS_CATEGORY_ROCM_COUNTER_COLLECTION).name },
            .columns = {}
        };
        for(const auto& meta : stream.entry.device->get_counter_metadata())
        {
            stream.column_of.emplace(meta.counter_id, description.columns.size());
            description.columns.push_back(make_column(meta, gpu_index));
        }
        if(description.columns.empty())
        {
            LOG_DEBUG("GPU {} has no hardware counter, not sampled", gpu_index);
            return;
        }
        stream.rows.open(description, out, name(), gpu_index);
        m_gpus.push_back(std::move(stream));
    }

    /** Reads the GPU into its row. @return false when the SDK returned no records (no
     * row) */
    [[nodiscard]] static bool fill_row(gpu_stream& stream, std::uint64_t timeline_ns)
    {
        // The device does not filter by enabled metrics; its counters were chosen when
        // the provider was created.
        const records_t& records = stream.entry.device->sample_metrics({}, timeline_ns);
        if(records.empty())
        {
            return false;
        }
        std::ranges::fill(stream.rows.row, collection::k_no_value);
        for(const auto& record : records)
        {
            if(const auto column = stream.column_of.find(record.counter_id);
               column != stream.column_of.end())
            {
                stream.rows.row[column->second] = record.value;
            }
        }
        return true;
    }

    /** The SDK counter provider; null after release(). */
    std::shared_ptr<Provider> m_provider;
    /** GPUs with at least one counter. */
    std::vector<gpu_stream> m_gpus;
};
}  // namespace rocprofsys::pmc::pollers

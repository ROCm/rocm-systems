// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "library/pmc/collectors/hipfile/hipfile_traits.hpp"
#include "library/pmc/collectors/hipfile/types.hpp"
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
/** Polls the hipFile I/O statistics of each GPU: one row per GPU. A poll where hipFile
 * cannot be queried (the target did no hipFile I/O yet) writes no row: that is not a
 * measurement of zero. release() shuts the provider down. */
template <typename Provider, typename Device, typename Settings>
class hipfile final : public collection::source_interface
{
    using traits_t  = pmc::collectors::hipfile::hipfile_traits<Provider, Device>;
    using enabled_t = pmc::collectors::hipfile::enabled_metrics;
    using metric_t  = pmc::collectors::hipfile::metric_desc;

    static_assert(policies::pmc::hipfile_device<Device, enabled_t>);
    static_assert(policies::pmc::hipfile_settings<Settings>);
    static_assert(policies::pmc::stoppable_provider<Provider>);

public:
    /** @param provider the hipFile provider, or null */
    explicit hipfile(std::shared_ptr<Provider> provider)
    : m_provider(std::move(provider))
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return "hipfile"; }

    [[nodiscard]] bool describe(collection::sink_interface& out) override
    {
        if(!m_provider)
        {
            return false;
        }
        m_enabled = traits_t::template get_enabled_metrics<Settings>();
        for(auto& entry : traits_t::template enumerate_devices<Settings>(m_provider))
        {
            add_gpu(std::move(entry), out);
        }
        LOG_INFO("Enabled {} GPU devices for hipFile sampling", m_gpus.size());
        if(m_gpus.empty())
        {
            release();
            return false;
        }
        return true;
    }

    void poll(std::uint64_t timeline_ns, collection::sink_interface& out) override
    {
        poll_streams(m_gpus, timeline_ns, out, [this, timeline_ns](gpu_stream& gpu) {
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
        /** The device and its supported metrics. */
        traits_t::device_entry entry;
        /** Its row. */
        row_stream rows;
        /** One METRIC_TABLE entry per column, in column order. */
        std::vector<const metric_t*> metrics;
    };

    /** The column of `metric` on GPU `gpu_index`. */
    [[nodiscard]] static collection::column_spec make_column(const metric_t& metric,
                                                             std::size_t     gpu_index)
    {
        namespace hipfile_metrics = pmc::collectors::hipfile;
        const auto track          = hipfile_metrics::track_name(gpu_index, metric.suffix);
        return { .symbol = metric.suffix,
                 .description =
                     std::string{
                         category_info(ROCPROFSYS_CATEGORY_HIPFILE).description },
                 .unit                = metric.unit,
                 .pmc_name            = hipfile_metrics::pmc_name(metric.suffix),
                 .perfetto_track_name = track,
                 .rocpd_track_name    = track,
                 .category_id         = ROCPROFSYS_CATEGORY_HIPFILE,
                 .block               = {},
                 .expression          = {} };
    }

    /** Builds and registers the schema of one GPU; a GPU without columns is skipped. */
    void add_gpu(traits_t::device_entry entry, collection::sink_interface& out)
    {
        gpu_stream stream{ .entry = std::move(entry), .rows = {}, .metrics = {} };
        const auto effective = m_enabled.value & stream.entry.supported_metrics.value;
        const auto gpu_index = stream.entry.device->get_index();

        collection::schema description{
            .agent       = agent_type::gpu,
            .device      = static_cast<std::uint32_t>(gpu_index),
            .target_arch = "GPU",
            .event_category =
                std::string{ category_info(ROCPROFSYS_CATEGORY_HIPFILE).name },
            .columns = {}
        };
        for(const auto& metric : pmc::collectors::hipfile::METRIC_TABLE)
        {
            if((effective & (1U << metric.bit)) != 0U)
            {
                description.columns.push_back(make_column(metric, gpu_index));
                stream.metrics.push_back(&metric);
            }
        }
        if(description.columns.empty())
        {
            LOG_DEBUG("hipFile GPU {} has no enabled metric, not sampled", gpu_index);
            return;
        }
        stream.rows.open(description, out, name(), gpu_index);
        m_gpus.push_back(std::move(stream));
    }

    /** Reads the GPU into its row. @return false when hipFile could not be queried (no
     * row) */
    [[nodiscard]] bool fill_row(gpu_stream& stream, std::uint64_t timeline_ns) const
    {
        const auto reading = stream.entry.device->get_metrics(m_enabled, timeline_ns);
        if(reading.query_failed)
        {
            return false;
        }
        std::ranges::transform(
            stream.metrics, stream.rows.row.begin(),
            [&reading](const metric_t* metric) { return metric->value(reading); });
        return true;
    }

    /** The hipFile provider; null after release(). */
    std::shared_ptr<Provider> m_provider;
    /** The metrics the settings enable. */
    enabled_t m_enabled{};
    /** GPUs with at least one column. */
    std::vector<gpu_stream> m_gpus;
};
}  // namespace rocprofsys::pmc::pollers

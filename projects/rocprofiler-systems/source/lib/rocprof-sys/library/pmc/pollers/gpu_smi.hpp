// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "library/pmc/collectors/gpu/gpu_traits.hpp"
#include "library/pmc/pollers/catalog/gpu_smi.hpp"
#include "library/pmc/pollers/row_stream.hpp"
#include "logger/debug.hpp"
#include "policies/pmc/device_policy.hpp"
#include "policies/pmc/settings_policy.hpp"
#include "rocprofiler-systems/categories.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::pollers
{
/** Polls the GPUs through AMD SMI: one row per GPU. A value the GPU does not report is
 * k_no_value. The provider is shared with the NIC poller. */
template <typename Provider, typename Device, typename Settings>
class gpu_smi final : public collection::source_interface
{
    using traits_t  = pmc::collectors::gpu::gpu_traits<Provider, Device>;
    using enabled_t = catalog::gpu_smi::enabled_t;

    static_assert(policies::pmc::gpu_smi_device<Device, enabled_t>);
    static_assert(policies::pmc::gpu_smi_settings<Settings>);

public:
    /** @param provider the AMD SMI provider, or null */
    explicit gpu_smi(std::shared_ptr<Provider> provider)
    : m_provider(std::move(provider))
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return "gpu_smi"; }

    [[nodiscard]] bool describe(collection::sink_interface& out) override
    {
        if(!m_provider)
        {
            return false;
        }
        m_enabled = traits_t::template get_enabled_metrics<Settings>();
        for(auto& entry : traits_t::template enumerate_devices<Settings>(m_provider))
        {
            add_device(std::move(entry), out);
        }
        LOG_INFO("Enabled {} GPU devices for AMD SMI sampling", m_devices.size());
        if(m_devices.empty())
        {
            release();
            return false;
        }
        return true;
    }

    void poll(std::uint64_t timeline_ns, collection::sink_interface& out) override
    {
        poll_streams(m_devices, timeline_ns, out,
                     [this, timeline_ns](device_stream& device) {
                         fill_row(device, timeline_ns);
                         return true;
                     });
    }

    void release() override
    {
        m_devices.clear();
        m_provider.reset();
    }

private:
    /** One GPU and the row it writes. */
    struct device_stream
    {
        /** The device and its supported metrics. */
        traits_t::device_entry entry;
        /** Its row. */
        row_stream rows;
        /** One per column, in column order. */
        std::vector<catalog::gpu_smi::reader> readers;
    };

    /** Builds and registers the schema of one GPU; a GPU without columns is skipped. */
    void add_device(traits_t::device_entry entry, collection::sink_interface& out)
    {
        device_stream stream{ .entry = std::move(entry), .rows = {}, .readers = {} };
        enabled_t     effective{};
        effective.value = m_enabled.value & stream.entry.supported_metrics.value;
        const auto device_id =
            static_cast<std::uint32_t>(stream.entry.device->get_index());

        collection::schema description{
            .agent       = agent_type::gpu,
            .device      = device_id,
            .target_arch = "GPU",
            .event_category =
                std::string{ category_info(ROCPROFSYS_CATEGORY_AMD_SMI).name },
            .columns = {}
        };
        for(const auto& metric : catalog::gpu_smi::k_scalar_metrics)
        {
            if(metric.enabled(effective))
            {
                add_column(description, stream, metric, { .read = metric.read });
            }
        }
        for(const auto& metric : catalog::gpu_smi::k_array_metrics)
        {
            if(metric.enabled(effective))
            {
                add_supported_elements(description, stream, metric);
            }
        }
        if(description.columns.empty())
        {
            LOG_DEBUG("GPU device {} has no enabled metric, not sampled", device_id);
            return;
        }
        stream.rows.open(description, out, name(), device_id);
        m_devices.push_back(std::move(stream));
    }

    /** Adds a column per element of `metric` that the device reported at its init
     * reading. */
    static void add_supported_elements(collection::schema&            description,
                                       device_stream&                 stream,
                                       const catalog::gpu_smi::row_t& metric)
    {
        const auto& support = stream.entry.device->support_reading();
        for(std::size_t index = 0; index < metric.outer * metric.inner; ++index)
        {
            const catalog::gpu_smi::reader item{ .read  = metric.read,
                                                 .outer = index / metric.inner,
                                                 .inner = index % metric.inner };
            if(!std::isnan(item.read(support, item.outer, item.inner)))
            {
                add_column(description, stream, metric, item);
            }
        }
    }

    /** Adds the column of one element and its reader. */
    static void add_column(collection::schema& description, device_stream& stream,
                           const catalog::gpu_smi::row_t&  metric,
                           const catalog::gpu_smi::reader& item)
    {
        description.columns.push_back(catalog::gpu_smi::make_column(
            metric, description.device, item.outer, item.inner));
        stream.readers.push_back(item);
    }

    /** Reads the GPU into its row. */
    void fill_row(device_stream& stream, std::uint64_t timeline_ns) const
    {
        const auto reading = stream.entry.device->get_metrics(m_enabled, timeline_ns);
        std::ranges::transform(stream.readers, stream.rows.row.begin(),
                               [&reading](const catalog::gpu_smi::reader& item) {
                                   return item.read(reading, item.outer, item.inner);
                               });
    }

    /** The AMD SMI provider; null after release(). */
    std::shared_ptr<Provider> m_provider;
    /** The metrics the settings enable. */
    enabled_t m_enabled{};
    /** GPUs with at least one column. */
    std::vector<device_stream> m_devices;
};
}  // namespace rocprofsys::pmc::pollers

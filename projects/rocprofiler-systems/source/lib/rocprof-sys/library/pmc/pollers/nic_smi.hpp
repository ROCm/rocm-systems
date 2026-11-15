// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"
#include "library/pmc/collectors/nic/nic_traits.hpp"
#include "library/pmc/pollers/catalog/nic_smi.hpp"
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
/** Polls the AI NICs through AMD SMI: one row per NIC with its enabled and supported
 * RDMA counters. The provider is shared with the GPU poller. */
template <typename Provider, typename Device, typename Settings>
class nic_smi final : public collection::source_interface
{
    using traits_t  = pmc::collectors::nic::nic_traits<Provider, Device>;
    using enabled_t = catalog::nic_smi::enabled_t;

    static_assert(policies::pmc::nic_device<Device>);
    static_assert(policies::pmc::nic_settings<Settings>);

public:
    /** @param provider the AMD SMI provider, or null */
    explicit nic_smi(std::shared_ptr<Provider> provider)
    : m_provider(std::move(provider))
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return "nic_smi"; }

    [[nodiscard]] bool describe(collection::sink_interface& out) override
    {
        if(!m_provider)
        {
            return false;
        }
        m_enabled    = traits_t::template get_enabled_metrics<Settings>();
        auto entries = traits_t::template enumerate_devices<Settings>(m_provider);
        for(std::size_t position = 0; position < entries.size(); ++position)
        {
            add_nic(std::move(entries[position]), position, out);
        }
        LOG_INFO("Enabled {} NIC devices for AMD SMI sampling", m_nics.size());
        if(m_nics.empty())
        {
            release();
            return false;
        }
        return true;
    }

    void poll(std::uint64_t timeline_ns, collection::sink_interface& out) override
    {
        poll_streams(m_nics, timeline_ns, out, [](nic_stream& nic) {
            fill_row(nic);
            return true;
        });
    }

    void release() override
    {
        m_nics.clear();
        m_provider.reset();
    }

private:
    /** One NIC and the row it writes. */
    struct nic_stream
    {
        /** The device and its supported metrics. */
        traits_t::device_entry entry;
        /** Its row. */
        row_stream rows;
        /** One catalog row per column, in column order. */
        std::vector<const catalog::nic_smi::row_t*> metrics;
    };

    /** Builds and registers the schema of one NIC; a NIC without columns is skipped.
     * `position` is the agent type index the traits gave the NIC; the tracks are named
     * after the AMD SMI index. */
    void add_nic(traits_t::device_entry entry, std::size_t position,
                 collection::sink_interface& out)
    {
        nic_stream stream{ .entry = std::move(entry), .rows = {}, .metrics = {} };
        enabled_t  effective{};
        effective.value      = m_enabled.value & stream.entry.supported_metrics.value;
        const auto smi_index = stream.entry.device->get_index();

        collection::schema description{
            .agent       = agent_type::nic,
            .device      = static_cast<std::uint32_t>(position),
            .target_arch = "NIC",
            .event_category =
                std::string{ category_info(ROCPROFSYS_CATEGORY_AMD_SMI_AINIC).name },
            .columns = {}
        };
        for(const auto& metric : catalog::nic_smi::k_metrics)
        {
            if(metric.enabled(effective))
            {
                description.columns.push_back(
                    catalog::nic_smi::make_column(metric, smi_index));
                stream.metrics.push_back(&metric);
            }
        }
        if(description.columns.empty())
        {
            LOG_DEBUG("NIC device {} has no enabled metric, not sampled", smi_index);
            return;
        }
        stream.rows.open(description, out, name(), smi_index);
        m_nics.push_back(std::move(stream));
    }

    /** Reads the NIC into its row. */
    static void fill_row(nic_stream& stream)
    {
        const auto reading = stream.entry.device->get_nic_metrics();
        std::ranges::transform(stream.metrics, stream.rows.row.begin(),
                               [&reading](const catalog::nic_smi::row_t* metric) {
                                   return metric->read(reading, 0, 0);
                               });
    }

    /** The AMD SMI provider; null after release(). */
    std::shared_ptr<Provider> m_provider;
    /** The metrics the settings enable. */
    enabled_t m_enabled{};
    /** NICs with at least one column. */
    std::vector<nic_stream> m_nics;
};
}  // namespace rocprofsys::pmc::pollers

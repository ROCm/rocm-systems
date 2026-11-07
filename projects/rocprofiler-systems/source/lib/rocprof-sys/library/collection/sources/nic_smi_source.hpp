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
#include "library/pmc/collectors/nic/nic_traits.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "logger/debug.hpp"
#include "rocprofiler-systems/categories.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace rocprofsys::collection::sources
{
namespace nic_smi
{
using enabled_t = pmc::collectors::nic::enabled_metrics;
using metrics_t = pmc::collectors::nic::metrics;

struct nic_metric
{
    std::uint32_t enabled_bit;  ///< bit of the metric in `enabled_metrics`
    std::uint64_t metrics_t::*value;
    std::uint32_t             category_id;
    const char*               pmc_name;  ///< the rocpd event name
    const char*               rocpd_track_name;
    const char*               perfetto_label;
    const char*               unit;
    const char*               symbol;
    const char*               description;
};

inline constexpr std::array<nic_metric, 10> k_nic_metrics{ {
    { .enabled_bit = 0,
      .value       = &metrics_t::rx_rdma_ucast_bytes,
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_RX_UCAST_BYTES,
      .pmc_name    = tim::trait::name<tim::category::amd_smi_nic_rx_ucast_bytes>::value,
      .rocpd_track_name = "ainic_rx_rdma_ucast_bytes",
      .perfetto_label   = "RX RDMA BYTES",
      .unit             = "bytes",
      .symbol           = "NIC RX UCast Bytes",
      .description =
          tim::trait::name<tim::category::amd_smi_nic_rx_ucast_bytes>::description },
    { .enabled_bit = 1,
      .value       = &metrics_t::tx_rdma_ucast_bytes,
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_TX_UCAST_BYTES,
      .pmc_name    = tim::trait::name<tim::category::amd_smi_nic_tx_ucast_bytes>::value,
      .rocpd_track_name = "ainic_tx_rdma_ucast_bytes",
      .perfetto_label   = "TX RDMA BYTES",
      .unit             = "bytes",
      .symbol           = "NIC TX UCast Bytes",
      .description =
          tim::trait::name<tim::category::amd_smi_nic_tx_ucast_bytes>::description },
    { .enabled_bit = 2,
      .value       = &metrics_t::rx_rdma_ucast_pkts,
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_RX_UCAST_PKTS,
      .pmc_name    = tim::trait::name<tim::category::amd_smi_nic_rx_ucast_pkts>::value,
      .rocpd_track_name = "ainic_rx_rdma_ucast_pkts",
      .perfetto_label   = "RX RDMA PACKETS",
      .unit             = "packets",
      .symbol           = "NIC RX UCast PKTS",
      .description =
          tim::trait::name<tim::category::amd_smi_nic_rx_ucast_pkts>::description },
    { .enabled_bit = 3,
      .value       = &metrics_t::tx_rdma_ucast_pkts,
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_TX_UCAST_PKTS,
      .pmc_name    = tim::trait::name<tim::category::amd_smi_nic_tx_ucast_pkts>::value,
      .rocpd_track_name = "ainic_tx_rdma_ucast_pkts",
      .perfetto_label   = "TX RDMA PACKETS",
      .unit             = "packets",
      .symbol           = "NIC TX UCast PKTS",
      .description =
          tim::trait::name<tim::category::amd_smi_nic_tx_ucast_pkts>::description },
    { .enabled_bit      = 4,
      .value            = &metrics_t::rx_rdma_cnp_pkts,
      .category_id      = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_RX_CNP_PKTS,
      .pmc_name         = tim::trait::name<tim::category::amd_smi_nic_rx_cnp_pkts>::value,
      .rocpd_track_name = "ainic_rx_rdma_cnp_pkts",
      .perfetto_label   = "RX CNP PACKETS",
      .unit             = "packets",
      .symbol           = "NIC RX CNP PKTS",
      .description =
          tim::trait::name<tim::category::amd_smi_nic_rx_cnp_pkts>::description },
    { .enabled_bit      = 5,
      .value            = &metrics_t::tx_rdma_cnp_pkts,
      .category_id      = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_TX_CNP_PKTS,
      .pmc_name         = tim::trait::name<tim::category::amd_smi_nic_tx_cnp_pkts>::value,
      .rocpd_track_name = "ainic_tx_rdma_cnp_pkts",
      .perfetto_label   = "TX CNP PACKETS",
      .unit             = "packets",
      .symbol           = "NIC TX CNP PKTS",
      .description =
          tim::trait::name<tim::category::amd_smi_nic_tx_cnp_pkts>::description },
    { .enabled_bit = 6,
      .value       = &metrics_t::tx_rdma_ack_timeout,
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_TX_ACK_TIMEOUT,
      .pmc_name = tim::trait::name<tim::category::amd_smi_nic_tx_rdma_ack_timeout>::value,
      .rocpd_track_name = "ainic_tx_rdma_ack_timeout",
      .perfetto_label   = "TX ACK TIMEOUT",
      .unit             = "timeouts",
      .symbol           = "NIC TX RDMA ACK Timeout",
      .description =
          tim::trait::name<tim::category::amd_smi_nic_tx_rdma_ack_timeout>::description },
    { .enabled_bit = 7,
      .value       = &metrics_t::resp_tx_pkt_seq_err,
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_RESP_TX_PKT_SEQ_ERR,
      .pmc_name = tim::trait::name<tim::category::amd_smi_nic_resp_tx_pkt_seq_err>::value,
      .rocpd_track_name = "ainic_resp_tx_pkt_seq_err",
      .perfetto_label   = "RESP TX PKT SEQ ERR",
      .unit             = "errors",
      .symbol           = "NIC RESP TX PKT SEQ Error",
      .description =
          tim::trait::name<tim::category::amd_smi_nic_resp_tx_pkt_seq_err>::description },
    { .enabled_bit = 8,
      .value       = &metrics_t::req_rx_pkt_seq_err,
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_REQ_RX_PKT_SEQ_ERR,
      .pmc_name = tim::trait::name<tim::category::amd_smi_nic_req_rx_pkt_seq_err>::value,
      .rocpd_track_name = "ainic_req_rx_pkt_seq_err",
      .perfetto_label   = "REQ RX PKT SEQ ERR",
      .unit             = "errors",
      .symbol           = "NIC REQ RX PKT SEQ Error",
      .description =
          tim::trait::name<tim::category::amd_smi_nic_req_rx_pkt_seq_err>::description },
    { .enabled_bit = 9,
      .value       = &metrics_t::req_rx_impl_nak_seq_err,
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_REQ_RX_IMPL_NAK_SEQ_ERR,
      .pmc_name =
          tim::trait::name<tim::category::amd_smi_nic_req_rx_impl_nak_seq_err>::value,
      .rocpd_track_name = "ainic_req_rx_impl_nak_seq_err",
      .perfetto_label   = "REQ RX IMPL NAK SEQ ERR",
      .unit             = "errors",
      .symbol           = "NIC REQ RX Impl NAK SEQ Error",
      .description      = tim::trait::name<
               tim::category::amd_smi_nic_req_rx_impl_nak_seq_err>::description },
} };

inline std::span<const nic_metric>
nic_metrics()
{
    return k_nic_metrics;
}
}  // namespace nic_smi

/** The AMD SMI provider is shared with the GPU source and shut down by its owner. */
template <typename Provider, typename Device, typename Settings,
          typename Clock = std::chrono::steady_clock>
class nic_smi_source final : public source_interface
{
public:
    using provider_getter = std::function<std::shared_ptr<Provider>()>;

    /** @param get_provider returns the started provider, or null when AMD SMI is off */
    nic_smi_source(std::chrono::nanoseconds interval, provider_getter get_provider,
                   scheduler_factory<Clock> make_scheduler)
    : m_interval(interval)
    , m_get_provider(std::move(get_provider))
    , m_scheduler(make_scheduler("nic"))
    {}

    [[nodiscard]] std::string_view name() const noexcept override { return "nic"; }

    void describe(source_id src_id, sink_interface& out) override
    {
        m_id  = src_id;
        m_out = &out;

        auto provider = m_get_provider();
        if(!provider)
        {
            return;
        }

        m_enabled    = traits_t::template get_enabled_metrics<Settings>();
        auto entries = traits_t::template enumerate_devices<Settings>(provider);
        for(std::size_t position = 0; position < entries.size(); ++position)
        {
            m_nics.push_back(make_stream(std::move(entries[position]), position));
        }
        m_provider = std::move(provider);
        LOG_INFO("Enabled {} NIC devices for AMD SMI sampling", m_nics.size());
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
        m_nics.clear();
        m_provider.reset();
    }

private:
    using traits_t       = pmc::collectors::nic::nic_traits<Provider, Device>;
    using device_entry_t = traits_t::device_entry;

    struct nic_stream
    {
        device_entry_t                                   entry;
        schema_id                                        schema = 0;
        std::vector<std::uint64_t nic_smi::metrics_t::*> members;  // one per column
        std::vector<double>                              row;
        bool                                             failed = false;
    };

    static column_spec make_column(const nic_smi::nic_metric& metric,
                                   std::size_t                smi_index)
    {
        column_spec column;
        column.name        = metric.symbol;
        column.description = metric.description;
        column.unit        = metric.unit;
        column.pmc_name    = metric.pmc_name;
        column.perfetto_track_name =
            fmt::format("NIC [{}] {} (S)", smi_index, metric.perfetto_label);
        column.rocpd_track_name = metric.rocpd_track_name;
        column.category_id      = metric.category_id;
        return column;
    }

    // `position` is the type index the traits gave the NIC agent; the tracks are named
    // after the AMD SMI index.
    nic_stream make_stream(device_entry_t entry, std::size_t position)
    {
        nic_stream stream;
        stream.entry = std::move(entry);

        nic_smi::enabled_t effective;
        effective.value      = m_enabled.value & stream.entry.supported_metrics.value;
        const auto smi_index = stream.entry.device->get_index();

        schema description;
        description.name        = fmt::format("NIC [{}]", smi_index);
        description.target_arch = "NIC";
        description.agent       = agent_type::nic;
        description.device      = static_cast<std::uint32_t>(position);

        for(const auto& metric : nic_smi::nic_metrics())
        {
            if((effective.value & (std::uint32_t{ 1 } << metric.enabled_bit)) == 0)
            {
                continue;
            }
            description.columns.push_back(make_column(metric, smi_index));
            stream.members.push_back(metric.value);
        }

        stream.row.assign(description.columns.size(), 0.0);
        stream.schema = m_out->register_schema(m_id, description);
        return stream;
    }

    void poll_once(std::uint64_t timeline_ns)
    {
        m_timestamp = timeline_ns;
        for(auto& nic : m_nics)
        {
            if(!nic.failed)
            {
                poll_nic(nic);
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

    void poll_nic(nic_stream& stream)
    {
        try
        {
            const auto reading = stream.entry.device->get_nic_metrics();
            std::transform(
                stream.members.begin(), stream.members.end(), stream.row.begin(),
                [&reading](auto member) { return static_cast<double>(reading.*member); });
        } catch(const std::exception& error)
        {
            LOG_ERROR("Reading metrics failed for NIC device {}. Error: {}. Disabling "
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
    nic_smi::enabled_t                m_enabled{};
    std::shared_ptr<Provider>         m_provider;
    std::vector<nic_stream>           m_nics;
};
}  // namespace rocprofsys::collection::sources

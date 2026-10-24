// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/categories.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "logger/debug.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace rocprofsys::pmc::collectors::nic
{

/** One NIC RDMA metric: everything the outputs need, and where its value comes from. */
struct metric_row
{
    std::size_t   category        = 0;        ///< category enum id (Perfetto category)
    const char*   pmc_name        = nullptr;  ///< rocpd pmc_info name = category name
    const char*   description     = nullptr;
    const char*   label           = nullptr;  ///< Perfetto track label
    const char*   rocpd_track     = nullptr;  ///< rocpd track name (device-agnostic)
    const char*   symbol          = nullptr;
    const char*   units           = nullptr;
    std::uint64_t metrics::*field = nullptr;
};

/** Fill the category id, PMC name and description of a row from its category. */
template <typename Category>
[[nodiscard]] constexpr metric_row
with_category(metric_row row) noexcept
{
    row.category    = category_enum_id<Category>::value;
    row.pmc_name    = trait::name<Category>::value;
    row.description = trait::name<Category>::description;
    return row;
}

// clang-format off
inline constexpr std::array k_metric_rows{
    with_category<category::amd_smi_nic_rx_ucast_bytes>({ .label = "RX RDMA BYTES", .rocpd_track = "ainic_rx_rdma_ucast_bytes", .symbol = "NIC RX UCast Bytes", .units = "bytes", .field = &metrics::rx_rdma_ucast_bytes }),
    with_category<category::amd_smi_nic_tx_ucast_bytes>({ .label = "TX RDMA BYTES", .rocpd_track = "ainic_tx_rdma_ucast_bytes", .symbol = "NIC TX UCast Bytes", .units = "bytes", .field = &metrics::tx_rdma_ucast_bytes }),
    with_category<category::amd_smi_nic_rx_ucast_pkts>({ .label = "RX RDMA PACKETS", .rocpd_track = "ainic_rx_rdma_ucast_pkts", .symbol = "NIC RX UCast PKTS", .units = "packets", .field = &metrics::rx_rdma_ucast_pkts }),
    with_category<category::amd_smi_nic_tx_ucast_pkts>({ .label = "TX RDMA PACKETS", .rocpd_track = "ainic_tx_rdma_ucast_pkts", .symbol = "NIC TX UCast PKTS", .units = "packets", .field = &metrics::tx_rdma_ucast_pkts }),
    with_category<category::amd_smi_nic_rx_cnp_pkts>({ .label = "RX CNP PACKETS", .rocpd_track = "ainic_rx_rdma_cnp_pkts", .symbol = "NIC RX CNP PKTS", .units = "packets", .field = &metrics::rx_rdma_cnp_pkts }),
    with_category<category::amd_smi_nic_tx_cnp_pkts>({ .label = "TX CNP PACKETS", .rocpd_track = "ainic_tx_rdma_cnp_pkts", .symbol = "NIC TX CNP PKTS", .units = "packets", .field = &metrics::tx_rdma_cnp_pkts }),
    with_category<category::amd_smi_nic_tx_rdma_ack_timeout>({ .label = "TX ACK TIMEOUT", .rocpd_track = "ainic_tx_rdma_ack_timeout", .symbol = "NIC TX RDMA ACK Timeout", .units = "timeouts", .field = &metrics::tx_rdma_ack_timeout }),
    with_category<category::amd_smi_nic_resp_tx_pkt_seq_err>({ .label = "RESP TX PKT SEQ ERR", .rocpd_track = "ainic_resp_tx_pkt_seq_err", .symbol = "NIC RESP TX PKT SEQ Error", .units = "errors", .field = &metrics::resp_tx_pkt_seq_err }),
    with_category<category::amd_smi_nic_req_rx_pkt_seq_err>({ .label = "REQ RX PKT SEQ ERR", .rocpd_track = "ainic_req_rx_pkt_seq_err", .symbol = "NIC REQ RX PKT SEQ Error", .units = "errors", .field = &metrics::req_rx_pkt_seq_err }),
    with_category<category::amd_smi_nic_req_rx_impl_nak_seq_err>({ .label = "REQ RX IMPL NAK SEQ ERR", .rocpd_track = "ainic_req_rx_impl_nak_seq_err", .symbol = "NIC REQ RX Impl NAK SEQ Error", .units = "errors", .field = &metrics::req_rx_impl_nak_seq_err }),
};
// clang-format on

/**
 * @brief Output policy for writing NIC RDMA samples to the trace cache.
 *
 * Every NIC registers one PMC schema (names, units, agent) at config time; each sample is
 * then stored as a generic pmc_values_sample that carries only the values, in table
 * order.
 */
struct cache_policy
{
    /**
     * @brief Initialize trace cache category metadata for NIC RDMA metrics.
     */
    static void initialize_category_metadata()
    {
        trace_cache::get_metadata_registry().add_string("ainic");
    }

    /**
     * @brief Initialize NIC track metadata (rocpd tracks are shared by all NICs).
     */
    static void initialize_tracks_metadata()
    {
        for(const auto& row : k_metric_rows)
        {
            trace_cache::get_metadata_registry().add_track(
                { .track_name = row.rocpd_track,
                  .thread_id  = std::nullopt,
                  .extdata    = "{}" });
        }
    }

    /**
     * @brief Register the PMC info and the PMC schema of one NIC.
     *
     * The agent is resolved here once, from the AMD SMI index to the agent's type index,
     * so the pmc_info rows and the samples name the same agent (see ISS-109).
     *
     * @param nic_id AMD SMI index of the NIC
     * @param nic_name NIC product name (unused)
     */
    static void initialize_pmc_metadata(size_t                              nic_id,
                                        [[maybe_unused]] const std::string& nic_name)
    {
        std::size_t type_index = 0;
        try
        {
            type_index = get_agent_manager_instance()
                             .get_agent_by_id(nic_id, agent_type::nic)
                             .device_type_index;
        } catch(const std::out_of_range& e)
        {
            LOG_WARNING("NIC {} has no agent; its samples are not stored: {}", nic_id,
                        e.what());
            return;
        }

        trace_cache::info::pmc_schema schema{
            .type             = agent_type::nic,
            .agent_type_index = type_index,
            .event_category   = trait::name<category::amd_smi_nic>::value,
            .series           = {}
        };
        schema.series.reserve(k_metric_rows.size());
        for(const auto& row : k_metric_rows)
        {
            register_pmc_info(type_index, row);
            schema.series.push_back(
                { .perfetto_category = row.category,
                  .perfetto_track    = fmt::format("NIC [{}] {} (S)", nic_id, row.label),
                  .pmc_name          = row.pmc_name,
                  .rocpd_track       = row.rocpd_track,
                  .units             = row.units });
        }
        schema_ids()[nic_id] =
            trace_cache::get_metadata_registry().add_pmc_schema(std::move(schema));
    }

    /**
     * @brief Store a NIC sample to the trace cache as a pmc_values_sample.
     *
     * Every NIC metric is always enabled and a NIC is either fully supported or not
     * sampled, so the values are the whole table.
     *
     * @param device_id NIC device identifier
     * @param metric_values Collected metric values
     * @param timestamp Sample timestamp in nanoseconds
     */
    static void store_sample(size_t                                  device_id,
                             [[maybe_unused]] const std::string&     device_name,
                             [[maybe_unused]] const enabled_metrics& enabled_metrics_cfg,
                             [[maybe_unused]] const enabled_metrics& supported_metrics,
                             const metrics& metric_values, std::uint64_t timestamp)
    {
        const auto found = schema_ids().find(device_id);
        if(found == schema_ids().end())
        {
            return;
        }

        std::array<double, k_metric_rows.size()> values{};
        for(std::size_t idx = 0; idx < k_metric_rows.size(); ++idx)
        {
            values[idx] = static_cast<double>(metric_values.*k_metric_rows[idx].field);
        }
        trace_cache::get_buffer_storage().store(
            trace_cache::pmc_values_sample{ found->second, timestamp, values });
    }

private:
    /** Schema id per NIC (AMD SMI index); written at config, read on every sample. */
    static std::unordered_map<std::size_t, std::uint32_t>& schema_ids()
    {
        static std::unordered_map<std::size_t, std::uint32_t> s_ids;
        return s_ids;
    }

    static void register_pmc_info(std::size_t type_index, const metric_row& row)
    {
        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::nic,
              .agent_type_index = type_index,
              .target_arch      = "NIC",
              .event_code       = 0,
              .instance_id      = 0,
              .name             = row.pmc_name,
              .symbol           = row.symbol,
              .description      = row.description,
              .long_description = "",
              .component        = "",
              .units            = row.units,
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = "",
              .expression       = "",
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });
    }
};

}  // namespace rocprofsys::pmc::collectors::nic

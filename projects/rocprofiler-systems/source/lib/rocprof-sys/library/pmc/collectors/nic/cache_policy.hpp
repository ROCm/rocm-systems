// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/categories.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/nic/sample.hpp"
#include "library/pmc/collectors/nic/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace rocprofsys::pmc::collectors::nic
{

/**
 * @brief Output policy for writing NIC RDMA samples to the trace cache.
 *
 * This policy handles serialization of NIC RDMA metric samples into the
 * rocprofiler-systems trace cache for later analysis and visualization.
 *
 * @see perfetto_policy for direct Perfetto trace output
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
     * @brief Initialize NIC track metadata.
     */
    static void initialize_tracks_metadata()
    {
        const auto thread_id = std::nullopt;

        trace_cache::get_metadata_registry().ensure_track(
            "ainic_rx_rdma_ucast_bytes", [&] {
                return rocprofsys::trace_cache::info::track{
                    .track_name = "ainic_rx_rdma_ucast_bytes",
                    .thread_id  = thread_id,
                    .extdata    = "{}"
                };
            });
        trace_cache::get_metadata_registry().ensure_track(
            "ainic_tx_rdma_ucast_bytes", [&] {
                return rocprofsys::trace_cache::info::track{
                    .track_name = "ainic_tx_rdma_ucast_bytes",
                    .thread_id  = thread_id,
                    .extdata    = "{}"
                };
            });
        trace_cache::get_metadata_registry().ensure_track(
            "ainic_rx_rdma_ucast_pkts", [&] {
                return rocprofsys::trace_cache::info::track{
                    .track_name = "ainic_rx_rdma_ucast_pkts",
                    .thread_id  = thread_id,
                    .extdata    = "{}"
                };
            });
        trace_cache::get_metadata_registry().ensure_track(
            "ainic_tx_rdma_ucast_pkts", [&] {
                return rocprofsys::trace_cache::info::track{
                    .track_name = "ainic_tx_rdma_ucast_pkts",
                    .thread_id  = thread_id,
                    .extdata    = "{}"
                };
            });
        trace_cache::get_metadata_registry().ensure_track("ainic_rx_rdma_cnp_pkts", [&] {
            return rocprofsys::trace_cache::info::track{ .track_name =
                                                             "ainic_rx_rdma_cnp_pkts",
                                                         .thread_id = thread_id,
                                                         .extdata   = "{}" };
        });
        trace_cache::get_metadata_registry().ensure_track("ainic_tx_rdma_cnp_pkts", [&] {
            return rocprofsys::trace_cache::info::track{ .track_name =
                                                             "ainic_tx_rdma_cnp_pkts",
                                                         .thread_id = thread_id,
                                                         .extdata   = "{}" };
        });
        trace_cache::get_metadata_registry().ensure_track(
            "ainic_tx_rdma_ack_timeout", [&] {
                return rocprofsys::trace_cache::info::track{
                    .track_name = "ainic_tx_rdma_ack_timeout",
                    .thread_id  = thread_id,
                    .extdata    = "{}"
                };
            });
        trace_cache::get_metadata_registry().ensure_track(
            "ainic_resp_tx_pkt_seq_err", [&] {
                return rocprofsys::trace_cache::info::track{
                    .track_name = "ainic_resp_tx_pkt_seq_err",
                    .thread_id  = thread_id,
                    .extdata    = "{}"
                };
            });
        trace_cache::get_metadata_registry().ensure_track(
            "ainic_req_rx_pkt_seq_err", [&] {
                return rocprofsys::trace_cache::info::track{
                    .track_name = "ainic_req_rx_pkt_seq_err",
                    .thread_id  = thread_id,
                    .extdata    = "{}"
                };
            });
        trace_cache::get_metadata_registry().ensure_track(
            "ainic_req_rx_impl_nak_seq_err", [&] {
                return rocprofsys::trace_cache::info::track{
                    .track_name = "ainic_req_rx_impl_nak_seq_err",
                    .thread_id  = thread_id,
                    .extdata    = "{}"
                };
            });
    }

    /**
     * @brief Initialize per-device PMC metadata for NIC RDMA metrics.
     *
     * @param nic_id NIC device identifier
     * @param nic_name NIC device name (e.g., "enp226s0")
     */
    static void initialize_pmc_metadata(size_t                              nic_id,
                                        [[maybe_unused]] const std::string& nic_name)
    {
        constexpr size_t      k_event_code       = 0;
        constexpr size_t      k_instance_id      = 0;
        constexpr const char* k_long_description = "";
        constexpr const char* k_component        = "";
        constexpr const char* k_block            = "";
        constexpr const char* k_expression       = "";
        constexpr const char* k_target_arch      = "NIC";

        const auto add_nic_pmc = [&](const char* name, const char* symbol,
                                     const char* description, const char* units) {
            trace_cache::get_metadata_registry().add_pmc_info(
                { .type             = agent_type::nic,
                  .agent_type_index = nic_id,
                  .target_arch      = k_target_arch,
                  .event_code       = k_event_code,
                  .instance_id      = k_instance_id,
                  .name             = name,
                  .symbol           = symbol,
                  .description      = description,
                  .long_description = k_long_description,
                  .component        = k_component,
                  .units            = units,
                  .value_type       = rocprofsys::trace_cache::ABSOLUTE,
                  .block            = k_block,
                  .expression       = k_expression,
                  .is_constant      = 0,
                  .is_derived       = 0,
                  .extdata          = "{}" });
        };

        add_nic_pmc(
            trait::name<category::amd_smi_nic_rx_ucast_pkts>::value, "NIC RX UCast PKTS",
            trait::name<category::amd_smi_nic_rx_ucast_pkts>::description, "packets");
        add_nic_pmc(
            trait::name<category::amd_smi_nic_tx_ucast_pkts>::value, "NIC TX UCast PKTS",
            trait::name<category::amd_smi_nic_tx_ucast_pkts>::description, "packets");
        add_nic_pmc(
            trait::name<category::amd_smi_nic_rx_cnp_pkts>::value, "NIC RX CNP PKTS",
            trait::name<category::amd_smi_nic_rx_cnp_pkts>::description, "packets");
        add_nic_pmc(
            trait::name<category::amd_smi_nic_tx_cnp_pkts>::value, "NIC TX CNP PKTS",
            trait::name<category::amd_smi_nic_tx_cnp_pkts>::description, "packets");
        add_nic_pmc(trait::name<category::amd_smi_nic_rx_ucast_bytes>::value,
                    "NIC RX UCast Bytes",
                    trait::name<category::amd_smi_nic_rx_ucast_bytes>::description,
                    "bytes");
        add_nic_pmc(trait::name<category::amd_smi_nic_tx_ucast_bytes>::value,
                    "NIC TX UCast Bytes",
                    trait::name<category::amd_smi_nic_tx_ucast_bytes>::description,
                    "bytes");
        add_nic_pmc(trait::name<category::amd_smi_nic_tx_rdma_ack_timeout>::value,
                    "NIC TX RDMA ACK Timeout",
                    trait::name<category::amd_smi_nic_tx_rdma_ack_timeout>::description,
                    "timeouts");
        add_nic_pmc(trait::name<category::amd_smi_nic_resp_tx_pkt_seq_err>::value,
                    "NIC RESP TX PKT SEQ Error",
                    trait::name<category::amd_smi_nic_resp_tx_pkt_seq_err>::description,
                    "errors");
        add_nic_pmc(trait::name<category::amd_smi_nic_req_rx_pkt_seq_err>::value,
                    "NIC REQ RX PKT SEQ Error",
                    trait::name<category::amd_smi_nic_req_rx_pkt_seq_err>::description,
                    "errors");
        add_nic_pmc(
            trait::name<category::amd_smi_nic_req_rx_impl_nak_seq_err>::value,
            "NIC REQ RX Impl NAK SEQ Error",
            trait::name<category::amd_smi_nic_req_rx_impl_nak_seq_err>::description,
            "errors");
    }

    /**
     * @brief Store a NIC sample to the trace cache.
     *
     * @param device_id NIC device identifier
     * @param device_name NIC device name
     * @param enabled_metrics_cfg Metrics enabled by configuration
     * @param supported_metrics Metrics supported by this device
     * @param metric_values Collected metric values
     * @param timestamp Sample timestamp in nanoseconds
     */
    static void store_sample(size_t device_id, const std::string& device_name,
                             const enabled_metrics& enabled_metrics_cfg,
                             const enabled_metrics& supported_metrics,
                             const metrics& metric_values, std::uint64_t timestamp)
    {
        enabled_metrics active_metrics;
        active_metrics.value = enabled_metrics_cfg.value & supported_metrics.value;

        trace_cache::get_buffer_storage().store(trace_cache::ainic_pmc_sample{
            active_metrics, static_cast<std::uint32_t>(device_id), device_name, timestamp,
            metric_values });
    }
};

}  // namespace rocprofsys::pmc::collectors::nic

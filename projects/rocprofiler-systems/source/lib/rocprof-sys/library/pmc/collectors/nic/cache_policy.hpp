// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/nic/sample.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "logger/debug.hpp"

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

        trace_cache::get_metadata_registry().add_track(
            { "ainic_rx_rdma_ucast_bytes", thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { "ainic_tx_rdma_ucast_bytes", thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { "ainic_rx_rdma_ucast_pkts", thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { "ainic_tx_rdma_ucast_pkts", thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { "ainic_rx_rdma_cnp_pkts", thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { "ainic_tx_rdma_cnp_pkts", thread_id, "{}" });
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
        constexpr size_t      EVENT_CODE       = 0;
        constexpr size_t      INSTANCE_ID      = 0;
        constexpr const char* LONG_DESCRIPTION = "";
        constexpr const char* COMPONENT        = "";
        constexpr const char* BLOCK            = "";
        constexpr const char* EXPRESSION       = "";
        constexpr const char* TARGET_ARCH      = "";

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::NIC, nic_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_nic_rx_ucast_pkts>::value,
              "NIC RX UCast PKTS",
              trait::name<category::amd_smi_nic_rx_ucast_pkts>::description,
              LONG_DESCRIPTION, COMPONENT, "packets", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::NIC, nic_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_nic_tx_ucast_pkts>::value,
              "NIC TX UCast PKTS",
              trait::name<category::amd_smi_nic_tx_ucast_pkts>::description,
              LONG_DESCRIPTION, COMPONENT, "packets", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::NIC, nic_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_nic_rx_cnp_pkts>::value, "NIC RX CNP PKTS",
              trait::name<category::amd_smi_nic_rx_cnp_pkts>::description,
              LONG_DESCRIPTION, COMPONENT, "packets", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::NIC, nic_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_nic_tx_cnp_pkts>::value, "NIC TX CNP PKTS",
              trait::name<category::amd_smi_nic_tx_cnp_pkts>::description,
              LONG_DESCRIPTION, COMPONENT, "packets", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::NIC, nic_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_nic_rx_ucast_bytes>::value,
              "NIC RX UCast Bytes",
              trait::name<category::amd_smi_nic_rx_ucast_bytes>::description,
              LONG_DESCRIPTION, COMPONENT, "bytes", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::NIC, nic_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_nic_tx_ucast_bytes>::value,
              "NIC TX UCast Bytes",
              trait::name<category::amd_smi_nic_tx_ucast_bytes>::description,
              LONG_DESCRIPTION, COMPONENT, "bytes", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });
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
                             const metrics& metric_values, uint64_t timestamp)
    {
        enabled_metrics _enabled_metrics;
        _enabled_metrics.value = enabled_metrics_cfg.value & supported_metrics.value;

        trace_cache::get_buffer_storage().store(trace_cache::ainic_pmc_sample{
            _enabled_metrics, static_cast<uint32_t>(device_id), device_name, timestamp,
            metric_values });
    }
};

}  // namespace rocprofsys::pmc::collectors::nic

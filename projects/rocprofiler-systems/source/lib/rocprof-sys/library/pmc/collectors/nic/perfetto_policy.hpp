// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/perfetto.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <timemory/units.hpp>

#include <map>
#include <memory>
#include <spdlog/fmt/fmt.h>
#include <vector>

namespace rocprofsys::pmc::collectors::nic
{

namespace
{

struct nic_track_description
{
    const char* track_name;
    const char* units;
    size_t      track_index = 0;
};

// Helper function to create enabled_metrics value from bit positions
// See enabled_metrics definition in pmc/collectors/nic/types.hpp for bit position
// documentation
inline constexpr uint32_t
make_nic_metric_value(std::initializer_list<uint8_t> bit_positions)
{
    uint32_t value = 0;
    for(auto bit : bit_positions)
    {
        value |= (1u << bit);
    }
    return value;
}

const auto RX_RDMA_UCAST_BYTES_VALUE = make_nic_metric_value({ 0 });
const auto TX_RDMA_UCAST_BYTES_VALUE = make_nic_metric_value({ 1 });
const auto RX_RDMA_UCAST_PKTS_VALUE  = make_nic_metric_value({ 2 });
const auto TX_RDMA_UCAST_PKTS_VALUE  = make_nic_metric_value({ 3 });
const auto RX_RDMA_CNP_PKTS_VALUE    = make_nic_metric_value({ 4 });
const auto TX_RDMA_CNP_PKTS_VALUE    = make_nic_metric_value({ 5 });

struct nic_perfetto_sample
{
    size_t  timestamp;
    metrics metric_values;
};

}  // namespace

/**
 * @brief Output policy for writing NIC RDMA samples directly to Perfetto traces.
 *
 * This policy handles real-time serialization of NIC RDMA metric samples into
 * Perfetto trace format, creating counter tracks for each metric type.
 *
 * @see cache_policy for writing to trace cache instead
 */
struct perfetto_policy
{
    using counter_track = perfetto_counter_track<metrics>;

    // Static storage for Perfetto tracks and sample buffering (C++17 inline static)
    static inline std::map<size_t, std::map<uint32_t, nic_track_description>> tracks{};
    static inline std::map<size_t, std::unique_ptr<std::vector<nic_perfetto_sample>>>
        bundle{};

    /**
     * @brief Initialize Perfetto storage for the given NIC devices.
     *
     * Allocates storage buffers for Perfetto samples for each NIC device.
     *
     * @tparam DeviceVector Container type holding NIC device handles
     * @param devices Vector of NIC devices to initialize storage for
     */
    template <typename DeviceVector>
    static void init_storage(const DeviceVector& devices)
    {
        for(const auto& device : devices)
        {
            perfetto_policy::bundle.insert(
                { device->get_index(),
                  std::make_unique<std::vector<nic_perfetto_sample>>() });
        }
    }

    /**
     * @brief Set up Perfetto counter tracks for the specified NIC device metrics.
     *
     * Creates named counter tracks in the Perfetto trace for each enabled metric.
     *
     * @param device_index NIC device index
     * @param device_name NIC device name (e.g., "enp226s0")
     * @param enabled_metric_config Bitfield of metrics to create tracks for
     */
    static void setup_counter_tracks(size_t device_index, const std::string& device_name,
                                     const enabled_metrics& enabled_metric_config)
    {
        auto addendum = [&](const char* metric_name) {
            return fmt::format("NIC {} {} [{}] (S)", device_name, metric_name,
                               device_index);
        };

        auto& tracks = perfetto_policy::tracks[device_index];

        if(enabled_metric_config.bits.rx_rdma_ucast_bytes)
        {
            tracks[RX_RDMA_UCAST_BYTES_VALUE] = {
                "RX RDMA Bytes", "bytes",
                counter_track::emplace(device_index, addendum("RX RDMA Bytes"), "bytes")
            };
        }

        if(enabled_metric_config.bits.tx_rdma_ucast_bytes)
        {
            tracks[TX_RDMA_UCAST_BYTES_VALUE] = {
                "TX RDMA Bytes", "bytes",
                counter_track::emplace(device_index, addendum("TX RDMA Bytes"), "bytes")
            };
        }

        if(enabled_metric_config.bits.rx_rdma_ucast_pkts)
        {
            tracks[RX_RDMA_UCAST_PKTS_VALUE] = {
                "RX RDMA Pkts", "packets",
                counter_track::emplace(device_index, addendum("RX RDMA Pkts"), "packets")
            };
        }

        if(enabled_metric_config.bits.tx_rdma_ucast_pkts)
        {
            tracks[TX_RDMA_UCAST_PKTS_VALUE] = {
                "TX RDMA Pkts", "packets",
                counter_track::emplace(device_index, addendum("TX RDMA Pkts"), "packets")
            };
        }

        if(enabled_metric_config.bits.rx_rdma_cnp_pkts)
        {
            tracks[RX_RDMA_CNP_PKTS_VALUE] = {
                "RX CNP Pkts", "packets",
                counter_track::emplace(device_index, addendum("RX CNP Pkts"), "packets")
            };
        }

        if(enabled_metric_config.bits.tx_rdma_cnp_pkts)
        {
            tracks[TX_RDMA_CNP_PKTS_VALUE] = {
                "TX CNP Pkts", "packets",
                counter_track::emplace(device_index, addendum("TX CNP Pkts"), "packets")
            };
        }
    }

    /**
     * @brief Store a NIC sample for later Perfetto serialization.
     *
     * Buffers the metric sample for batch processing during post_process().
     *
     * @param device_index NIC device index
     * @param metric_values Collected metric values
     * @param timestamp Sample timestamp in nanoseconds
     */
    static void store_sample(size_t device_index, const metrics& metric_values,
                             uint64_t timestamp)
    {
        auto it = perfetto_policy::bundle.find(device_index);
        if(it != perfetto_policy::bundle.end())
        {
            it->second->emplace_back(nic_perfetto_sample{ timestamp, metric_values });
        }
    }

    /**
     * @brief Post-process buffered samples and write to Perfetto trace.
     *
     * Serializes all buffered NIC samples to Perfetto counter tracks.
     * This is called at the end of profiling to flush all samples.
     *
     * @tparam DeviceVector Container type holding NIC device handles
     * @param devices Vector of NIC devices
     * @param enabled_metrics Metrics that were enabled during collection
     */
    template <typename DeviceVector>
    static void post_process(
        const DeviceVector&                                 devices,
        ::rocprofsys::pmc::collectors::nic::enabled_metrics enabled_metrics)
    {
        for(const auto& device : devices)
        {
            post_process_device(device->get_index(), enabled_metrics,
                                device->get_supported_metrics());
        }
    }

    static void post_process_device(
        size_t                                              device_index,
        ::rocprofsys::pmc::collectors::nic::enabled_metrics enabled_metrics,
        ::rocprofsys::pmc::collectors::nic::enabled_metrics supported_metrics)
    {
        auto bundle_it = perfetto_policy::bundle.find(device_index);
        if(bundle_it == perfetto_policy::bundle.end() || !bundle_it->second)
        {
            return;
        }

        auto& samples = *bundle_it->second;

        const auto& thread_info = thread_info::get(0, InternalTID);
        if(!thread_info)
        {
            return;
        }

        ::rocprofsys::pmc::collectors::nic::enabled_metrics effective_metrics = {
            .value =
                static_cast<uint32_t>(enabled_metrics.value & supported_metrics.value)
        };

        if(effective_metrics.value == 0)
        {
            return;
        }

        auto tracks_it = perfetto_policy::tracks.find(device_index);
        if(tracks_it == perfetto_policy::tracks.end())
        {
            return;
        }

        auto& tracks = tracks_it->second;

        for(const auto& sample : samples)
        {
            const auto ts = sample.timestamp;

            if(!thread_info->is_valid_time(ts))
            {
                LOG_WARNING("Invalid timestamp {} for NIC sample", ts);
                continue;
            }

            // RX RDMA unicast bytes
            if(effective_metrics.bits.rx_rdma_ucast_bytes)
            {
                auto it = tracks.find(RX_RDMA_UCAST_BYTES_VALUE);
                if(it != tracks.end())
                {
                    TRACE_COUNTER(
                        "nic_rx_ucast_bytes",
                        counter_track::at(device_index, it->second.track_index), ts,
                        static_cast<double>(sample.metric_values.rx_rdma_ucast_bytes));
                }
            }

            // TX RDMA unicast bytes
            if(effective_metrics.bits.tx_rdma_ucast_bytes)
            {
                auto it = tracks.find(TX_RDMA_UCAST_BYTES_VALUE);
                if(it != tracks.end())
                {
                    TRACE_COUNTER(
                        "nic_tx_ucast_bytes",
                        counter_track::at(device_index, it->second.track_index), ts,
                        static_cast<double>(sample.metric_values.tx_rdma_ucast_bytes));
                }
            }

            // RX RDMA unicast packets
            if(effective_metrics.bits.rx_rdma_ucast_pkts)
            {
                auto it = tracks.find(RX_RDMA_UCAST_PKTS_VALUE);
                if(it != tracks.end())
                {
                    TRACE_COUNTER(
                        "nic_rx_ucast_pkts",
                        counter_track::at(device_index, it->second.track_index), ts,
                        static_cast<double>(sample.metric_values.rx_rdma_ucast_pkts));
                }
            }

            // TX RDMA unicast packets
            if(effective_metrics.bits.tx_rdma_ucast_pkts)
            {
                auto it = tracks.find(TX_RDMA_UCAST_PKTS_VALUE);
                if(it != tracks.end())
                {
                    TRACE_COUNTER(
                        "nic_tx_ucast_pkts",
                        counter_track::at(device_index, it->second.track_index), ts,
                        static_cast<double>(sample.metric_values.tx_rdma_ucast_pkts));
                }
            }

            // RX RDMA CNP packets
            if(effective_metrics.bits.rx_rdma_cnp_pkts)
            {
                auto it = tracks.find(RX_RDMA_CNP_PKTS_VALUE);
                if(it != tracks.end())
                {
                    TRACE_COUNTER(
                        "nic_rx_cnp_pkts",
                        counter_track::at(device_index, it->second.track_index), ts,
                        static_cast<double>(sample.metric_values.rx_rdma_cnp_pkts));
                }
            }

            // TX RDMA CNP packets
            if(effective_metrics.bits.tx_rdma_cnp_pkts)
            {
                auto it = tracks.find(TX_RDMA_CNP_PKTS_VALUE);
                if(it != tracks.end())
                {
                    TRACE_COUNTER(
                        "nic_tx_cnp_pkts",
                        counter_track::at(device_index, it->second.track_index), ts,
                        static_cast<double>(sample.metric_values.tx_rdma_cnp_pkts));
                }
            }
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::nic

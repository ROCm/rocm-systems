// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <cstdint>

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
inline constexpr std::uint32_t
make_nic_metric_value(std::initializer_list<std::uint8_t> bit_positions)
{
    std::uint32_t value = 0;
    for(auto bit : bit_positions)
    {
        value |= (1u << bit);
    }
    return value;
}

inline constexpr auto RX_RDMA_UCAST_BYTES_VALUE     = make_nic_metric_value({ 0 });
inline constexpr auto TX_RDMA_UCAST_BYTES_VALUE     = make_nic_metric_value({ 1 });
inline constexpr auto RX_RDMA_UCAST_PKTS_VALUE      = make_nic_metric_value({ 2 });
inline constexpr auto TX_RDMA_UCAST_PKTS_VALUE      = make_nic_metric_value({ 3 });
inline constexpr auto RX_RDMA_CNP_PKTS_VALUE        = make_nic_metric_value({ 4 });
inline constexpr auto TX_RDMA_CNP_PKTS_VALUE        = make_nic_metric_value({ 5 });
inline constexpr auto TX_RDMA_ACK_TIMEOUT_VALUE     = make_nic_metric_value({ 6 });
inline constexpr auto RESP_TX_PKT_SEQ_ERR_VALUE     = make_nic_metric_value({ 7 });
inline constexpr auto REQ_RX_PKT_SEQ_ERR_VALUE      = make_nic_metric_value({ 8 });
inline constexpr auto REQ_RX_IMPL_NAK_SEQ_ERR_VALUE = make_nic_metric_value({ 9 });

// Single source of truth for NIC track labels and units. Adding a new metric
// requires exactly one new row here (plus the bit position above and the
// corresponding member in the metrics/enabled_metrics structs).
inline std::map<std::uint32_t, nic_track_description>
make_default_nic_tracks()
{
    return {
        { RX_RDMA_UCAST_BYTES_VALUE, { "RX RDMA BYTES", "bytes", 0 } },
        { TX_RDMA_UCAST_BYTES_VALUE, { "TX RDMA BYTES", "bytes", 0 } },
        { RX_RDMA_UCAST_PKTS_VALUE, { "RX RDMA PACKETS", "packets", 0 } },
        { TX_RDMA_UCAST_PKTS_VALUE, { "TX RDMA PACKETS", "packets", 0 } },
        { RX_RDMA_CNP_PKTS_VALUE, { "RX CNP PACKETS", "packets", 0 } },
        { TX_RDMA_CNP_PKTS_VALUE, { "TX CNP PACKETS", "packets", 0 } },
        { TX_RDMA_ACK_TIMEOUT_VALUE, { "TX ACK TIMEOUT", "timeouts", 0 } },
        { RESP_TX_PKT_SEQ_ERR_VALUE, { "RESP TX PKT SEQ ERR", "errors", 0 } },
        { REQ_RX_PKT_SEQ_ERR_VALUE, { "REQ RX PKT SEQ ERR", "errors", 0 } },
        { REQ_RX_IMPL_NAK_SEQ_ERR_VALUE, { "REQ RX IMPL NAK SEQ ERR", "errors", 0 } },
    };
}

struct nic_perfetto_sample
{
    size_t  timestamp;
    metrics metric_values;
};

// Emit a single NIC counter sample. CategoryTp is a registered Perfetto category
// trait (see core/categories.hpp); trait::name<CategoryTp>::value is a
// compile-time literal, which is what TRACE_COUNTER's category argument requires
// (a runtime string cannot be used). This replaces a former per-metric emit macro
// with a type-checked inline function.
template <typename CategoryTp>
inline void
emit_nic_counter(const enabled_metrics& effective_metrics, std::uint32_t bit_key,
                 size_t                                                device_index,
                 const std::map<std::uint32_t, nic_track_description>& device_tracks,
                 size_t ts, std::uint64_t value)
{
    if((effective_metrics.value & bit_key) == 0) return;
    auto it = device_tracks.find(bit_key);
    if(it == device_tracks.end()) return;
    TRACE_COUNTER(
        trait::name<CategoryTp>::value,
        perfetto_counter_track<metrics>::at(device_index, it->second.track_index), ts,
        static_cast<double>(value));
}

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
    static inline std::map<size_t, std::map<std::uint32_t, nic_track_description>>
        tracks{};
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

        auto& device_tracks = perfetto_policy::tracks[device_index];

        for(auto& [bit_value, description] : make_default_nic_tracks())
        {
            if((enabled_metric_config.value & bit_value) == 0) continue;
            description.track_index = counter_track::emplace(
                device_index, addendum(description.track_name), description.units);
            device_tracks[bit_value] = description;
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
                             std::uint64_t timestamp)
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

        ::rocprofsys::pmc::collectors::nic::enabled_metrics effective_metrics{};
        effective_metrics.value =
            static_cast<std::uint32_t>(enabled_metrics.value & supported_metrics.value);

        if(effective_metrics.value == 0)
        {
            return;
        }

        auto tracks_it = perfetto_policy::tracks.find(device_index);
        if(tracks_it == perfetto_policy::tracks.end())
        {
            return;
        }

        auto& device_tracks = tracks_it->second;

        for(const auto& sample : samples)
        {
            const auto ts = sample.timestamp;

            if(!thread_info->is_valid_time(ts))
            {
                LOG_WARNING("Invalid timestamp {} for NIC sample", ts);
                continue;
            }

            // Emit one counter sample per enabled metric. The category travels as a
            // registered category trait so TRACE_COUNTER still sees a compile-time
            // literal; see emit_nic_counter above.
            emit_nic_counter<category::amd_smi_nic_rx_ucast_bytes>(
                effective_metrics, RX_RDMA_UCAST_BYTES_VALUE, device_index, device_tracks,
                ts, sample.metric_values.rx_rdma_ucast_bytes);
            emit_nic_counter<category::amd_smi_nic_tx_ucast_bytes>(
                effective_metrics, TX_RDMA_UCAST_BYTES_VALUE, device_index, device_tracks,
                ts, sample.metric_values.tx_rdma_ucast_bytes);
            emit_nic_counter<category::amd_smi_nic_rx_ucast_pkts>(
                effective_metrics, RX_RDMA_UCAST_PKTS_VALUE, device_index, device_tracks,
                ts, sample.metric_values.rx_rdma_ucast_pkts);
            emit_nic_counter<category::amd_smi_nic_tx_ucast_pkts>(
                effective_metrics, TX_RDMA_UCAST_PKTS_VALUE, device_index, device_tracks,
                ts, sample.metric_values.tx_rdma_ucast_pkts);
            emit_nic_counter<category::amd_smi_nic_rx_cnp_pkts>(
                effective_metrics, RX_RDMA_CNP_PKTS_VALUE, device_index, device_tracks,
                ts, sample.metric_values.rx_rdma_cnp_pkts);
            emit_nic_counter<category::amd_smi_nic_tx_cnp_pkts>(
                effective_metrics, TX_RDMA_CNP_PKTS_VALUE, device_index, device_tracks,
                ts, sample.metric_values.tx_rdma_cnp_pkts);
            emit_nic_counter<category::amd_smi_nic_tx_rdma_ack_timeout>(
                effective_metrics, TX_RDMA_ACK_TIMEOUT_VALUE, device_index, device_tracks,
                ts, sample.metric_values.tx_rdma_ack_timeout);
            emit_nic_counter<category::amd_smi_nic_resp_tx_pkt_seq_err>(
                effective_metrics, RESP_TX_PKT_SEQ_ERR_VALUE, device_index, device_tracks,
                ts, sample.metric_values.resp_tx_pkt_seq_err);
            emit_nic_counter<category::amd_smi_nic_req_rx_pkt_seq_err>(
                effective_metrics, REQ_RX_PKT_SEQ_ERR_VALUE, device_index, device_tracks,
                ts, sample.metric_values.req_rx_pkt_seq_err);
            emit_nic_counter<category::amd_smi_nic_req_rx_impl_nak_seq_err>(
                effective_metrics, REQ_RX_IMPL_NAK_SEQ_ERR_VALUE, device_index,
                device_tracks, ts, sample.metric_values.req_rx_impl_nak_seq_err);
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::nic

// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <cstddef>
#include <cstdint>

#include <fmt/format.h>
#include <map>
#include <memory>
#include <vector>

namespace rocprofsys::pmc::collectors::nic
{

struct nic_track_description
{
    const char* track_name;
    const char* units;
};

inline constexpr auto RX_RDMA_UCAST_BYTES_VALUE     = std::uint32_t(1) << 0;
inline constexpr auto TX_RDMA_UCAST_BYTES_VALUE     = std::uint32_t(1) << 1;
inline constexpr auto RX_RDMA_UCAST_PKTS_VALUE      = std::uint32_t(1) << 2;
inline constexpr auto TX_RDMA_UCAST_PKTS_VALUE      = std::uint32_t(1) << 3;
inline constexpr auto RX_RDMA_CNP_PKTS_VALUE        = std::uint32_t(1) << 4;
inline constexpr auto TX_RDMA_CNP_PKTS_VALUE        = std::uint32_t(1) << 5;
inline constexpr auto TX_RDMA_ACK_TIMEOUT_VALUE     = std::uint32_t(1) << 6;
inline constexpr auto RESP_TX_PKT_SEQ_ERR_VALUE     = std::uint32_t(1) << 7;
inline constexpr auto REQ_RX_PKT_SEQ_ERR_VALUE      = std::uint32_t(1) << 8;
inline constexpr auto REQ_RX_IMPL_NAK_SEQ_ERR_VALUE = std::uint32_t(1) << 9;

// Default track labels and units, keyed by metric bit value. Adding a metric
// touches several ordered lists that must stay in sync: the *_VALUE bit constant
// above, a row here, the enabled_metrics/metrics members in types.hpp, and an
// emit_nic_counter<> call in post_process_device.
inline const std::map<std::uint32_t, nic_track_description>&
make_default_nic_tracks()
{
    static const std::map<std::uint32_t, nic_track_description> tracks = {
        { RX_RDMA_UCAST_BYTES_VALUE,
          { .track_name = "RX RDMA BYTES", .units = "bytes" } },
        { TX_RDMA_UCAST_BYTES_VALUE,
          { .track_name = "TX RDMA BYTES", .units = "bytes" } },
        { RX_RDMA_UCAST_PKTS_VALUE,
          { .track_name = "RX RDMA PACKETS", .units = "packets" } },
        { TX_RDMA_UCAST_PKTS_VALUE,
          { .track_name = "TX RDMA PACKETS", .units = "packets" } },
        { RX_RDMA_CNP_PKTS_VALUE,
          { .track_name = "RX CNP PACKETS", .units = "packets" } },
        { TX_RDMA_CNP_PKTS_VALUE,
          { .track_name = "TX CNP PACKETS", .units = "packets" } },
        { TX_RDMA_ACK_TIMEOUT_VALUE,
          { .track_name = "TX ACK TIMEOUT", .units = "timeouts" } },
        { RESP_TX_PKT_SEQ_ERR_VALUE,
          { .track_name = "RESP TX PKT SEQ ERR", .units = "errors" } },
        { REQ_RX_PKT_SEQ_ERR_VALUE,
          { .track_name = "REQ RX PKT SEQ ERR", .units = "errors" } },
        { REQ_RX_IMPL_NAK_SEQ_ERR_VALUE,
          { .track_name = "REQ RX IMPL NAK SEQ ERR", .units = "errors" } },
    };
    return tracks;
}

struct nic_perfetto_sample
{
    std::uint64_t timestamp;
    metrics       metric_values;
};

// Emit a single NIC counter sample to a pre-resolved track. CategoryTp is a
// registered Perfetto category trait (see core/categories.hpp);
// trait::name<CategoryTp>::value is a compile-time literal, which is what
// TRACE_COUNTER's category argument requires (a runtime string cannot be used).
// track_index < 0 means the metric is disabled or has no track, so the emit is
// skipped; callers resolve it once (see resolve_nic_track) instead of looking it
// up per sample.
template <typename CategoryTp>
inline void
emit_nic_counter(size_t device_index, std::int64_t track_index, std::uint64_t ts,
                 std::uint64_t value)
{
    if(track_index < 0) return;
    TRACE_COUNTER(trait::name<CategoryTp>::value,
                  perfetto_counter_track<metrics>::at(device_index,
                                                      static_cast<size_t>(track_index)),
                  ts, static_cast<double>(value));
}

// Resolve a metric's track index once (before the per-sample loop). Returns -1 if
// the metric is not enabled or has no registered track for this device.
inline std::int64_t
resolve_nic_track(const enabled_metrics& effective_metrics, std::uint32_t bit_key,
                  const std::map<std::uint32_t, size_t>& device_tracks)
{
    if((effective_metrics.value & bit_key) == 0) return -1;
    auto it = device_tracks.find(bit_key);
    if(it == device_tracks.end()) return -1;
    return static_cast<std::int64_t>(it->second);
}

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

    // Static storage for Perfetto tracks and sample buffering (C++17 inline static).
    // Per device: metric bit value -> resolved counter-track index.
    static inline std::map<size_t, std::map<std::uint32_t, size_t>> tracks{};
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

        for(const auto& [bit_value, description] : make_default_nic_tracks())
        {
            if((enabled_metric_config.value & bit_value) == 0) continue;
            device_tracks[bit_value] = counter_track::emplace(
                device_index, addendum(description.track_name), description.units);
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
            it->second->emplace_back(nic_perfetto_sample{
                .timestamp = timestamp, .metric_values = metric_values });
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
     * @param enabled Metrics that were enabled during collection
     */
    template <typename DeviceVector>
    static void post_process(const DeviceVector&                                 devices,
                             ::rocprofsys::pmc::collectors::nic::enabled_metrics enabled)
    {
        for(const auto& device : devices)
        {
            post_process_device(device->get_index(), enabled,
                                device->get_supported_metrics());
        }
    }

    static void post_process_device(
        size_t device_index, ::rocprofsys::pmc::collectors::nic::enabled_metrics enabled,
        ::rocprofsys::pmc::collectors::nic::enabled_metrics supported)
    {
        auto bundle_it = perfetto_policy::bundle.find(device_index);
        if(bundle_it == perfetto_policy::bundle.end() || !bundle_it->second)
        {
            return;
        }

        auto& samples = *bundle_it->second;

        const auto& tinfo = thread_info::get(0, InternalTID);
        if(!tinfo)
        {
            return;
        }

        ::rocprofsys::pmc::collectors::nic::enabled_metrics effective_metrics{};
        effective_metrics.value =
            static_cast<std::uint32_t>(enabled.value & supported.value);

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

        // Resolve each enabled metric's track index once; device_tracks is fixed
        // after setup, so there is no need to look it up per sample.
        const auto rx_rdma_ucast_bytes_idx = resolve_nic_track(
            effective_metrics, RX_RDMA_UCAST_BYTES_VALUE, device_tracks);
        const auto tx_rdma_ucast_bytes_idx = resolve_nic_track(
            effective_metrics, TX_RDMA_UCAST_BYTES_VALUE, device_tracks);
        const auto rx_rdma_ucast_pkts_idx =
            resolve_nic_track(effective_metrics, RX_RDMA_UCAST_PKTS_VALUE, device_tracks);
        const auto tx_rdma_ucast_pkts_idx =
            resolve_nic_track(effective_metrics, TX_RDMA_UCAST_PKTS_VALUE, device_tracks);
        const auto rx_rdma_cnp_pkts_idx =
            resolve_nic_track(effective_metrics, RX_RDMA_CNP_PKTS_VALUE, device_tracks);
        const auto tx_rdma_cnp_pkts_idx =
            resolve_nic_track(effective_metrics, TX_RDMA_CNP_PKTS_VALUE, device_tracks);
        const auto tx_rdma_ack_timeout_idx = resolve_nic_track(
            effective_metrics, TX_RDMA_ACK_TIMEOUT_VALUE, device_tracks);
        const auto resp_tx_pkt_seq_err_idx = resolve_nic_track(
            effective_metrics, RESP_TX_PKT_SEQ_ERR_VALUE, device_tracks);
        const auto req_rx_pkt_seq_err_idx =
            resolve_nic_track(effective_metrics, REQ_RX_PKT_SEQ_ERR_VALUE, device_tracks);
        const auto req_rx_impl_nak_seq_err_idx = resolve_nic_track(
            effective_metrics, REQ_RX_IMPL_NAK_SEQ_ERR_VALUE, device_tracks);

        for(const auto& sample : samples)
        {
            const auto ts = sample.timestamp;

            if(!tinfo->is_valid_time(ts))
            {
                LOG_WARNING("Invalid timestamp {} for NIC sample", ts);
                continue;
            }

            // Emit one counter sample per enabled metric, using the track indices
            // resolved above. The category travels as a registered category trait so
            // TRACE_COUNTER still sees a compile-time literal; see emit_nic_counter.
            emit_nic_counter<category::amd_smi_nic_rx_ucast_bytes>(
                device_index, rx_rdma_ucast_bytes_idx, ts,
                sample.metric_values.rx_rdma_ucast_bytes);
            emit_nic_counter<category::amd_smi_nic_tx_ucast_bytes>(
                device_index, tx_rdma_ucast_bytes_idx, ts,
                sample.metric_values.tx_rdma_ucast_bytes);
            emit_nic_counter<category::amd_smi_nic_rx_ucast_pkts>(
                device_index, rx_rdma_ucast_pkts_idx, ts,
                sample.metric_values.rx_rdma_ucast_pkts);
            emit_nic_counter<category::amd_smi_nic_tx_ucast_pkts>(
                device_index, tx_rdma_ucast_pkts_idx, ts,
                sample.metric_values.tx_rdma_ucast_pkts);
            emit_nic_counter<category::amd_smi_nic_rx_cnp_pkts>(
                device_index, rx_rdma_cnp_pkts_idx, ts,
                sample.metric_values.rx_rdma_cnp_pkts);
            emit_nic_counter<category::amd_smi_nic_tx_cnp_pkts>(
                device_index, tx_rdma_cnp_pkts_idx, ts,
                sample.metric_values.tx_rdma_cnp_pkts);
            emit_nic_counter<category::amd_smi_nic_tx_rdma_ack_timeout>(
                device_index, tx_rdma_ack_timeout_idx, ts,
                sample.metric_values.tx_rdma_ack_timeout);
            emit_nic_counter<category::amd_smi_nic_resp_tx_pkt_seq_err>(
                device_index, resp_tx_pkt_seq_err_idx, ts,
                sample.metric_values.resp_tx_pkt_seq_err);
            emit_nic_counter<category::amd_smi_nic_req_rx_pkt_seq_err>(
                device_index, req_rx_pkt_seq_err_idx, ts,
                sample.metric_values.req_rx_pkt_seq_err);
            emit_nic_counter<category::amd_smi_nic_req_rx_impl_nak_seq_err>(
                device_index, req_rx_impl_nak_seq_err_idx, ts,
                sample.metric_values.req_rx_impl_nak_seq_err);
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::nic

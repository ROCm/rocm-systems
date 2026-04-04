// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/perfetto.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/thread_info.hpp"

#include <cstddef>
#include <cstdint>
#include <spdlog/fmt/fmt.h>
#include <timemory/units.hpp>

#include <vector>

namespace rocprofsys::pmc::collectors::gpu
{

namespace detail
{

struct track_description
{
    const char*         track_name;
    const char*         units;
    std::vector<size_t> track_indexes;
};

// Helper function to create enabled_metrics value from bit positions
// See enabled_metrics definition in pmc/collectors/gpu/types.hpp for bit position
// documentation
inline constexpr uint32_t
make_metric_value(std::initializer_list<uint8_t> bit_positions)
{
    uint32_t value = 0;
    for(auto bit : bit_positions)
    {
        value |= (1u << bit);
    }
    return value;
}

const auto GFX_BUSY_VALUE      = make_metric_value({ 5 });     // gfx_activity
const auto UMC_BUSY_VALUE      = make_metric_value({ 6 });     // umc_activity
const auto MM_BUSY_VALUE       = make_metric_value({ 7 });     // mm_activity
const auto TEMPERATURE_VALUE   = make_metric_value({ 3, 4 });  // hotspot, edge
const auto CURRENT_POWER_VALUE = make_metric_value({ 0, 1 });  // current, average
const auto MEMORY_USAGE_VALUE  = make_metric_value({ 2 });     // memory_usage
const auto VCN_ACTIVITY_VALUE  = make_metric_value({ 8 });     // vcn_activity
const auto JPEG_ACTIVITY_VALUE = make_metric_value({ 9 });     // jpeg_activity
const auto VCN_BUSY_VALUE      = make_metric_value({ 10 });    // vcn_busy (MI300)
const auto JPEG_BUSY_VALUE     = make_metric_value({ 11 });    // jpeg_busy (MI300)
const auto XGMI_VALUE          = make_metric_value({ 12 });    // xgmi
const auto PCIE_VALUE          = make_metric_value({ 13 });    // pcie
const auto SDMA_USAGE_VALUE    = make_metric_value({ 14 });    // sdma_usage

inline std::unordered_map<uint32_t, track_description>
make_default_tracks()
{
    return {
        { GFX_BUSY_VALUE, { "GFX Busy", "%", {} } },
        { UMC_BUSY_VALUE, { "UMC Busy", "%", {} } },
        { MM_BUSY_VALUE, { "MM Busy", "%", {} } },
        { TEMPERATURE_VALUE, { "Temperature", "deg C", {} } },
        { CURRENT_POWER_VALUE, { "Current Power", "watts", {} } },
        { MEMORY_USAGE_VALUE, { "Memory Usage", "megabytes", {} } },
        { VCN_ACTIVITY_VALUE, { "VCN Activity", "%", {} } },
        { JPEG_ACTIVITY_VALUE, { "JPEG Activity", "%", {} } },
        { VCN_BUSY_VALUE, { "VCN Busy", "%", {} } },
        { JPEG_BUSY_VALUE, { "JPEG Busy", "%", {} } },
        { XGMI_VALUE, { "XGMI", "", {} } },
        { PCIE_VALUE, { "PCIe", "", {} } },
        { SDMA_USAGE_VALUE, { "SDMA Usage", "%", {} } },
    };
}

struct xgmi_track_set
{
    std::vector<size_t> link_width;
    std::vector<size_t> link_speed;
    std::vector<size_t> read_data;
    std::vector<size_t> write_data;
};

struct pcie_track_set
{
    std::vector<size_t> link_width;
    std::vector<size_t> link_speed;
    std::vector<size_t> bandwidth_acc;
    std::vector<size_t> bandwidth_inst;
};

struct perfetto_amd_smi_sample
{
    uint64_t                      timestamp;
    pmc::collectors::gpu::metrics metrics;
};

struct perfetto_device_data
{
    std::unique_ptr<std::vector<perfetto_amd_smi_sample>> samples;
    enabled_metrics                                       supported_metrics;
    std::unordered_map<uint32_t, track_description>       tracks;
    xgmi_track_set                                        xgmi_tracks;
    pcie_track_set                                        pcie_tracks;
};

inline std::map<size_t, perfetto_device_data>&
get_perfetto_data()
{
    static std::map<size_t, perfetto_device_data> data;
    return data;
}

}  // namespace detail

/**
 * @brief Output policy for writing PMC samples directly to Perfetto traces.
 *
 * This policy handles real-time serialization of AMD SMI metric samples into
 * Perfetto trace format, creating counter tracks for each metric type.
 * Supports both device-level metrics (Radeon) and per-XCP metrics (MI300 series).
 *
 * @see cache_policy for writing to trace cache instead
 */
struct perfetto_policy
{
    using counter_track = perfetto_counter_track<metrics>;

    /**
     * @brief Initialize Perfetto storage for the given device entries.
     *
     * Allocates storage buffers for Perfetto samples for each GPU device
     * and caches supported metrics for post-processing.
     *
     * @tparam DeviceEntryVector Container type holding device entries (device +
     * supported_metrics)
     * @param device_entries Vector of device entries to initialize storage for
     */
    template <typename DeviceEntryVector>
    static void init_storage(const DeviceEntryVector& device_entries)
    {
        for(const auto& entry : device_entries)
        {
            auto idx                         = entry.device->get_index();
            detail::get_perfetto_data()[idx] = {
                std::make_unique<std::vector<detail::perfetto_amd_smi_sample>>(),
                entry.supported_metrics,
                detail::make_default_tracks(),
                {},
                {}
            };
        }
    }

    /**
     * @brief Set up Perfetto counter tracks for the specified device metrics.
     *
     * Creates named counter tracks in the Perfetto trace for each enabled metric,
     * handling both simple metrics and array metrics (VCN, JPEG, XGMI, PCIe).
     *
     * @param device_index GPU device index
     * @param enabled_metric_config Bitfield of metrics to create tracks for
     */
    static void setup_counter_tracks(size_t                 device_index,
                                     const enabled_metrics& enabled_metric_config)
    {
        auto addendum = [&](const char* name) {
            return fmt::format("GPU {} [{}] (S)", name, device_index);
        };

        auto addendum_blk = [&](std::size_t i, const char* metric,
                                std::size_t xcp_idx = SIZE_MAX) {
            if(xcp_idx != SIZE_MAX)
            {
                return fmt::format("GPU [{}] {} XCP_{}: [{:02d}] (S)", device_index,
                                   metric, xcp_idx, i);
            }
            return fmt::format("GPU [{}] {} [{:02d}] (S)", device_index, metric, i);
        };

        auto& device_data = detail::get_perfetto_data()[device_index];
        auto& tracks      = device_data.tracks;

        // Clear track indexes from previous setup calls to prevent
        // stale track IDs when metric configuration changes between runs
        for(auto& [_, description] : tracks)
        {
            description.track_indexes.clear();
        }
        device_data.xgmi_tracks = {};
        device_data.pcie_tracks = {};

        LOG_DEBUG("[GPU perfetto_policy] Setting up counter tracks for device {}, "
                  "enabled_metrics=0x{:x}",
                  device_index, enabled_metric_config.value);

        for(auto& [num, description] : tracks)
        {
            auto enabled_metric = num & enabled_metric_config.value;
            if(enabled_metric == 0)
            {
                continue;
            }

            const auto process_xcp_array = [&](detail::track_description& desc,
                                               size_t array_size, size_t xcp_id) {
                for(std::size_t i = 0; i < array_size; ++i)
                {
                    const auto track_id = counter_track::emplace(
                        device_index, addendum_blk(i, desc.track_name, xcp_id),
                        desc.units);
                    desc.track_indexes.emplace_back(track_id);
                }
            };

            if(enabled_metric == detail::VCN_BUSY_VALUE ||
               enabled_metric == detail::JPEG_BUSY_VALUE)
            {
                // Per-XCP metrics (MI300): create tracks for each XCP partition
                auto array_size = (enabled_metric == detail::VCN_BUSY_VALUE)
                                      ? AMDSMI_MAX_NUM_VCN
                                      : ROCPROFSYS_AMDSMI_JPEG_ENGINE_COUNT;
                for(std::size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
                {
                    process_xcp_array(description, array_size, xcp);
                }
            }
            else if(enabled_metric == detail::VCN_ACTIVITY_VALUE ||
                    enabled_metric == detail::JPEG_ACTIVITY_VALUE)
            {
                // Device-level metrics (Radeon): flat array, no XCP dimension
                auto array_size = (enabled_metric == detail::VCN_ACTIVITY_VALUE)
                                      ? AMDSMI_MAX_NUM_VCN
                                      : ROCPROFSYS_AMDSMI_JPEG_ENGINE_COUNT;
                for(std::size_t i = 0; i < array_size; ++i)
                {
                    description.track_indexes.emplace_back(counter_track::emplace(
                        device_index, addendum_blk(i, description.track_name),
                        description.units));
                }
            }
            else
            {
                description.track_indexes.emplace_back(counter_track::emplace(
                    device_index, addendum(description.track_name), description.units));
            }
        }

        if(enabled_metric_config.bits.xgmi)
        {
            auto& xgmi_tracks = device_data.xgmi_tracks;

            xgmi_tracks.link_width.emplace_back(counter_track::emplace(
                device_index, addendum("XGMI Link Width"), "lanes"));
            xgmi_tracks.link_speed.emplace_back(counter_track::emplace(
                device_index, addendum("XGMI Link Speed"), "Mbps"));

            for(std::size_t link = 0; link < AMDSMI_MAX_NUM_XGMI_LINKS; ++link)
            {
                xgmi_tracks.read_data.emplace_back(counter_track::emplace(
                    device_index, addendum_blk(link, "XGMI Read Data"), "KB"));
                xgmi_tracks.write_data.emplace_back(counter_track::emplace(
                    device_index, addendum_blk(link, "XGMI Write Data"), "KB"));
            }
        }

        if(enabled_metric_config.bits.pcie)
        {
            auto& pcie_tracks = device_data.pcie_tracks;

            pcie_tracks.link_width.emplace_back(counter_track::emplace(
                device_index, addendum("PCIe Link Width"), "lanes"));
            pcie_tracks.link_speed.emplace_back(counter_track::emplace(
                device_index, addendum("PCIe Link Speed"), "MT/s"));
            pcie_tracks.bandwidth_acc.emplace_back(counter_track::emplace(
                device_index, addendum("PCIe Bandwidth Acc"), "bytes"));
            pcie_tracks.bandwidth_inst.emplace_back(counter_track::emplace(
                device_index, addendum("PCIe Bandwidth Inst"), "bytes/s"));
        }
    }

    /**
     * @brief Store a PMC sample for later Perfetto serialization.
     *
     * Buffers the metric sample for batch processing during post_process().
     *
     * @param device_index GPU device index
     * @param metric_values Collected metric values
     * @param timestamp Sample timestamp in nanoseconds
     */
    static void store_sample(size_t device_index, const metrics& metric_values,
                             uint64_t timestamp)
    {
        detail::get_perfetto_data()[device_index].samples->emplace_back(
            detail::perfetto_amd_smi_sample{ timestamp, metric_values });
    }

    /**
     * @brief Post-process buffered samples and write to Perfetto trace.
     *
     * Serializes all buffered PMC samples to Perfetto counter tracks.
     * This is called at the end of profiling to flush all samples.
     * Supported metrics are retrieved from the cached device data.
     *
     * @param enabled_metrics Metrics that were enabled during collection
     */
    static void post_process(pmc::collectors::gpu::enabled_metrics enabled_metrics_cfg)
    {
        for(const auto& [device_index, data] : detail::get_perfetto_data())
        {
            post_process_device(device_index, enabled_metrics_cfg,
                                data.supported_metrics);
        }
    }

private:
    static void post_process_device(
        size_t device_index, pmc::collectors::gpu::enabled_metrics enabled_metrics_cfg,
        pmc::collectors::gpu::enabled_metrics supported_metrics)
    {
        auto& samples = *detail::get_perfetto_data()[device_index].samples;

        LOG_DEBUG("[GPU perfetto_policy] Post-processing {} PMC samples for device [{}], "
                  "enabled=0x{:x}, supported=0x{:x}",
                  samples.size(), device_index, enabled_metrics_cfg.value,
                  supported_metrics.value);

        const auto& thread_info = thread_info::get(0, InternalTID);
        if(!thread_info)
        {
            return;
        }

        pmc::collectors::gpu::enabled_metrics effective_metrics;
        effective_metrics.value =
            static_cast<uint32_t>(enabled_metrics_cfg.value & supported_metrics.value);

        if(effective_metrics.value == 0)
        {
            LOG_DEBUG("No enabled PMC metrics for device [{}]", device_index);
            return;
        }

        auto& device_data = detail::get_perfetto_data()[device_index];
        auto& tracks      = device_data.tracks;

        for(const auto& sample : samples)
        {
            const auto ts = sample.timestamp;

            if(!thread_info->is_valid_time(ts))
            {
                LOG_WARNING("Invalid timestamp {} for PMC sample", ts);
                continue;
            }

            process_basic_metrics(device_index, ts, sample.metrics, effective_metrics,
                                  tracks);
            process_xcp_activity(device_index, ts, sample.metrics, effective_metrics,
                                 tracks);
            process_xgmi_metrics(device_index, ts, sample.metrics, effective_metrics,
                                 device_data.xgmi_tracks);
            process_pcie_metrics(device_index, ts, sample.metrics, effective_metrics,
                                 device_data.pcie_tracks);
        }
    }

private:
    static void process_basic_metrics(
        size_t device_index, uint64_t ts, const metrics& metric_values,
        const enabled_metrics&                                   effective_metrics,
        std::unordered_map<uint32_t, detail::track_description>& tracks)
    {
        auto gfx_it = tracks.find(detail::GFX_BUSY_VALUE);
        if(effective_metrics.bits.gfx_activity && gfx_it != tracks.end() &&
           !gfx_it->second.track_indexes.empty())
        {
            TRACE_COUNTER(
                "device_busy_gfx",
                counter_track::at(device_index, gfx_it->second.track_indexes[0]), ts,
                static_cast<double>(metric_values.gfx_activity));
        }

        auto umc_it = tracks.find(detail::UMC_BUSY_VALUE);
        if(effective_metrics.bits.umc_activity && umc_it != tracks.end() &&
           !umc_it->second.track_indexes.empty())
        {
            TRACE_COUNTER(
                "device_busy_umc",
                counter_track::at(device_index, umc_it->second.track_indexes[0]), ts,
                static_cast<double>(metric_values.umc_activity));
        }

        auto mm_it = tracks.find(detail::MM_BUSY_VALUE);
        if(effective_metrics.bits.mm_activity && mm_it != tracks.end() &&
           !mm_it->second.track_indexes.empty())
        {
            TRACE_COUNTER("device_busy_mm",
                          counter_track::at(device_index, mm_it->second.track_indexes[0]),
                          ts, static_cast<double>(metric_values.mm_activity));
        }

        auto temp_it = tracks.find(detail::TEMPERATURE_VALUE);
        if((effective_metrics.bits.edge_temperature ||
            effective_metrics.bits.hotspot_temperature) &&
           temp_it != tracks.end() && !temp_it->second.track_indexes.empty())
        {
            const double temp = effective_metrics.bits.hotspot_temperature
                                    ? metric_values.hotspot_temperature
                                    : metric_values.edge_temperature;
            TRACE_COUNTER(
                "device_temp",
                counter_track::at(device_index, temp_it->second.track_indexes[0]), ts,
                temp);
        }

        auto power_it = tracks.find(detail::CURRENT_POWER_VALUE);
        if((effective_metrics.bits.average_socket_power ||
            effective_metrics.bits.current_socket_power) &&
           power_it != tracks.end() && !power_it->second.track_indexes.empty())
        {
            const double power = effective_metrics.bits.average_socket_power
                                     ? metric_values.average_socket_power
                                     : metric_values.current_socket_power;
            TRACE_COUNTER(
                "device_power",
                counter_track::at(device_index, power_it->second.track_indexes[0]), ts,
                power);
        }

        auto memory_it = tracks.find(detail::MEMORY_USAGE_VALUE);
        if(effective_metrics.bits.memory_usage && memory_it != tracks.end() &&
           !memory_it->second.track_indexes.empty())
        {
            const double usage =
                metric_values.memory_usage / static_cast<double>(tim::units::megabyte);
            TRACE_COUNTER(
                "device_memory_usage",
                counter_track::at(device_index, memory_it->second.track_indexes[0]), ts,
                usage);
        }

        auto sdma_it = tracks.find(detail::SDMA_USAGE_VALUE);
        if(effective_metrics.bits.sdma_usage && sdma_it != tracks.end() &&
           !sdma_it->second.track_indexes.empty())
        {
            TRACE_COUNTER(
                "device_sdma_usage",
                counter_track::at(device_index, sdma_it->second.track_indexes[0]), ts,
                static_cast<double>(metric_values.sdma_usage));
        }
    }

    static void process_xcp_activity(
        size_t device_index, uint64_t ts,
        const pmc::collectors::gpu::metrics&                     metric_values,
        const pmc::collectors::gpu::enabled_metrics&             effective_metrics,
        std::unordered_map<uint32_t, detail::track_description>& tracks)
    {
        // Per-XCP VCN busy metrics (MI300)
        auto vcn_busy_it = tracks.find(detail::VCN_BUSY_VALUE);
        if(effective_metrics.bits.vcn_busy && vcn_busy_it != tracks.end() &&
           !vcn_busy_it->second.track_indexes.empty())
        {
            size_t engine_id = 0;
            for(const auto& xcp_stats : metric_values.xcp_stats)
            {
                for(const auto& vcn_val : xcp_stats.vcn_busy)
                {
                    if(vcn_val != std::numeric_limits<uint16_t>::max() &&
                       engine_id < vcn_busy_it->second.track_indexes.size())
                    {
                        TRACE_COUNTER("device_vcn_activity",
                                      counter_track::at(
                                          device_index,
                                          vcn_busy_it->second.track_indexes[engine_id++]),
                                      ts, vcn_val);
                    }
                }
            }
        }

        // Device-level VCN activity (Radeon)
        auto vcn_it = tracks.find(detail::VCN_ACTIVITY_VALUE);
        if(effective_metrics.bits.vcn_activity && vcn_it != tracks.end() &&
           !vcn_it->second.track_indexes.empty())
        {
            size_t engine_id = 0;
            for(const auto& vcn_val : metric_values.vcn_activity)
            {
                if(vcn_val != std::numeric_limits<uint16_t>::max() &&
                   engine_id < vcn_it->second.track_indexes.size())
                {
                    TRACE_COUNTER(
                        "device_vcn_activity",
                        counter_track::at(device_index,
                                          vcn_it->second.track_indexes[engine_id++]),
                        ts, vcn_val);
                }
            }
        }

        // Per-XCP JPEG busy metrics (MI300)
        auto jpeg_busy_it = tracks.find(detail::JPEG_BUSY_VALUE);
        if(effective_metrics.bits.jpeg_busy && jpeg_busy_it != tracks.end() &&
           !jpeg_busy_it->second.track_indexes.empty())
        {
            size_t engine_id = 0;
            for(const auto& xcp_stats : metric_values.xcp_stats)
            {
                for(const auto& jpeg_val : xcp_stats.jpeg_busy)
                {
                    if(jpeg_val != std::numeric_limits<uint16_t>::max() &&
                       engine_id < jpeg_busy_it->second.track_indexes.size())
                    {
                        TRACE_COUNTER(
                            "device_jpeg_activity",
                            counter_track::at(
                                device_index,
                                jpeg_busy_it->second.track_indexes[engine_id++]),
                            ts, jpeg_val);
                    }
                }
            }
        }

        // Device-level JPEG activity (Radeon)
        auto jpeg_it = tracks.find(detail::JPEG_ACTIVITY_VALUE);
        if(effective_metrics.bits.jpeg_activity && jpeg_it != tracks.end() &&
           !jpeg_it->second.track_indexes.empty())
        {
            size_t engine_id = 0;
            for(const auto& jpeg_val : metric_values.jpeg_activity)
            {
                if(jpeg_val != std::numeric_limits<uint16_t>::max() &&
                   engine_id < jpeg_it->second.track_indexes.size())
                {
                    TRACE_COUNTER(
                        "device_jpeg_activity",
                        counter_track::at(device_index,
                                          jpeg_it->second.track_indexes[engine_id++]),
                        ts, jpeg_val);
                }
            }
        }
    }

    static void process_xgmi_metrics(size_t device_index, uint64_t ts,
                                     const metrics&                metric_values,
                                     const enabled_metrics&        effective_metrics,
                                     const detail::xgmi_track_set& xgmi_tracks)
    {
        if(!effective_metrics.bits.xgmi)
        {
            return;
        }

        if(!xgmi_tracks.link_width.empty() && metric_values.xgmi.link.width != 0)
        {
            TRACE_COUNTER("device_xgmi_link_width",
                          counter_track::at(device_index, xgmi_tracks.link_width[0]), ts,
                          static_cast<double>(metric_values.xgmi.link.width));
        }

        if(!xgmi_tracks.link_speed.empty() && metric_values.xgmi.link.speed != 0)
        {
            TRACE_COUNTER("device_xgmi_link_speed",
                          counter_track::at(device_index, xgmi_tracks.link_speed[0]), ts,
                          static_cast<double>(metric_values.xgmi.link.speed));
        }

        for(size_t link = 0;
            link < AMDSMI_MAX_NUM_XGMI_LINKS && link < xgmi_tracks.read_data.size();
            ++link)
        {
            if(metric_values.xgmi.data_acc.read[link] != 0)
            {
                TRACE_COUNTER(
                    "device_xgmi_read_data",
                    counter_track::at(device_index, xgmi_tracks.read_data[link]), ts,
                    static_cast<double>(metric_values.xgmi.data_acc.read[link]));
            }
        }

        for(size_t link = 0;
            link < AMDSMI_MAX_NUM_XGMI_LINKS && link < xgmi_tracks.write_data.size();
            ++link)
        {
            if(metric_values.xgmi.data_acc.write[link] != 0)
            {
                TRACE_COUNTER(
                    "device_xgmi_write_data",
                    counter_track::at(device_index, xgmi_tracks.write_data[link]), ts,
                    static_cast<double>(metric_values.xgmi.data_acc.write[link]));
            }
        }
    }

    static void process_pcie_metrics(size_t device_index, uint64_t ts,
                                     const metrics&                metric_values,
                                     const enabled_metrics&        effective_metrics,
                                     const detail::pcie_track_set& pcie_tracks)
    {
        if(!effective_metrics.bits.pcie)
        {
            return;
        }

        if(!pcie_tracks.link_width.empty() && metric_values.pcie.link.width != 0)
        {
            TRACE_COUNTER("device_pcie_link_width",
                          counter_track::at(device_index, pcie_tracks.link_width[0]), ts,
                          static_cast<double>(metric_values.pcie.link.width));
        }

        if(!pcie_tracks.link_speed.empty() && metric_values.pcie.link.speed != 0)
        {
            TRACE_COUNTER("device_pcie_link_speed",
                          counter_track::at(device_index, pcie_tracks.link_speed[0]), ts,
                          static_cast<double>(metric_values.pcie.link.speed));
        }

        if(!pcie_tracks.bandwidth_acc.empty() && metric_values.pcie.bandwidth.acc != 0)
        {
            TRACE_COUNTER("device_pcie_bandwidth_acc",
                          counter_track::at(device_index, pcie_tracks.bandwidth_acc[0]),
                          ts, static_cast<double>(metric_values.pcie.bandwidth.acc));
        }

        if(!pcie_tracks.bandwidth_inst.empty() && metric_values.pcie.bandwidth.inst != 0)
        {
            TRACE_COUNTER("device_pcie_bandwidth_inst",
                          counter_track::at(device_index, pcie_tracks.bandwidth_inst[0]),
                          ts, static_cast<double>(metric_values.pcie.bandwidth.inst));
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu

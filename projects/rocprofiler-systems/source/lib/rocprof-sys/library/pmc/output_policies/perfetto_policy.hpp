// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/perfetto.hpp"
#include "library/pmc/gpu/metric_descriptors.hpp"
#include "library/pmc/gpu/types.hpp"
#include "library/thread_info.hpp"

#include <mutex>
#include <timemory/units.hpp>

#include <vector>

#include <spdlog/fmt/ranges.h>

namespace rocprofsys
{
namespace pmc
{
namespace output_policies
{

#if ROCPROFSYS_USE_ROCM > 0

// Import common GPU collector types
using pmc::gpu::enabled_metrics;
using pmc::gpu::metrics;

// Use centralized metric masks from metric_descriptors.hpp
namespace metric_masks = ::rocprofsys::pmc::gpu::metric_masks;

/**
 * @brief Output policy for writing PMC samples directly to Perfetto traces.
 *
 * This policy handles real-time serialization of AMD SMI metric samples into
 * Perfetto trace format, creating counter tracks for each metric type.
 * Supports both device-level metrics (Radeon) and per-XCP metrics.
 *
 * @see cache_policy for writing to trace cache instead
 */
struct perfetto_policy
{
    // Types for track management
    struct track_description
    {
        const char*         track_name;
        const char*         units;
        std::vector<size_t> track_indexes;
    };

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
        size_t            timestamp;
        pmc::gpu::metrics metrics;
    };

    // Static state accessors - using inline static to ensure single instance across TUs
    static std::unordered_map<uint32_t, track_description>& get_perfetto_tracks()
    {
        static std::unordered_map<uint32_t, track_description> tracks{
            { metric_masks::gfx_activity, { "GFX Busy", "%", {} } },
            { metric_masks::umc_activity, { "UMC Busy", "%", {} } },
            { metric_masks::mm_activity, { "MM Busy", "%", {} } },
            { metric_masks::temperature, { "Temperature", "deg C", {} } },
            { metric_masks::power, { "Current Power", "watts", {} } },
            { metric_masks::memory_usage, { "Memory Usage", "megabytes", {} } },
            // Device-level VCN/JPEG activity
            { metric_masks::vcn_activity, { "VCN Activity", "%", {} } },
            { metric_masks::jpeg_activity, { "JPEG Activity", "%", {} } },
            // Per-XCP VCN/JPEG busy
            { metric_masks::xcp_vcn_activity, { "VCN Busy", "%", {} } },
            { metric_masks::xcp_jpeg_activity, { "JPEG Busy", "%", {} } },
            { metric_masks::xgmi_mask, { "XGMI", "", {} } },
            { metric_masks::pcie_mask, { "PCIe", "", {} } },
        };
        return tracks;
    }

    static std::map<size_t, xgmi_track_set>& get_xgmi_tracks()
    {
        static std::map<size_t, xgmi_track_set> tracks;
        return tracks;
    }

    static std::map<size_t, pcie_track_set>& get_pcie_tracks()
    {
        static std::map<size_t, pcie_track_set> tracks;
        return tracks;
    }

    static std::map<size_t, std::unique_ptr<std::vector<perfetto_amd_smi_sample>>>&
    get_perfetto_bundle()
    {
        static std::map<size_t, std::unique_ptr<std::vector<perfetto_amd_smi_sample>>>
            bundle;
        return bundle;
    }

    static std::map<size_t, pmc::gpu::enabled_metrics>& get_device_registry()
    {
        static std::map<size_t, pmc::gpu::enabled_metrics> registry;
        return registry;
    }

    using counter_track = perfetto_counter_track<metrics>;

    /**
     * @brief Initialize Perfetto storage for the given processor devices.
     *
     * Allocates storage buffers for Perfetto samples for each GPU device.
     *
     * @tparam DeviceEntryVector Container of device_entry structs (device + supported
     * metrics)
     * @param entries Vector of device entries to initialize storage for
     */
    template <typename DeviceEntryVector>
    static void init_storage(const DeviceEntryVector& entries)
    {
        for(const auto& entry : entries)
        {
            auto device_index = entry.device->get_index();
            get_perfetto_bundle().insert(
                { device_index,
                  std::make_unique<std::vector<perfetto_amd_smi_sample>>() });
            get_device_registry().insert({ device_index, entry.supported_metrics });
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

        auto& tracks = get_perfetto_tracks();

        for(auto& [num, description] : tracks)
        {
            auto enabled_metric = num & enabled_metric_config.value();
            if(enabled_metric == 0)
            {
                continue;
            }

            const auto process_xcp_array = [&](track_description& desc, size_t array_size,
                                               size_t xcp_id) {
                for(std::size_t i = 0; i < array_size; ++i)
                {
                    const auto track_id = counter_track::emplace(
                        device_index, addendum_blk(i, desc.track_name, xcp_id),
                        desc.units);
                    desc.track_indexes.emplace_back(track_id);
                }
            };

            // Check if this is a VCN or JPEG track (device-level or per-XCP)
            const bool is_vcn_track  = (num == metric_masks::vcn_activity ||
                                       num == metric_masks::xcp_vcn_activity);
            const bool is_jpeg_track = (num == metric_masks::jpeg_activity ||
                                        num == metric_masks::xcp_jpeg_activity);

            if(is_vcn_track || is_jpeg_track)
            {
                for(std::size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
                {
                    process_xcp_array(description,
                                      is_vcn_track ? AMDSMI_MAX_NUM_VCN
                                                   : ROCPROFSYS_MAX_NUM_JPEG_ENGINES,
                                      xcp);
                }
            }
            else
            {
                description.track_indexes.emplace_back(counter_track::emplace(
                    device_index, addendum(description.track_name), description.units));
            }
        }

        if(enabled_metric_config.xgmi())
        {
            auto& xgmi_tracks = get_xgmi_tracks()[device_index];

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

        if(enabled_metric_config.pcie())
        {
            auto& pcie_tracks = get_pcie_tracks()[device_index];

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
                             unsigned long timestamp)
    {
        get_perfetto_bundle()[device_index]->emplace_back(
            perfetto_amd_smi_sample{ timestamp, metric_values });
    }

    /**
     * @brief Post-process buffered samples and write to Perfetto trace.
     *
     * Serializes all buffered PMC samples to Perfetto counter tracks.
     * This is called at the end of profiling to flush all samples.
     * Uses the internally tracked device registry populated during init_storage().
     *
     * @param enabled_metrics Metrics that were enabled during collection
     */
    static void post_process(pmc::gpu::enabled_metrics enabled_metrics)
    {
        const auto& registry = get_device_registry();
        LOG_DEBUG("Post-processing {} devices", registry.size());
        for(const auto& [device_index, supported_metrics] : registry)
        {
            post_process_device(device_index, enabled_metrics, supported_metrics);
        }
    }

private:
    static void post_process_device(size_t                    device_index,
                                    pmc::gpu::enabled_metrics enabled_metrics,
                                    pmc::gpu::enabled_metrics supported_metrics)
    {
        auto& samples = *get_perfetto_bundle()[device_index];

        LOG_DEBUG("Post-processing {} PMC samples for device [{}]", samples.size(),
                  device_index);

        const auto& thread_info = thread_info::get(0, InternalTID);
        if(!thread_info)
        {
            return;
        }

        pmc::gpu::enabled_metrics effective_metrics(enabled_metrics.value() &
                                                    supported_metrics.value());

        if(effective_metrics.none())
        {
            LOG_DEBUG("No enabled PMC metrics for device [{}]", device_index);
            return;
        }

        auto& tracks = get_perfetto_tracks();

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
                                 enabled_metrics, supported_metrics, tracks);
            process_xgmi_metrics(device_index, ts, sample.metrics, effective_metrics);
            process_pcie_metrics(device_index, ts, sample.metrics, effective_metrics);
        }
    }

private:
    static void process_basic_metrics(
        size_t device_index, size_t ts, const metrics& metric_values,
        const enabled_metrics&                           effective_metrics,
        std::unordered_map<uint32_t, track_description>& tracks)
    {
        if(effective_metrics.gfx_activity() &&
           !tracks.at(metric_masks::gfx_activity).track_indexes.empty())
        {
            TRACE_COUNTER(
                "device_busy_gfx",
                counter_track::at(device_index,
                                  tracks.at(metric_masks::gfx_activity).track_indexes[0]),
                ts, static_cast<double>(metric_values.gfx_activity));
        }

        if(effective_metrics.umc_activity() &&
           !tracks.at(metric_masks::umc_activity).track_indexes.empty())
        {
            TRACE_COUNTER(
                "device_busy_umc",
                counter_track::at(device_index,
                                  tracks.at(metric_masks::umc_activity).track_indexes[0]),
                ts, static_cast<double>(metric_values.umc_activity));
        }

        if(effective_metrics.mm_activity() &&
           !tracks.at(metric_masks::mm_activity).track_indexes.empty())
        {
            TRACE_COUNTER(
                "device_busy_mm",
                counter_track::at(device_index,
                                  tracks.at(metric_masks::mm_activity).track_indexes[0]),
                ts, static_cast<double>(metric_values.mm_activity));
        }

        if((effective_metrics.edge_temperature() ||
            effective_metrics.hotspot_temperature()) &&
           !tracks.at(metric_masks::temperature).track_indexes.empty())
        {
            const double temp = effective_metrics.hotspot_temperature()
                                    ? metric_values.hotspot_temperature
                                    : metric_values.edge_temperature;
            TRACE_COUNTER(
                "device_temp",
                counter_track::at(device_index,
                                  tracks.at(metric_masks::temperature).track_indexes[0]),
                ts, temp);
        }

        if((effective_metrics.average_socket_power() ||
            effective_metrics.current_socket_power()) &&
           !tracks.at(metric_masks::power).track_indexes.empty())
        {
            const double power = effective_metrics.average_socket_power()
                                     ? metric_values.average_socket_power
                                     : metric_values.current_socket_power;
            TRACE_COUNTER(
                "device_power",
                counter_track::at(device_index,
                                  tracks.at(metric_masks::power).track_indexes[0]),
                ts, power);
        }

        if(effective_metrics.memory_usage() &&
           !tracks.at(metric_masks::memory_usage).track_indexes.empty())
        {
            const double usage =
                metric_values.memory_usage / static_cast<double>(tim::units::megabyte);
            TRACE_COUNTER(
                "device_memory_usage",
                counter_track::at(device_index,
                                  tracks.at(metric_masks::memory_usage).track_indexes[0]),
                ts, usage);
        }
    }

    static void process_xcp_activity(
        size_t device_index, size_t ts, const pmc::gpu::metrics& metric_values,
        const pmc::gpu::enabled_metrics&                 effective_metrics,
        const pmc::gpu::enabled_metrics&                 enabled_metric_config,
        const pmc::gpu::enabled_metrics&                 supported_metric_config,
        std::unordered_map<uint32_t, track_description>& tracks)
    {
        if(effective_metrics.xcp_vcn_activity() &&
           !tracks.at(metric_masks::xcp_vcn_activity).track_indexes.empty())
        {
            size_t engine_id = 0;
            for(const auto& xcp_stats : metric_values.xcp_stats)
            {
                for(const auto& vcn_val : xcp_stats.vcn_busy)
                {
                    if(vcn_val != std::numeric_limits<uint16_t>::max() &&
                       engine_id <
                           tracks.at(metric_masks::xcp_vcn_activity).track_indexes.size())
                    {
                        TRACE_COUNTER(
                            "device_vcn_busy",
                            counter_track::at(device_index,
                                              tracks.at(metric_masks::xcp_vcn_activity)
                                                  .track_indexes[engine_id++]),
                            ts, vcn_val);
                    }
                }
            }
        }

        if(effective_metrics.vcn_activity() &&
           !tracks.at(metric_masks::vcn_activity).track_indexes.empty())
        {
            size_t engine_id = 0;
            for(const auto& vcn_val : metric_values.vcn_activity)
            {
                if(vcn_val != std::numeric_limits<uint16_t>::max() &&
                   engine_id < tracks.at(metric_masks::vcn_activity).track_indexes.size())
                {
                    TRACE_COUNTER("device_vcn_activity",
                                  counter_track::at(device_index,
                                                    tracks.at(metric_masks::vcn_activity)
                                                        .track_indexes[engine_id++]),
                                  ts, vcn_val);
                }
            }
        }

        LOG_DEBUG("JPEG activity: {}, enabled: {}, supported: {}",
                  effective_metrics.jpeg_activity(),
                  enabled_metric_config.jpeg_activity(),
                  supported_metric_config.jpeg_activity());

        if(effective_metrics.xcp_jpeg_activity() &&
           !tracks.at(metric_masks::xcp_jpeg_activity).track_indexes.empty())
        {
            size_t engine_id = 0;
            for(const auto& xcp_stats : metric_values.xcp_stats)
            {
                for(const auto& jpeg_val : xcp_stats.jpeg_busy)
                {
                    if(jpeg_val != std::numeric_limits<uint16_t>::max() &&
                       engine_id < tracks.at(metric_masks::xcp_jpeg_activity)
                                       .track_indexes.size())
                    {
                        TRACE_COUNTER(
                            "device_jpeg_busy",
                            counter_track::at(device_index,
                                              tracks.at(metric_masks::xcp_jpeg_activity)
                                                  .track_indexes[engine_id++]),
                            ts, jpeg_val);
                    }
                }
            }
        }

        if(effective_metrics.jpeg_activity() &&
           !tracks.at(metric_masks::jpeg_activity).track_indexes.empty())
        {
            size_t engine_id = 0;
            for(const auto& jpeg_val : metric_values.jpeg_activity)
            {
                if(jpeg_val != std::numeric_limits<uint16_t>::max() &&
                   engine_id <
                       tracks.at(metric_masks::jpeg_activity).track_indexes.size())
                {
                    TRACE_COUNTER("device_jpeg_activity",
                                  counter_track::at(device_index,
                                                    tracks.at(metric_masks::jpeg_activity)
                                                        .track_indexes[engine_id++]),
                                  ts, jpeg_val);
                }
            }
        }
    }

    static void process_xgmi_metrics(size_t device_index, size_t ts,
                                     const metrics&         metric_values,
                                     const enabled_metrics& effective_metrics)
    {
        if(!effective_metrics.xgmi())
        {
            return;
        }

        auto xgmi_it = get_xgmi_tracks().find(device_index);
        if(xgmi_it == get_xgmi_tracks().end())
        {
            return;
        }

        const auto& xgmi_tracks = xgmi_it->second;

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

    static void process_pcie_metrics(size_t device_index, size_t ts,
                                     const metrics&         metric_values,
                                     const enabled_metrics& effective_metrics)
    {
        if(!effective_metrics.pcie())
        {
            return;
        }

        auto pcie_it = get_pcie_tracks().find(device_index);
        if(pcie_it == get_pcie_tracks().end())
        {
            return;
        }

        const auto& pcie_tracks = pcie_it->second;

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

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace output_policies
}  // namespace pmc
}  // namespace rocprofsys

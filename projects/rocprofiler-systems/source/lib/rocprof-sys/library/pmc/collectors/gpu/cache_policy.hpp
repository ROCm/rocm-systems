// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/gpu/sample.hpp"
#include "library/pmc/collectors/gpu/types.hpp"

#include <timemory/units.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace rocprofsys::pmc::collectors::gpu
{

/**
 * @brief Output policy for writing GPU PMC samples to the trace cache.
 *
 * This policy handles serialization of AMD SMI GPU metric samples into the
 * rocprofiler-systems trace cache for later analysis and visualization.
 * It manages category metadata initialization and per-device PMC metadata
 * registration.
 *
 * @see perfetto_policy for direct Perfetto trace output
 */
struct cache_policy
{
    /**
     * @brief Initialize trace cache category metadata for AMD SMI metrics.
     *
     * Registers category names in the trace cache metadata registry.
     * This is called once during initialization.
     */
    static void initialize_category_metadata()
    {
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::amd_smi>::value);
    }

    static void initialize_tracks_metadata()
    {
        const auto thread_id = std::nullopt;

        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_gfx_busy>(),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_umc_busy>(),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_mm_busy>(),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_power>(), thread_id,
              "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_temp>(), thread_id,
              "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_memory_usage>(),
              thread_id, "{}" });

        auto add_vcn_track = [&](std::optional<int> xcp_idx) {
            for(int clk = 0; clk < AMDSMI_MAX_NUM_VCN; ++clk)
            {
                auto name =
                    trace_cache::info::format_track_name<category::amd_smi_vcn_activity>(
                        xcp_idx, clk);
                trace_cache::get_metadata_registry().add_track(
                    { name.c_str(), thread_id, "{}" });
            }
        };

        auto add_jpeg_track = [&](std::optional<int> xcp_idx) {
            for(size_t clk = 0; clk < ROCPROFSYS_AMDSMI_JPEG_ENGINE_COUNT; ++clk)
            {
                auto name =
                    trace_cache::info::format_track_name<category::amd_smi_jpeg_activity>(
                        xcp_idx, clk);
                trace_cache::get_metadata_registry().add_track(
                    { name.c_str(), thread_id, "{}" });
            }
        };

        for(int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            add_vcn_track(xcp);
            add_jpeg_track(xcp);
        }

        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_xgmi_link_width>(),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_xgmi_link_speed>(),
              thread_id, "{}" });

        for(int vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
        {
            auto vcn_name =
                trace_cache::info::format_track_name<category::amd_smi_vcn_activity>(
                    std::nullopt, vcn);
            trace_cache::get_metadata_registry().add_track(
                { vcn_name.c_str(), thread_id, "{}" });
        }

        for(int link = 0; link < AMDSMI_MAX_NUM_XGMI_LINKS; ++link)
        {
            auto read_name =
                trace_cache::info::format_track_name<category::amd_smi_xgmi_read_data>(
                    std::nullopt, link);
            trace_cache::get_metadata_registry().add_track(
                { read_name.c_str(), thread_id, "{}" });

            auto write_name =
                trace_cache::info::format_track_name<category::amd_smi_xgmi_write_data>(
                    std::nullopt, link);
            trace_cache::get_metadata_registry().add_track(
                { write_name.c_str(), thread_id, "{}" });
        }

        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_sdma_usage>(),
              thread_id, "{}" });

        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_pcie_link_width>(),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<category::amd_smi_pcie_link_speed>(),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<
                  category::amd_smi_pcie_bandwidth_acc>(),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::format_track_name<
                  category::amd_smi_pcie_bandwidth_inst>(),
              thread_id, "{}" });
    }

    /**
     * @brief Initialize per-device PMC metadata for AMD SMI metrics.
     *
     * Registers PMC metadata (name, description, units, etc.) for each metric type
     * that can be collected from the specified GPU device.
     *
     * @param gpu_id GPU device identifier for which to register metadata
     */
    static void initialize_pmc_metadata(size_t gpu_id)
    {
        // Metadata field constants for PMC info registration
        constexpr size_t      EVENT_CODE       = 0;
        constexpr size_t      INSTANCE_ID      = 0;
        constexpr const char* LONG_DESCRIPTION = "";
        constexpr const char* COMPONENT        = "";
        constexpr const char* BLOCK            = "";
        constexpr const char* EXPRESSION       = "";
        constexpr const char* CELSIUS_DEGREES  = "\u00B0C";
        constexpr const char* TARGET_ARCH      = "GPU";

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_gfx_busy>::value, "GFX Busy",
              trait::name<category::amd_smi_gfx_busy>::description, LONG_DESCRIPTION,
              COMPONENT, trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_umc_busy>::value, "UMC Busy",
              trait::name<category::amd_smi_umc_busy>::description, LONG_DESCRIPTION,
              COMPONENT, trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_mm_busy>::value, "MM Busy",
              trait::name<category::amd_smi_mm_busy>::description, LONG_DESCRIPTION,
              COMPONENT, trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_temp>::value, "Temp",
              trait::name<category::amd_smi_temp>::description, LONG_DESCRIPTION,
              COMPONENT, CELSIUS_DEGREES, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
              EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_power>::value, "Pow",
              trait::name<category::amd_smi_power>::description, LONG_DESCRIPTION,
              COMPONENT, "W", rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0,
              0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_memory_usage>::value, "MemUsg",
              trait::name<category::amd_smi_memory_usage>::description, LONG_DESCRIPTION,
              COMPONENT, tim::units::mem_repr(tim::units::megabyte),
              rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });

        for(int vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
        {
            auto vcn_name =
                trace_cache::info::format_track_name<category::amd_smi_vcn_activity>(vcn);

            trace_cache::get_metadata_registry().add_pmc_info(
                { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                  vcn_name.c_str(), vcn_name.c_str(),
                  "VCN (Video Decode) Engine Activity", LONG_DESCRIPTION, COMPONENT,
                  trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
                  EXPRESSION, 0, 0 });
        }

        for(int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            for(int vcn = 0; vcn < AMDSMI_MAX_NUM_VCN; ++vcn)
            {
                auto vcn_name =
                    trace_cache::info::format_track_name<category::amd_smi_vcn_activity>(
                        xcp, vcn);

                trace_cache::get_metadata_registry().add_pmc_info(
                    { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                      vcn_name.c_str(), vcn_name.c_str(),
                      "VCN (Video Decode) Engine Activity", LONG_DESCRIPTION, COMPONENT,
                      trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
                      EXPRESSION, 0, 0 });
            }
        }

        for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            for(size_t jpeg = 0; jpeg < ROCPROFSYS_AMDSMI_JPEG_ENGINE_COUNT; ++jpeg)
            {
                auto jpeg_name =
                    trace_cache::info::format_track_name<category::amd_smi_jpeg_activity>(
                        xcp, jpeg);
                trace_cache::get_metadata_registry().add_pmc_info(
                    { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                      jpeg_name.c_str(), jpeg_name.c_str(),
                      "JPEG (Image Decode) Engine Activity", LONG_DESCRIPTION, COMPONENT,
                      trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
                      EXPRESSION, 0, 0 });
            }
        }

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_sdma_usage>::value, "SDMA Usage",
              trait::name<category::amd_smi_sdma_usage>::description, LONG_DESCRIPTION,
              COMPONENT, trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_xgmi_link_width>::value, "XGMI Width",
              trait::name<category::amd_smi_xgmi_link_width>::description,
              LONG_DESCRIPTION, COMPONENT, "lanes", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_xgmi_link_speed>::value, "XGMI Speed",
              trait::name<category::amd_smi_xgmi_link_speed>::description,
              LONG_DESCRIPTION, COMPONENT, "Mbps", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_xgmi_read_data>::value, "XGMI Read",
              trait::name<category::amd_smi_xgmi_read_data>::description,
              LONG_DESCRIPTION, COMPONENT, "KB", rocprofsys::trace_cache::ABSOLUTE, BLOCK,
              EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_xgmi_write_data>::value, "XGMI Write",
              trait::name<category::amd_smi_xgmi_write_data>::description,
              LONG_DESCRIPTION, COMPONENT, "KB", rocprofsys::trace_cache::ABSOLUTE, BLOCK,
              EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_pcie_link_width>::value, "PCIe Width",
              trait::name<category::amd_smi_pcie_link_width>::description,
              LONG_DESCRIPTION, COMPONENT, "lanes", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_pcie_link_speed>::value, "PCIe Speed",
              trait::name<category::amd_smi_pcie_link_speed>::description,
              LONG_DESCRIPTION, COMPONENT, "MT/s", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_pcie_bandwidth_acc>::value, "PCIe BW Acc",
              trait::name<category::amd_smi_pcie_bandwidth_acc>::description,
              LONG_DESCRIPTION, COMPONENT, "bytes", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              trait::name<category::amd_smi_pcie_bandwidth_inst>::value, "PCIe BW Inst",
              trait::name<category::amd_smi_pcie_bandwidth_inst>::description,
              LONG_DESCRIPTION, COMPONENT, "bytes/s", rocprofsys::trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0 });
    }

    /**
     * @brief Store a PMC sample to the trace cache.
     *
     * @param device_id GPU device identifier
     * @param device_name Device name (unused for GPU, kept for API consistency)
     * @param enabled_metrics Metrics requested by user configuration
     * @param supported_metrics Metrics supported by this device
     * @param metrics Collected metric values
     * @param timestamp Sample timestamp in nanoseconds
     */
    static void store_sample(size_t device_id, const std::string& /*device_name*/,
                             const enabled_metrics& enabled_metrics_cfg,
                             const enabled_metrics& supported_metrics,
                             const metrics& metric_values, uint64_t timestamp)
    {
        enabled_metrics _enabled_metrics;
        _enabled_metrics.value = enabled_metrics_cfg.value & supported_metrics.value;

        trace_cache::get_buffer_storage().store(trace_cache::gpu_pmc_sample{
            _enabled_metrics, static_cast<uint32_t>(device_id), timestamp,
            metric_values });
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu

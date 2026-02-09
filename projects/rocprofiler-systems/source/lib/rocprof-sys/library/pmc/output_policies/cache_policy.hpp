// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/gpu/types.hpp"

#include <timemory/units.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>

namespace rocprofsys
{
namespace pmc
{
namespace output_policies
{

#if ROCPROFSYS_USE_ROCM > 0

/**
 * @brief Output policy for writing PMC samples to the trace cache.
 *
 * This policy handles serialization of AMD SMI metric samples into the
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
        if(!get_use_cache_output())
        {
            return;
        }
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::amd_smi>::value);
    }

    static void initialize_smi_tracks_metadata()
    {
        if(!get_use_cache_output())
        {
            return;
        }

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
            for(int clk = 0; clk < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++clk)
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
    static void initialize_smi_pmc_metadata(size_t gpu_id)
    {
        if(!get_use_cache_output())
        {
            return;
        }

        // Metadata field constants for PMC info registration
        // Empty strings indicate optional fields not used for AMD SMI metrics
        constexpr size_t      EVENT_CODE       = 0;   // No specific event code
        constexpr size_t      INSTANCE_ID      = 0;   // Default instance
        constexpr const char* LONG_DESCRIPTION = "";  // Using short description only
        constexpr const char* COMPONENT        = "";  // No specific component
        constexpr const char* BLOCK            = "";  // No specific block
        constexpr const char* EXPRESSION       = "";  // No derived expression
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

        for(int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            for(int jpeg = 0; jpeg < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++jpeg)
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
     * Writes the collected metric values to the trace cache buffer for later
     * post-processing and analysis. Only metrics that are both supported and
     * enabled will be stored.
     *
     * @param device_id GPU device identifier
     * @param supported_metrics Metrics supported by this device
     * @param enabled_metrics Metrics requested by user configuration
     * @param metrics Collected metric values
     * @param timestamp Sample timestamp in nanoseconds
     */
    static void store_sample(size_t                           device_id,
                             const pmc::gpu::enabled_metrics& supported_metrics,
                             const pmc::gpu::enabled_metrics& enabled_metrics,
                             const pmc::gpu::metrics& metrics, unsigned long timestamp)
    {
        if(!get_use_cache_output())
        {
            return;
        }
        pmc::gpu::enabled_metrics _enabled_metrics = { .value = enabled_metrics.value &
                                                                supported_metrics.value };

        trace_cache::get_buffer_storage().store(trace_cache::gpu_pmc_sample{
            _enabled_metrics, static_cast<uint32_t>(device_id), timestamp, metrics });
    }
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace output_policies
}  // namespace pmc
}  // namespace rocprofsys

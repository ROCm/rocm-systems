// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/collectors/gpu/sample.hpp"
#include "library/pmc/collectors/gpu/types.hpp"

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
            { .track_name =
                  trace_cache::info::format_track_name<category::amd_smi_gfx_busy>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name =
                  trace_cache::info::format_track_name<category::amd_smi_umc_busy>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name =
                  trace_cache::info::format_track_name<category::amd_smi_mm_busy>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name =
                  trace_cache::info::format_track_name<category::amd_smi_power>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name =
                  trace_cache::info::format_track_name<category::amd_smi_temp>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name =
                  trace_cache::info::format_track_name<category::amd_smi_memory_usage>(),
              .thread_id = thread_id,
              .extdata   = "{}" });

        auto const add_vcn_track = [&](std::optional<int> xcp_idx) {
            for(size_t clk = 0; clk < MAX_NUM_VCN; ++clk)
            {
                auto const name =
                    trace_cache::info::format_track_name<category::amd_smi_vcn_activity>(
                        xcp_idx, clk);
                trace_cache::get_metadata_registry().add_track(
                    { .track_name = name, .thread_id = thread_id, .extdata = "{}" });
            }
        };

        auto const add_jpeg_track = [&](std::optional<int> xcp_idx) {
            for(size_t clk = 0; clk < MAX_NUM_JPEG_V1; ++clk)
            {
                auto const name =
                    trace_cache::info::format_track_name<category::amd_smi_jpeg_activity>(
                        xcp_idx, clk);
                trace_cache::get_metadata_registry().add_track(
                    { .track_name = name, .thread_id = thread_id, .extdata = "{}" });
            }
        };

        for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
        {
            add_vcn_track(xcp);
            add_jpeg_track(xcp);
        }

        trace_cache::get_metadata_registry().add_track(
            { .track_name = trace_cache::info::format_track_name<
                  category::amd_smi_xgmi_link_width>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name = trace_cache::info::format_track_name<
                  category::amd_smi_xgmi_link_speed>(),
              .thread_id = thread_id,
              .extdata   = "{}" });

        for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
        {
            auto const vcn_name =
                trace_cache::info::format_track_name<category::amd_smi_vcn_activity>(
                    std::nullopt, vcn);
            trace_cache::get_metadata_registry().add_track(
                { .track_name = vcn_name, .thread_id = thread_id, .extdata = "{}" });
        }

        for(size_t jpeg = 0; jpeg < MAX_NUM_JPEG; ++jpeg)
        {
            auto const jpeg_name =
                trace_cache::info::format_track_name<category::amd_smi_jpeg_activity>(
                    std::nullopt, jpeg);
            trace_cache::get_metadata_registry().add_track(
                { .track_name = jpeg_name, .thread_id = thread_id, .extdata = "{}" });
        }

        for(size_t link = 0; link < MAX_NUM_XGMI_LINKS; ++link)
        {
            auto const read_name = trace_cache::info::format_link_track_name(
                trait::name<category::amd_smi_xgmi_read_data>::value, link);
            trace_cache::get_metadata_registry().add_track(
                { .track_name = read_name, .thread_id = thread_id, .extdata = "{}" });

            auto const write_name = trace_cache::info::format_link_track_name(
                trait::name<category::amd_smi_xgmi_write_data>::value, link);
            trace_cache::get_metadata_registry().add_track(
                { .track_name = write_name, .thread_id = thread_id, .extdata = "{}" });
        }

        trace_cache::get_metadata_registry().add_track(
            { .track_name =
                  trace_cache::info::format_track_name<category::amd_smi_sdma_usage>(),
              .thread_id = thread_id,
              .extdata   = "{}" });

        trace_cache::get_metadata_registry().add_track(
            { .track_name =
                  trace_cache::info::format_track_name<category::amd_smi_gfx_clock>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name =
                  trace_cache::info::format_track_name<category::amd_smi_mem_clock>(),
              .thread_id = thread_id,
              .extdata   = "{}" });

        trace_cache::get_metadata_registry().add_track(
            { .track_name = trace_cache::info::format_track_name<
                  category::amd_smi_pcie_link_width>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name = trace_cache::info::format_track_name<
                  category::amd_smi_pcie_link_speed>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name = trace_cache::info::format_track_name<
                  category::amd_smi_pcie_bandwidth_acc>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
        trace_cache::get_metadata_registry().add_track(
            { .track_name = trace_cache::info::format_track_name<
                  category::amd_smi_pcie_bandwidth_inst>(),
              .thread_id = thread_id,
              .extdata   = "{}" });
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
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_gfx_busy>::value,
              .symbol           = "GFX Busy",
              .description      = trait::name<category::amd_smi_gfx_busy>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = trace_cache::PERCENTAGE,
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_umc_busy>::value,
              .symbol           = "UMC Avg. Busy",
              .description      = trait::name<category::amd_smi_umc_busy>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = trace_cache::PERCENTAGE,
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_mm_busy>::value,
              .symbol           = "MM Busy",
              .description      = trait::name<category::amd_smi_mm_busy>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = trace_cache::PERCENTAGE,
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_temp>::value,
              .symbol           = "Temp",
              .description      = trait::name<category::amd_smi_temp>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = CELSIUS_DEGREES,
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_power>::value,
              .symbol           = "Pow",
              .description      = trait::name<category::amd_smi_power>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "W",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_memory_usage>::value,
              .symbol           = "MemUsg",
              .description = trait::name<category::amd_smi_memory_usage>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "MB",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
        {
            auto const vcn_name =
                trace_cache::info::format_track_name<category::amd_smi_vcn_activity>(vcn);

            trace_cache::get_metadata_registry().add_pmc_info(
                { .type             = agent_type::gpu,
                  .agent_type_index = gpu_id,
                  .target_arch      = TARGET_ARCH,
                  .event_code       = EVENT_CODE,
                  .instance_id      = INSTANCE_ID,
                  .name             = vcn_name,
                  .symbol           = vcn_name,
                  .description      = "VCN (Video Decode) Engine Activity",
                  .long_description = LONG_DESCRIPTION,
                  .component        = COMPONENT,
                  .units            = trace_cache::PERCENTAGE,
                  .value_type       = rocprofsys::trace_cache::ABSOLUTE,
                  .block            = BLOCK,
                  .expression       = EXPRESSION,
                  .is_constant      = 0,
                  .is_derived       = 0,
                  .extdata          = "{}" });
        }

        for(size_t jpeg = 0; jpeg < MAX_NUM_JPEG; ++jpeg)
        {
            auto const jpeg_name =
                trace_cache::info::format_track_name<category::amd_smi_jpeg_activity>(
                    jpeg);

            trace_cache::get_metadata_registry().add_pmc_info(
                { .type             = agent_type::gpu,
                  .agent_type_index = gpu_id,
                  .target_arch      = TARGET_ARCH,
                  .event_code       = EVENT_CODE,
                  .instance_id      = INSTANCE_ID,
                  .name             = jpeg_name,
                  .symbol           = jpeg_name,
                  .description      = "JPEG (Image Decode) Engine Activity",
                  .long_description = LONG_DESCRIPTION,
                  .component        = COMPONENT,
                  .units            = trace_cache::PERCENTAGE,
                  .value_type       = rocprofsys::trace_cache::ABSOLUTE,
                  .block            = BLOCK,
                  .expression       = EXPRESSION,
                  .is_constant      = 0,
                  .is_derived       = 0,
                  .extdata          = "{}" });
        }

        for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
        {
            for(size_t vcn = 0; vcn < MAX_NUM_VCN; ++vcn)
            {
                auto const vcn_name =
                    trace_cache::info::format_track_name<category::amd_smi_vcn_activity>(
                        xcp, vcn);

                trace_cache::get_metadata_registry().add_pmc_info(
                    { .type             = agent_type::gpu,
                      .agent_type_index = gpu_id,
                      .target_arch      = TARGET_ARCH,
                      .event_code       = EVENT_CODE,
                      .instance_id      = INSTANCE_ID,
                      .name             = vcn_name,
                      .symbol           = vcn_name,
                      .description      = "VCN (Video Decode) Engine Activity",
                      .long_description = LONG_DESCRIPTION,
                      .component        = COMPONENT,
                      .units            = trace_cache::PERCENTAGE,
                      .value_type       = rocprofsys::trace_cache::ABSOLUTE,
                      .block            = BLOCK,
                      .expression       = EXPRESSION,
                      .is_constant      = 0,
                      .is_derived       = 0,
                      .extdata          = "{}" });
            }
        }

        for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
        {
            for(size_t jpeg = 0; jpeg < MAX_NUM_JPEG_V1; ++jpeg)
            {
                auto const jpeg_name =
                    trace_cache::info::format_track_name<category::amd_smi_jpeg_activity>(
                        xcp, jpeg);
                trace_cache::get_metadata_registry().add_pmc_info(
                    { .type             = agent_type::gpu,
                      .agent_type_index = gpu_id,
                      .target_arch      = TARGET_ARCH,
                      .event_code       = EVENT_CODE,
                      .instance_id      = INSTANCE_ID,
                      .name             = jpeg_name,
                      .symbol           = jpeg_name,
                      .description      = "JPEG (Image Decode) Engine Activity",
                      .long_description = LONG_DESCRIPTION,
                      .component        = COMPONENT,
                      .units            = trace_cache::PERCENTAGE,
                      .value_type       = rocprofsys::trace_cache::ABSOLUTE,
                      .block            = BLOCK,
                      .expression       = EXPRESSION,
                      .is_constant      = 0,
                      .is_derived       = 0,
                      .extdata          = "{}" });
            }
        }

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_sdma_usage>::value,
              .symbol           = "SDMA Usage",
              .description      = trait::name<category::amd_smi_sdma_usage>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = trace_cache::PERCENTAGE,
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_gfx_clock>::value,
              .symbol           = "GFX Clock",
              .description      = trait::name<category::amd_smi_gfx_clock>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "MHz",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_mem_clock>::value,
              .symbol           = "Mem Clock",
              .description      = trait::name<category::amd_smi_mem_clock>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "MHz",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_xgmi_link_width>::value,
              .symbol           = "XGMI Width",
              .description = trait::name<category::amd_smi_xgmi_link_width>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "lanes",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_xgmi_link_speed>::value,
              .symbol           = "XGMI Speed",
              .description = trait::name<category::amd_smi_xgmi_link_speed>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "Mbps",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        // XGMI data accumulators are reported per-link, so one PMC per link is needed
        for(size_t link = 0; link < MAX_NUM_XGMI_LINKS; ++link)
        {
            auto const read_name = trace_cache::info::format_link_pmc_name(
                trait::name<category::amd_smi_xgmi_read_data>::value, link);
            trace_cache::get_metadata_registry().add_pmc_info(
                { .type             = agent_type::gpu,
                  .agent_type_index = gpu_id,
                  .target_arch      = TARGET_ARCH,
                  .event_code       = EVENT_CODE,
                  .instance_id      = INSTANCE_ID,
                  .name             = read_name,
                  .symbol           = fmt::format("XGMI Read [Link {}]", link),
                  .description =
                      trait::name<category::amd_smi_xgmi_read_data>::description,
                  .long_description = LONG_DESCRIPTION,
                  .component        = COMPONENT,
                  .units            = "KB",
                  .value_type       = rocprofsys::trace_cache::ABSOLUTE,
                  .block            = BLOCK,
                  .expression       = EXPRESSION,
                  .is_constant      = 0,
                  .is_derived       = 0,
                  .extdata          = "{}" });

            auto const write_name = trace_cache::info::format_link_pmc_name(
                trait::name<category::amd_smi_xgmi_write_data>::value, link);
            trace_cache::get_metadata_registry().add_pmc_info(
                { .type             = agent_type::gpu,
                  .agent_type_index = gpu_id,
                  .target_arch      = TARGET_ARCH,
                  .event_code       = EVENT_CODE,
                  .instance_id      = INSTANCE_ID,
                  .name             = write_name,
                  .symbol           = fmt::format("XGMI Write [Link {}]", link),
                  .description =
                      trait::name<category::amd_smi_xgmi_write_data>::description,
                  .long_description = LONG_DESCRIPTION,
                  .component        = COMPONENT,
                  .units            = "KB",
                  .value_type       = rocprofsys::trace_cache::ABSOLUTE,
                  .block            = BLOCK,
                  .expression       = EXPRESSION,
                  .is_constant      = 0,
                  .is_derived       = 0,
                  .extdata          = "{}" });
        }

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_pcie_link_width>::value,
              .symbol           = "PCIe Width",
              .description = trait::name<category::amd_smi_pcie_link_width>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "lanes",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name             = trait::name<category::amd_smi_pcie_link_speed>::value,
              .symbol           = "PCIe Speed",
              .description = trait::name<category::amd_smi_pcie_link_speed>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "MT/s",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name   = trait::name<category::amd_smi_pcie_bandwidth_acc>::value,
              .symbol = "PCIe BW Acc",
              .description =
                  trait::name<category::amd_smi_pcie_bandwidth_acc>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "bytes",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { .type             = agent_type::gpu,
              .agent_type_index = gpu_id,
              .target_arch      = TARGET_ARCH,
              .event_code       = EVENT_CODE,
              .instance_id      = INSTANCE_ID,
              .name   = trait::name<category::amd_smi_pcie_bandwidth_inst>::value,
              .symbol = "PCIe BW Inst",
              .description =
                  trait::name<category::amd_smi_pcie_bandwidth_inst>::description,
              .long_description = LONG_DESCRIPTION,
              .component        = COMPONENT,
              .units            = "bytes/s",
              .value_type       = rocprofsys::trace_cache::ABSOLUTE,
              .block            = BLOCK,
              .expression       = EXPRESSION,
              .is_constant      = 0,
              .is_derived       = 0,
              .extdata          = "{}" });
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
                             const metrics& metric_values, std::uint64_t timestamp)
    {
        enabled_metrics _enabled_metrics;
        _enabled_metrics.value = enabled_metrics_cfg.value & supported_metrics.value;

        trace_cache::get_buffer_storage().store(trace_cache::gpu_pmc_sample{
            _enabled_metrics, static_cast<std::uint32_t>(device_id), timestamp,
            metric_values });
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu

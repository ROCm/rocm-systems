#pragma once
#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cache_utility.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/amd_smi/common.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <ostream>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{

struct rocpd
{
    static void initialize_smi_tracks_metadata(size_t gpu_id)
    {
        if(!get_use_rocpd())
        {
            return;
        }

        const auto thread_id = std::nullopt;
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::annotate_with_device_id<category::amd_smi_gfx_busy>(
                  gpu_id),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::annotate_with_device_id<category::amd_smi_umc_busy>(
                  gpu_id),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::annotate_with_device_id<category::amd_smi_mm_busy>(
                  gpu_id),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::annotate_with_device_id<category::amd_smi_power>(gpu_id),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::annotate_with_device_id<category::amd_smi_temp>(gpu_id),
              thread_id, "{}" });
        trace_cache::get_metadata_registry().add_track(
            { trace_cache::info::annotate_with_device_id<category::amd_smi_memory_usage>(
                  gpu_id),
              thread_id, "{}" });

        auto add_vcn_track = [&](std::optional<int> xcp_idx) {
            for(auto clk = 0; clk < AMDSMI_MAX_NUM_VCN; ++clk)
            {
                auto name = trace_cache::info::annotate_with_device_id<
                    category::amd_smi_vcn_activity>(gpu_id, xcp_idx, clk);
                trace_cache::get_metadata_registry().add_track(
                    { name.c_str(), thread_id, "{}" });
            }
        };

        auto add_jpeg_track = [&](std::optional<int> xcp_idx) {
            for(auto clk = 0; clk < AMDSMI_MAX_NUM_JPEG_ENGINES; ++clk)
            {
                auto name = trace_cache::info::annotate_with_device_id<
                    category::amd_smi_jpeg_activity>(gpu_id, xcp_idx, clk);
                trace_cache::get_metadata_registry().add_track(
                    { name.c_str(), thread_id, "{}" });
            }
        };

        for(int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            add_vcn_track(xcp);
        }

        for(auto xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            add_jpeg_track(xcp);
        }
    }

    static void initialize_smi_pmc_metadata(size_t gpu_id)
    {
        // TODO: Find the proper values for a following definitions
        size_t      EVENT_CODE       = 0;
        size_t      INSTANCE_ID      = 0;
        const char* LONG_DESCRIPTION = "";
        const char* COMPONENT        = "";
        const char* BLOCK            = "";
        const char* EXPRESSION       = "";
        const char* CELSIUS_DEGREES  = "\u00B0C";
        auto        ni               = node_info::get_instance();
        const char* TARGET_ARCH      = "GPU";

        if(!get_use_rocpd())
        {
            return;
        }

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

        auto add_vcn_pmc = [&](std::optional<int> xcp_idx) {
            for(int clk = 0; clk < AMDSMI_MAX_NUM_VCN; ++clk)
            {
                std::stringstream name_ss;
                name_ss << trait::name<category::amd_smi_vcn_activity>::value;
                if(xcp_idx) name_ss << "_" << *xcp_idx;
                name_ss << "_" << clk;

                std::stringstream symbol_ss;
                symbol_ss << "VcnAct";
                if(xcp_idx) symbol_ss << "_" << *xcp_idx;
                symbol_ss << "_" << clk;

                trace_cache::get_metadata_registry().add_pmc_info(
                    { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                      name_ss.str(), symbol_ss.str(),
                      trait::name<category::amd_smi_vcn_activity>::description,
                      LONG_DESCRIPTION, COMPONENT, trace_cache::PERCENTAGE,
                      rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
            }
        };

        auto add_jpeg_pmc = [&](std::optional<int> xcp_idx) {
            for(auto clk = 0; clk < AMDSMI_MAX_NUM_JPEG_ENGINES; ++clk)
            {
                std::stringstream name_ss;
                name_ss << trait::name<category::amd_smi_jpeg_activity>::value;
                if(xcp_idx) name_ss << "_" << *xcp_idx;
                name_ss << "_" << std::to_string(clk);

                std::stringstream symbol_ss;
                symbol_ss << "JpegAct";
                if(xcp_idx) symbol_ss << "_" << *xcp_idx;
                symbol_ss << "_" << std::to_string(clk);

                trace_cache::get_metadata_registry().add_pmc_info(
                    { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                      name_ss.str(), symbol_ss.str(),
                      trait::name<category::amd_smi_jpeg_activity>::description,
                      LONG_DESCRIPTION, COMPONENT, trace_cache::PERCENTAGE,
                      rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
            }
        };

        for(int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            add_vcn_pmc(xcp);
            add_jpeg_pmc(xcp);
        }
    }

    static void initialize_category_metadata()
    {
        if(!get_use_rocpd())
        {
            return;
        }
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::amd_smi>::value);
    }

    static void store_sample(size_t                    _device_id,
                             const smi_metric_options& _supported_metrics,
                             const smi_metric_options& _enabled_metrics,
                             const smi_metrics& _smi_metrics, unsigned long _timestamp)
    {
        if(!get_use_rocpd())
        {
            return;
        }

        trace_cache::get_buffer_storage().store(
            trace_cache::entry_type::amd_smi_sample, _device_id, _timestamp,
            (uint32_t) (_enabled_metrics.value & _supported_metrics.value),
            serialize_smi_metrics(_smi_metrics));
    };

private:
    static std::vector<uint8_t> serialize_smi_metrics(const smi_metrics& gpu_metrics)
    {
        auto                 metric_size = sizeof(gpu_metrics);
        std::vector<uint8_t> result(metric_size);
        std::memcpy(result.data(), &gpu_metrics, metric_size);
        return result;
    }
};

}  // namespace amd_smi
}  // namespace rocprofsys

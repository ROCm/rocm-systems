#pragma once
#include "core/config.hpp"
#include "core/gpu.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cache_utility.hpp"
#include "core/trace_cache/metadata_registry.hpp"

#include <cstddef>
#include <optional>

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
            for(auto clk = 0; clk < AMDSMI_MAX_NUM_JPEG; ++clk)
            {
                auto name = trace_cache::info::annotate_with_device_id<
                    category::amd_smi_jpeg_activity>(gpu_id, xcp_idx, clk);
                trace_cache::get_metadata_registry().add_track(
                    { name.c_str(), thread_id, "{}" });
            }
        };

        if(gpu::is_vcn_activity_supported(gpu_id))
        {
            add_vcn_track(std::nullopt);
        }
        else
        {
            for(int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                add_vcn_track(xcp);
            }
        }

        if(gpu::is_jpeg_activity_supported(gpu_id))
        {
            add_jpeg_track(std::nullopt);
        }
        else
        {
            for(auto xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                add_jpeg_track(xcp);
            }
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
            for(auto clk = 0; clk < AMDSMI_MAX_NUM_JPEG; ++clk)
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

        if(gpu::is_vcn_activity_supported(gpu_id))
        {
            add_vcn_pmc(std::nullopt);
        }
        else
        {
            for(int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                add_vcn_pmc(xcp);
            }
        }

        if(gpu::is_jpeg_activity_supported(gpu_id))
        {
            add_jpeg_pmc(std::nullopt);
        }
        else
        {
            for(auto xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                add_jpeg_pmc(xcp);
            }
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

        auto _power = _supported_metrics.bits.current_socket_power
                          ? _smi_metrics.current_socket_power
                          : _smi_metrics.average_socket_power;
        auto _temp  = _supported_metrics.bits.hotspot_temperature
                          ? static_cast<int32_t>(_smi_metrics.hotspot_temperature)
                          : static_cast<int32_t>(_smi_metrics.edge_temperature);

        trace_cache::get_buffer_storage().store(
            trace_cache::entry_type::amd_smi_sample,
            static_cast<size_t>(_enabled_metrics.value), _device_id, _timestamp,
            _smi_metrics.gfx_activity, _smi_metrics.umc_activity,
            _smi_metrics.mm_activity, _power, _temp, _smi_metrics.memory_usage,
            std::vector<uint8_t>(40));
    };

private:
    static std::vector<uint8_t> serialize_xcp_metrics(
        const bool& use_vcn_activity, const bool& use_jpeg_activity,
        const amdsmi_gpu_metrics_t& gpu_metrics)
    {
        // Chunk:
        // <vcn_data_0>..<vcn_data_[vcn_count]> // lower and higher byte
        // <jpeg_data_0>..<jpeg_data_[jpeg_count]> // lower and higher byte

        // Serialized:
        // <is_vcn_supported>
        // <is_jpeg_supported>
        // <xcp_count>
        // <vcn_count>
        // <jpeg_count>
        // Chunk_0
        // ...
        // Chunk_[xcp_count]

        constexpr uint8_t vcn_count          = AMDSMI_MAX_NUM_VCN;
        constexpr uint8_t jpeg_count         = AMDSMI_MAX_NUM_JPEG;
        constexpr uint8_t xcp_count          = AMDSMI_MAX_NUM_XCP;
        constexpr size_t  elem_size          = sizeof(uint16_t) / sizeof(uint8_t);
        constexpr uint8_t vector_size_header = sizeof(uint8_t);
        constexpr uint8_t serialized_data_headers =
            5 * vector_size_header;  // is_vcn_supported + is_jpeg_supported + xcp_count +
                                     // vcn_count + jpeg_count
        constexpr size_t chunk_size = ((vcn_count + jpeg_count) * elem_size);

        auto serialize_uint16_array = [](std::vector<uint8_t>& data, const uint16_t* arr,
                                         int array_size) {
            for(int i = 0; i < array_size; ++i)
            {
                data.push_back(static_cast<uint8_t>(arr[i] & 0xFF));
                data.push_back(static_cast<uint8_t>((arr[i] >> 8) & 0xFF));
            }
        };

        std::vector<uint8_t> result;

        const bool   is_vcn_jpeg_supported = (use_vcn_activity || use_jpeg_activity);
        const size_t chunk_count           = is_vcn_jpeg_supported ? 1 : xcp_count;
        const size_t total_size = serialized_data_headers + (chunk_count * chunk_size);

        result.reserve(total_size);

        result.push_back((uint8_t) use_vcn_activity);
        result.push_back((uint8_t) use_jpeg_activity);
        result.push_back(chunk_count);
        result.push_back(vcn_count);
        result.push_back(jpeg_count);

        for(size_t count = 0; count < chunk_count; ++count)
        {
            const auto* vcn_data =
                (is_vcn_jpeg_supported ? gpu_metrics.vcn_activity
                                       : gpu_metrics.xcp_stats[count].vcn_busy);
            const auto* jpeg_data =
                (is_vcn_jpeg_supported ? gpu_metrics.jpeg_activity
                                       : gpu_metrics.xcp_stats[count].jpeg_busy);

            serialize_uint16_array(result, vcn_data, vcn_count);
            serialize_uint16_array(result, jpeg_data, jpeg_count);
        }

        return result;
    }
};

}  // namespace amd_smi
}  // namespace rocprofsys

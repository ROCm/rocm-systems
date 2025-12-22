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
#include "library/amd_smi/common.hpp"

#include <timemory/units.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{

#if ROCPROFSYS_USE_ROCM > 0

struct rocpd_policy
{
    static void initialize_category_metadata()
    {
        if(!get_use_rocpd())
        {
            return;
        }
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::amd_smi>::value);
    }

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
            for(int clk = 0; clk < AMDSMI_MAX_NUM_VCN; ++clk)
            {
                auto name = trace_cache::info::annotate_with_device_id<
                    category::amd_smi_vcn_activity>(gpu_id, xcp_idx, clk);
                trace_cache::get_metadata_registry().add_track(
                    { name.c_str(), thread_id, "{}" });
            }
        };

        auto add_jpeg_track = [&](std::optional<int> xcp_idx) {
            for(int clk = 0; clk < ROCPROFSYS_MAX_NUM_JPEG_ENGINES; ++clk)
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
            add_jpeg_track(xcp);
        }
    }

    static void initialize_smi_pmc_metadata(size_t gpu_id)
    {
        if(!get_use_rocpd())
        {
            return;
        }

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
    }

    static void store_sample(size_t device_id, const enabled_metric& supported_metrics,
                             const enabled_metric& enabled_metrics,
                             const smi_metrics& metrics, unsigned long timestamp)
    {
        if(!get_use_rocpd())
        {
            return;
        }
        enabled_metric _enabled_metrics = { .value = enabled_metrics.value &
                                                     supported_metrics.value };

        trace_cache::get_buffer_storage().store(trace_cache::amd_smi_sample{
            _enabled_metrics, static_cast<uint32_t>(device_id), timestamp, metrics });
    }
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace amd_smi
}  // namespace rocprofsys

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
#include "core/perfetto.hpp"
#include "library/amd_smi/common.hpp"
#include "library/thread_info.hpp"

#include <mutex>
#include <timemory/units.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{

#if ROCPROFSYS_USE_ROCM > 0

namespace
{

struct track_description
{
    const char*         track_name;
    const char*         units;
    std::vector<size_t> track_indexes;
};

const auto GFX_BUSY_VALUE = enabled_metric{ .bits{ .gfx_activity = 1 } }.value;
const auto UMC_BUSY_VALUE = enabled_metric{ .bits{ .umc_activity = 1 } }.value;
const auto MM_BUSY_VALUE  = enabled_metric{ .bits{ .mm_activity = 1 } }.value;
const auto TEMPERATURE_VALUE =
    enabled_metric{ .bits{ .hotspot_temperature = 1, .edge_temperature = 1 } }.value;
const auto CURRENT_POWER_VALUE =
    enabled_metric{ .bits{ .current_socket_power = 1, .average_socket_power = 1 } }.value;
const auto MEMORY_USAGE_VALUE  = enabled_metric{ .bits{ .memory_usage = 1 } }.value;
const auto VCN_ACTIVITY_VALUE  = enabled_metric{ .bits{ .vcn_activity = 1 } }.value;
const auto JPEG_ACTIVITY_VALUE = enabled_metric{ .bits{ .jpeg_activity = 1 } }.value;

inline std::unordered_map<uint32_t, track_description>&
get_perfetto_tracks()
{
    static std::unordered_map<uint32_t, track_description> tracks{
        { GFX_BUSY_VALUE, { "GFX Busy", "%", {} } },
        { UMC_BUSY_VALUE, { "UMC Busy", "%", {} } },
        { MM_BUSY_VALUE, { "MM Busy", "%", {} } },
        { TEMPERATURE_VALUE, { "Temperature", "deg C", {} } },
        { CURRENT_POWER_VALUE, { "Current Power", "watts", {} } },
        { MEMORY_USAGE_VALUE, { "Memory Usage", "megabytes", {} } },
        { VCN_ACTIVITY_VALUE, { "VCN Activity", "%", {} } },
        { JPEG_ACTIVITY_VALUE, { "JPEG Activity", "%", {} } },
    };
    return tracks;
}

struct perfetto_amd_smi_sample
{
    size_t      timestamp;
    smi_metrics metrics;
};

inline std::map<size_t, std::unique_ptr<std::vector<perfetto_amd_smi_sample>>>&
get_perfetto_bundle()
{
    static std::map<size_t, std::unique_ptr<std::vector<perfetto_amd_smi_sample>>> bundle;
    return bundle;
}

}  // namespace

struct perfetto_policy
{
    static void init_storage(size_t device_index)
    {
        get_perfetto_bundle().insert(
            { device_index, std::make_unique<std::vector<perfetto_amd_smi_sample>>() });
    }

    static void setup_counter_tracks(size_t                device_index,
                                     const enabled_metric& enabled_metrics)
    {
        if(!get_use_perfetto())
        {
            return;
        }

        using counter_track = perfetto_counter_track<smi_metrics>;

        auto addendum = [&](const char* name) {
            return JOIN(" ", "GPU", name, JOIN("", '[', device_index, ']'), "(S)");
        };

        auto addendum_blk = [&](std::size_t i, const char* metric,
                                std::size_t xcp_idx = SIZE_MAX) {
            if(xcp_idx != SIZE_MAX)
            {
                return JOIN(" ", "GPU", JOIN("", '[', device_index, ']'), metric,
                            JOIN("", "XCP_", xcp_idx, ": [", (i < 10 ? "0" : ""), i, ']'),
                            "(S)");
            }
            return JOIN(" ", "GPU", JOIN("", '[', device_index, ']'), metric,
                        JOIN("", "[", (i < 10 ? "0" : ""), i, ']'), "(S)");
        };

        auto& tracks = get_perfetto_tracks();

        for(auto& [num, description] : tracks)
        {
            auto enabled_metric = num & enabled_metrics.value;
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

            if(enabled_metric == VCN_ACTIVITY_VALUE ||
               enabled_metric == JPEG_ACTIVITY_VALUE)
            {
                for(std::size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
                {
                    process_xcp_array(description,
                                      enabled_metric == VCN_ACTIVITY_VALUE
                                          ? AMDSMI_MAX_NUM_VCN
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
    }

    static void store_sample(size_t device_index, const smi_metrics& metrics,
                             unsigned long timestamp)
    {
        if(get_use_perfetto())
        {
            get_perfetto_bundle()[device_index]->emplace_back(
                perfetto_amd_smi_sample{ timestamp, metrics });
        }
    }

    static void post_process(size_t device_index, enabled_metric enabled_metrics,
                             enabled_metric supported_metrics)
    {
        if(!get_use_perfetto())
        {
            return;
        }

        using counter_track = perfetto_counter_track<smi_metrics>;

        auto&       samples      = *get_perfetto_bundle()[device_index];
        const auto& _thread_info = thread_info::get(0, InternalTID);

        ROCPROFSYS_VERBOSE(1, "Post-processing %zu amd-smi samples from device %zu\n",
                           samples.size(), device_index);

        if(!_thread_info)
        {
            return;
        }

        enabled_metric _enabled_metrics = {
            .value =
                static_cast<uint32_t>(enabled_metrics.value & supported_metrics.value)
        };

        auto& tracks = get_perfetto_tracks();

        for(auto& itr : samples)
        {
            const auto _ts = itr.timestamp;

            if(!_thread_info->is_valid_time(_ts))
            {
                continue;
            }

            const double _gfxbusy = itr.metrics.gfx_activity;
            const double _umcbusy = itr.metrics.umc_activity;
            const double _mmbusy  = itr.metrics.mm_activity;
            const double _temp    = _enabled_metrics.bits.hotspot_temperature
                                        ? itr.metrics.hotspot_temperature
                                        : itr.metrics.edge_temperature;
            const double _power   = _enabled_metrics.bits.average_socket_power
                                        ? itr.metrics.average_socket_power
                                        : itr.metrics.current_socket_power;
            const double _usage =
                itr.metrics.memory_usage / static_cast<double>(units::megabyte);

            if(_enabled_metrics.bits.gfx_activity &&
               !tracks.at(GFX_BUSY_VALUE).track_indexes.empty())
            {
                const auto track_index = tracks.at(GFX_BUSY_VALUE).track_indexes[0];
                TRACE_COUNTER("device_busy_gfx",
                              counter_track::at(device_index, track_index), _ts,
                              _gfxbusy);
            }
            if(_enabled_metrics.bits.umc_activity &&
               !tracks.at(UMC_BUSY_VALUE).track_indexes.empty())
            {
                const auto track_index = tracks.at(UMC_BUSY_VALUE).track_indexes[0];
                TRACE_COUNTER("device_busy_umc",
                              counter_track::at(device_index, track_index), _ts,
                              _umcbusy);
            }
            if(_enabled_metrics.bits.mm_activity &&
               !tracks.at(MM_BUSY_VALUE).track_indexes.empty())
            {
                const auto track_index = tracks.at(MM_BUSY_VALUE).track_indexes[0];
                TRACE_COUNTER("device_busy_mm",
                              counter_track::at(device_index, track_index), _ts, _mmbusy);
            }
            if((_enabled_metrics.bits.edge_temperature ||
                _enabled_metrics.bits.hotspot_temperature) &&
               !tracks.at(TEMPERATURE_VALUE).track_indexes.empty())
            {
                const auto track_index = tracks.at(TEMPERATURE_VALUE).track_indexes[0];
                TRACE_COUNTER("device_temp", counter_track::at(device_index, track_index),
                              _ts, _temp);
            }
            if((_enabled_metrics.bits.average_socket_power ||
                _enabled_metrics.bits.current_socket_power) &&
               !tracks.at(CURRENT_POWER_VALUE).track_indexes.empty())
            {
                const auto track_index = tracks.at(CURRENT_POWER_VALUE).track_indexes[0];
                TRACE_COUNTER("device_power",
                              counter_track::at(device_index, track_index), _ts, _power);
            }
            if(_enabled_metrics.bits.memory_usage &&
               !tracks.at(MEMORY_USAGE_VALUE).track_indexes.empty())
            {
                const auto track_index = tracks.at(MEMORY_USAGE_VALUE).track_indexes[0];
                TRACE_COUNTER("device_memory_usage",
                              counter_track::at(device_index, track_index), _ts, _usage);
            }

            if(_enabled_metrics.bits.vcn_activity &&
               !tracks.at(VCN_ACTIVITY_VALUE).track_indexes.empty())
            {
                size_t engine_id = 0;
                for(const auto& xcp_stats : itr.metrics.xcp_stats)
                {
                    for(const auto& vcn_val : xcp_stats.vcn_busy)
                    {
                        if(vcn_val != std::numeric_limits<uint16_t>::max() &&
                           engine_id < tracks.at(VCN_ACTIVITY_VALUE).track_indexes.size())
                        {
                            const auto track_index =
                                tracks.at(VCN_ACTIVITY_VALUE).track_indexes[engine_id++];
                            TRACE_COUNTER("device_vcn_activity",
                                          counter_track::at(device_index, track_index),
                                          _ts, vcn_val);
                        }
                    }
                }
            }

            std::once_flag once_flag;
            std::call_once(once_flag, [&]() {
                printf("JPEG activity: %d, enabled: %d, supported: %d\n",
                       _enabled_metrics.bits.jpeg_activity,
                       enabled_metrics.bits.jpeg_activity,
                       supported_metrics.bits.jpeg_activity);
            });

            if(_enabled_metrics.bits.jpeg_activity &&
               !tracks.at(JPEG_ACTIVITY_VALUE).track_indexes.empty())
            {
                size_t engine_id = 0;
                for(const auto& xcp_stats : itr.metrics.xcp_stats)
                {
                    for(const auto& jpeg_val : xcp_stats.jpeg_busy)
                    {
                        if(jpeg_val != std::numeric_limits<uint16_t>::max() &&
                           engine_id <
                               tracks.at(JPEG_ACTIVITY_VALUE).track_indexes.size())
                        {
                            const auto track_index =
                                tracks.at(JPEG_ACTIVITY_VALUE).track_indexes[engine_id++];
                            TRACE_COUNTER("device_jpeg_activity",
                                          counter_track::at(device_index, track_index),
                                          _ts, jpeg_val);
                        }
                    }
                }
            }
        }
    }
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace amd_smi
}  // namespace rocprofsys

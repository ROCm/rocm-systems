#pragma once

#include "core/config.hpp"
#include "core/perfetto.hpp"
#include "library/amd_smi/common.hpp"
#include "library/thread_info.hpp"

#include <unordered_map>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{

namespace
{

/*
Perfetto order
    0. GPU Current Power
    1. GPU GFX Busy
    2. GPU MM Busy
    3. GPU Memory Usage
    4. GPU Temperature
    5. GPU UMC Busy
*/
struct track_description
{
    const char* const   track_name;
    const char* const   units;
    std::vector<size_t> track_indexes;
};

const auto GFX_BUSY_TEST_VALUE = smi_metric_options{ .bits{ .gfx_activity = 1 } }.value;
const auto UMC_BUSY_TEST_VALUE = smi_metric_options{ .bits{ .umc_activity = 1 } }.value;
const auto MM_BUSY_TEST_VALUE  = smi_metric_options{ .bits{ .mm_activity = 1 } }.value;
const auto TEMPERATURE_TEST_VALUE =
    smi_metric_options{ .bits{ .hotspot_temperature = 1, .edge_temperature = 1 } }.value;
const auto CURRENT_POWER_TEST_VALUE =
    smi_metric_options{ .bits{ .current_socket_power = 1, .average_socket_power = 1 } }
        .value;
const auto MEMORY_USAGE_TEST_VALUE =
    smi_metric_options{ .bits{ .memory_usage = 1 } }.value;
const auto VCN_ACTIVITY_TEST_VALUE =
    smi_metric_options{ .bits{ .vcn_activity = 1 } }.value;
const auto JPEG_ACTIVITY_TEST_VALUE =
    smi_metric_options{ .bits{ .jpeg_activity = 1 } }.value;

std::unordered_map<uint32_t, track_description> perfetto_track_info{
    { GFX_BUSY_TEST_VALUE, { "GFX Busy", "%" } },
    { UMC_BUSY_TEST_VALUE, { "UMC Busy", "%" } },
    { MM_BUSY_TEST_VALUE, { "MM Busy", "%" } },
    { TEMPERATURE_TEST_VALUE, { "Temperature", "deg C" } },
    { CURRENT_POWER_TEST_VALUE, { "Current Power", "watts" } },
    { MEMORY_USAGE_TEST_VALUE, { "Memory Usage", "megabytes" } },
    { VCN_ACTIVITY_TEST_VALUE, { "VCN Activity", "%" } },
    { JPEG_ACTIVITY_TEST_VALUE, { "JPEG Activity", "%" } },
};

struct amd_smi_sample
{
    size_t      timestamp;
    smi_metrics metrics;
};

// TODO: we are keeping this just to support the old perfetto way
std::map<size_t, std::unique_ptr<std::vector<amd_smi_sample>>> g_perfetto_bundle;

}  // namespace

struct perfetto
{
    static void setup_counter_tracks(const size_t              device_index,
                                     const smi_metric_options& metrics)
    {
        if(!get_use_perfetto())
        {
            return;
        }

        using counter_track = perfetto_counter_track<smi_metrics>;

        auto addendum = [&](const char* _v) {
            return JOIN(" ", "GPU", _v, JOIN("", '[', device_index, ']'), "(S)");
        };

        auto addendum_blk = [&](std::size_t _i, const char* _metric,
                                std::size_t xcp_idx = SIZE_MAX) {
            if(xcp_idx != SIZE_MAX)
            {
                auto result =
                    JOIN(" ", "GPU", JOIN("", '[', device_index, ']'), _metric,
                         JOIN("", "XCP_", xcp_idx, ": [", (_i < 10 ? "0" : ""), _i, ']'),
                         "(S)");
                // std::cout << result << std::endl;
                return result;
            }
            else
            {
                return JOIN(" ", "GPU", JOIN("", '[', device_index, ']'), _metric,
                            JOIN("", "[", (_i < 10 ? "0" : ""), _i, ']'), "(S)");
            }
        };

        auto index{ 0 };

        for(auto& [num, description] : perfetto_track_info)
        {
            auto metric_map_id = num & metrics.value;
            if(metric_map_id > 0)
            {
                if(metric_map_id ==
                   smi_metric_options{ .bits{ .vcn_activity = 1 } }.value)
                {
                    for(std::size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
                    {
                        for(std::size_t i = 0; i < AMDSMI_MAX_NUM_VCN; ++i)
                        {
                            description.track_indexes.emplace_back(counter_track::emplace(
                                device_index,
                                addendum_blk(i, description.track_name, xcp),
                                description.units));
                        }
                    }
                }
                else if(metric_map_id ==
                        smi_metric_options{ .bits{ .jpeg_activity = 1 } }.value)
                {
                    for(std::size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
                    {
                        for(std::size_t i = 0; i < AMDSMI_MAX_NUM_JPEG_ENGINES; ++i)
                        {
                            description.track_indexes.emplace_back(counter_track::emplace(
                                device_index,
                                addendum_blk(i, description.track_name, xcp),
                                description.units));
                        }
                    }
                }
                else
                {
                    description.track_indexes.emplace_back(counter_track::emplace(
                        device_index, addendum(description.track_name),
                        description.units));
                }
            }
        }
    };

    static void init_storage(size_t device_index)
    {
        g_perfetto_bundle.insert(
            { device_index, std::make_unique<std::vector<amd_smi_sample>>() });
    };

    static void store_sample(size_t device_index, const smi_metrics& _smi_metrics,
                             unsigned long _timestamp)
    {
        if(get_use_perfetto())
        {
            g_perfetto_bundle[device_index]->emplace_back(
                amd_smi_sample{ _timestamp, _smi_metrics });
        }
    };

    static void post_process(size_t device_index, smi_metric_options enabled_metrics,
                             smi_metric_options supported_metrics)
    {
        if(!get_use_perfetto())
        {
            return;
        }

        using counter_track = perfetto_counter_track<smi_metrics>;

        auto        _amd_smi     = *g_perfetto_bundle[device_index];
        const auto& _thread_info = thread_info::get(0, InternalTID);

        ROCPROFSYS_VERBOSE(1, "Post-processing %zu amd-smi samples from device %zu\n",
                           _amd_smi.size(), device_index);

        ROCPROFSYS_CI_THROW(!_thread_info, "Missing thread info for thread 0");
        if(!_thread_info) return;

        smi_metric_options _enabled_metrics = { .value = enabled_metrics.value &
                                                         supported_metrics.value };

        for(auto& itr : _amd_smi)
        {
            uint64_t _ts = itr.timestamp;
            if(!_thread_info->is_valid_time(_ts)) continue;

            double _gfxbusy = itr.metrics.gfx_activity;
            double _umcbusy = itr.metrics.umc_activity;
            double _mmbusy  = itr.metrics.mm_activity;
            double _temp    = _enabled_metrics.bits.hotspot_temperature
                                  ? itr.metrics.hotspot_temperature
                                  : itr.metrics.edge_temperature;
            double _power   = _enabled_metrics.bits.average_socket_power
                                  ? itr.metrics.average_socket_power
                                  : itr.metrics.current_socket_power;
            double _usage =
                itr.metrics.memory_usage / static_cast<double>(units::megabyte);

            if(_enabled_metrics.bits.gfx_activity)
            {
                const auto& track_index =
                    perfetto_track_info.at(GFX_BUSY_TEST_VALUE).track_indexes[0];
                TRACE_COUNTER("device_busy_gfx",
                              counter_track::at(device_index, track_index), _ts,
                              _gfxbusy);
            }
            if(_enabled_metrics.bits.umc_activity)
            {
                const auto& track_index =
                    perfetto_track_info.at(UMC_BUSY_TEST_VALUE).track_indexes[0];
                TRACE_COUNTER("device_busy_umc",
                              counter_track::at(device_index, track_index), _ts,
                              _umcbusy);
            }
            if(_enabled_metrics.bits.mm_activity)
            {
                const auto& track_index =
                    perfetto_track_info.at(MM_BUSY_TEST_VALUE).track_indexes[0];
                TRACE_COUNTER("device_busy_mm",
                              counter_track::at(device_index, track_index), _ts, _mmbusy);
            }
            if(_enabled_metrics.bits.edge_temperature ||
               _enabled_metrics.bits.hotspot_temperature)
            {
                const auto& track_index =
                    perfetto_track_info.at(TEMPERATURE_TEST_VALUE).track_indexes[0];
                TRACE_COUNTER("device_temp", counter_track::at(device_index, track_index),
                              _ts, _temp);
            }
            if(_enabled_metrics.bits.average_socket_power ||
               _enabled_metrics.bits.current_socket_power)
            {
                const auto& track_index =
                    perfetto_track_info.at(CURRENT_POWER_TEST_VALUE).track_indexes[0];
                TRACE_COUNTER("device_power",
                              counter_track::at(device_index, track_index), _ts, _power);
            }
            if(_enabled_metrics.bits.memory_usage)
            {
                const auto& track_index =
                    perfetto_track_info.at(MEMORY_USAGE_TEST_VALUE).track_indexes[0];
                TRACE_COUNTER("device_memory_usage",
                              counter_track::at(device_index, track_index), _ts, _usage);
            }
            if(_enabled_metrics.bits.vcn_activity)
            {
                std::for_each(
                    std::begin(itr.metrics.xcp_stats), std::end(itr.metrics.xcp_stats),
                    [&, engine_id = 0](const auto& xcp_stats) mutable {
                        std::for_each(
                            std::begin(xcp_stats.vcn_busy), std::end(xcp_stats.vcn_busy),
                            [&](const auto& vcn_val) {
                                if(vcn_val != std::numeric_limits<uint16_t>::max())
                                {
                                    const auto& track_index =
                                        perfetto_track_info.at(VCN_ACTIVITY_TEST_VALUE)
                                            .track_indexes[engine_id++];
                                    TRACE_COUNTER(
                                        "device_vcn_activity",
                                        counter_track::at(device_index, track_index), _ts,
                                        vcn_val);
                                }
                            });
                    });
            }

            if(_enabled_metrics.bits.jpeg_activity)
            {
                std::for_each(
                    std::begin(itr.metrics.xcp_stats), std::end(itr.metrics.xcp_stats),
                    [&, engine_id = 0](const auto& xcp_stats) mutable {
                        std::for_each(
                            std::begin(xcp_stats.jpeg_busy),
                            std::end(xcp_stats.jpeg_busy), [&](const auto& jpeg_val) {
                                if(jpeg_val != std::numeric_limits<uint16_t>::max())
                                {
                                    const auto& track_index =
                                        perfetto_track_info.at(JPEG_ACTIVITY_TEST_VALUE)
                                            .track_indexes[engine_id++];
                                    TRACE_COUNTER(
                                        "device_jpeg_activity",
                                        counter_track::at(device_index, track_index), _ts,
                                        jpeg_val);
                                }
                            });
                    });
            }
        }
    }
};

}  // namespace amd_smi
}  // namespace rocprofsys
